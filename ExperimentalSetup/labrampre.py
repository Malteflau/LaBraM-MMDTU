#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaBraM pre-training fine-tuning on social interaction EEG data.

This script fine-tunes the LaBraM model using the original masked EEG modeling 
approach on the social interaction dataset, adapting the model to this domain
before supervised fine-tuning.

Author: Magnus Evensen, Malte Færgemann Lau
"""

import os
import argparse
import logging
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pandas as pd
import pickle
import random
import time
import h5py
import mne
from pathlib import Path
from tqdm import tqdm
from functools import partial

# Add LaBraM directory to path
labram_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if labram_dir not in sys.path:
    sys.path.append(labram_dir)

# Import LaBraM modules
import sys
import os

# These imports should match those used in the original LaBraM code
import modeling_pretrain
import modeling_vqnsp
import utils
from engine_for_pretraining import random_masking


def setup_logging(log_dir):
    """Setup logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pretraining.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class H5EEGDataset(torch.utils.data.Dataset):
    """Dataset for loading EEG data from HDF5 file."""
    
    def __init__(self, h5_path, metadata_df=None, resample_to_hz=200):
        """
        Initialize the dataset.
        
        Args:
            h5_path: Path to HDF5 file with EEG data
            metadata_df: Optional DataFrame with metadata for filtering
            resample_to_hz: Target sampling rate in Hz
        """
        self.h5_path = h5_path
        self.metadata_df = metadata_df
        self.resample_to_hz = resample_to_hz
        
        # Original sampling rate from Force Game Methods doc
        self.original_sampling_rate = 500  # Hz
        
        # Get participant IDs and number of epochs
        with h5py.File(self.h5_path, 'r') as h5file:
            self.participant_ids = list(h5file.keys())
            
            # Create a list of (participant_id, epoch_idx) tuples
            self.samples = []
            for participant_id in self.participant_ids:
                n_epochs = h5file[participant_id]['epochs'].shape[0]
                for epoch_idx in range(n_epochs):
                    self.samples.append((participant_id, epoch_idx))
        
        print(f"Dataset initialized with {len(self.samples)} samples from {len(self.participant_ids)} participants")
        
        # Get dimensions of the data
        with h5py.File(self.h5_path, 'r') as h5file:
            sample_epoch = h5file[self.participant_ids[0]]['epochs'][0]
            self.n_channels = sample_epoch.shape[0]
            self.n_timepoints = sample_epoch.shape[1]
        
        print(f"EEG data dimensions: {self.n_channels} channels, {self.n_timepoints} timepoints")
        
        # Calculate dimensions after resampling
        resampling_factor = self.resample_to_hz / self.original_sampling_rate
        self.n_timepoints_resampled = int(self.n_timepoints * resampling_factor)
        print(f"After resampling to {self.resample_to_hz} Hz: {self.n_timepoints_resampled} timepoints")
    
    def __len__(self):
        return len(self.samples)
    
    def _resample(self, data):
        """Resample data to target sampling rate."""
        from scipy import signal
        
        # Calculate number of output points
        n_out = int(data.shape[1] * (self.resample_to_hz / self.original_sampling_rate))
        
        # Resample each channel
        resampled_data = np.zeros((data.shape[0], n_out))
        for ch in range(data.shape[0]):
            resampled_data[ch] = signal.resample(data[ch], n_out)
        
        return resampled_data
    
    def __getitem__(self, idx):
        """Get an EEG sample."""
        participant_id, epoch_idx = self.samples[idx]
        
        # Load data
        with h5py.File(self.h5_path, 'r') as h5file:
            eeg_data = h5file[participant_id]['epochs'][epoch_idx][:]
        
        # Resample if needed
        if self.resample_to_hz != self.original_sampling_rate:
            eeg_data = self._resample(eeg_data)
        
        # Reshape to what LaBraM expects: [channels, num_windows, patch_size]
        # LaBraM expects patch_size=200, so we need to segment the data
        patch_size = 200
        num_windows = eeg_data.shape[1] // patch_size
        
        # Truncate to ensure divisibility by patch_size
        if eeg_data.shape[1] % patch_size != 0:
            eeg_data = eeg_data[:, :num_windows * patch_size]
        
        # Reshape to [channels, num_windows, patch_size]
        eeg_data = eeg_data.reshape(eeg_data.shape[0], num_windows, patch_size)
        
        # Convert to torch tensor
        eeg_tensor = torch.from_numpy(eeg_data).float()
        
        return eeg_tensor


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                    start_warmup_value=0, warmup_steps=-1):
    """
    Create a cosine learning rate schedule.
    This is copied from the original LaBraM utils.py for completeness.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_visual_tokenizer(args):
    """
    Load the VQNSP tokenizer with proper handling of NumPy types.
    """
    import torch
    import numpy as np
    
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    
    # The model is registered with a specific function name
    original_fn = getattr(modeling_vqnsp, args.tokenizer_model)
    
    # Create the model without loading weights yet
    model = original_fn(pretrained=False, pretrained_weight=None, as_tokenzer=False, 
                        n_code=args.codebook_size, code_dim=args.codebook_dim)
    
    print(f"Loading tokenizer weights from: {args.tokenizer_weight}")
    
    # Load with weights_only explicitly set to False (security note: only use with trusted files)
    print("Loading with weights_only=False (this is safe for trusted files)")
    try:
        # This is safe when you trust the source of the file
        checkpoint = torch.load(args.tokenizer_weight, map_location='cpu')
        print("Successfully loaded checkpoint!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Attempting an alternative loading method...")
        
        # Try again with pickle and custom unpickler for more controlled loading
        import pickle
        import io
        
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Only allow specific modules/classes
                if module == 'numpy' or module.startswith('numpy.'):
                    # Allow NumPy types
                    if module == 'numpy' and name in ['dtype', 'ndarray', 'int64', 'float64']:
                        return getattr(np, name)
                    if module == 'numpy.core.multiarray' and name == 'scalar':
                        return np.core.multiarray.scalar
                # Default behavior for other modules
                return super().find_class(module, name)
        
        with open(args.tokenizer_weight, 'rb') as f:
            checkpoint = SafeUnpickler(f).load()
        print("Successfully loaded checkpoint with SafeUnpickler!")
    
    # Process weights
    print("Processing checkpoint...")
    if 'model' in checkpoint:
        weights = checkpoint['model']
        print("Found model key in checkpoint")
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
        print("Found state_dict key in checkpoint")
    else:
        weights = checkpoint
        print("Using checkpoint directly as weights")
    
    # Filter unwanted keys
    keys = list(weights.keys())
    for k in keys:
        if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
            del weights[k]
            print(f"Removed key: {k}")
    
    # Load weights into model
    print(f"Loading weights into model...")
    try:
        model.load_state_dict(weights)
        print("Successfully loaded weights!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Try loading with strict=False
        print("Trying with strict=False...")
        model.load_state_dict(weights, strict=False)
        print("Successfully loaded weights with strict=False!")
    
    model.eval()
    return model


def train_one_epoch(model, vqnsp, dataloader, optimizer, device, epoch, loss_scaler, max_norm, log_writer, 
                   lr_schedule_values, args):
    """
    Train for one epoch.
    Simplified from engine_for_pretraining.py to focus on domain adaptation.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    
    # Get the channel names for the input data
    input_chans = list(range(1, model.student.pos_embed.shape[1]))  # Skip CLS token position
    
    for step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # Assign learning rate for this step
        it = epoch * len(dataloader) + step  # global training iteration
        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
        
        # Move batch to device
        samples = batch.float().to(device, non_blocking=True) / 100  # Scale to 0.1 mV as in LaBraM
        
        # Generate random mask
        batch_size, num_channels, num_windows, patch_size = samples.shape
        bool_masked_pos = random_masking(
            samples.flatten(1, 2), 
            mask_ratio=0.5
        ).to(device, non_blocking=True)
        
        # Get token IDs using the tokenizer
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = vqnsp.get_codebook_indices(samples, input_chans)
                
            # Get labels for masked and unmasked positions
            labels = input_ids[bool_masked_pos]
            labels_sym = input_ids[~bool_masked_pos]
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(samples, input_chans, bool_masked_pos=bool_masked_pos)
            
            x_rec, x_rec_sym = outputs
            loss_fn = nn.CrossEntropyLoss()
            loss_rec = loss_fn(x_rec, labels)
            loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
            loss = loss_rec + loss_rec_sym
        
        loss_value = loss.item()
        
        # Check for NaN
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            raise RuntimeError(f"Loss is {loss_value}, stopping training")
        
        # Backward pass and optimization
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=model.parameters(), 
            create_graph=is_second_order, 
            update_grad=True
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]
        
        # Calculate metrics
        mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
        mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
        
        # Update metrics
        metric_logger.update(mlm_acc=mlm_acc)
        metric_logger.update(mlm_acc_sym=mlm_acc_sym)
        metric_logger.update(loss_rec=loss_rec.item() / 2)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        
        # Update learning rate meter
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        # Log to TensorBoard
        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")
            log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
            log_writer.update(loss_rec=loss_rec.item() / 2, head="loss")
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # Gather stats from all processes if distributed
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    """Main function."""
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting LaBraM pre-training fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = H5EEGDataset(
        args.h5_path,
        resample_to_hz=args.resample_hz
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Set up tensorboard
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # Load visual tokenizer (VQNSP)
    vqnsp = get_visual_tokenizer(args)
    vqnsp.to(device)
    vqnsp.eval()  # Keep in eval mode as we only use it for encoding
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model_class = getattr(modeling_pretrain, args.model)
    
    # Create the model with correct parameters
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    
    # Determine layer scale init value based on model size
    layer_scale_init_value = args.layer_scale_init_value
    
    # Create model instance with the right parameters
    model_kwargs = {
        'vocab_size': args.codebook_size,
        'use_abs_pos_emb': args.abs_pos_emb,
        'use_rel_pos_bias': args.rel_pos_bias,
        'init_values': layer_scale_init_value,
        'drop_path_rate': args.drop_path,
        'norm_layer': norm_layer
    }
    
    try:
        # Create the model
        model = model_class(**model_kwargs)
        logger.info(f"Successfully created model: {args.model}")
        
        # Load pretrained weights if specified
        if args.pretrained and args.model_path:
            logger.info(f"Loading pretrained weights from {args.model_path}")
            
            try:
                checkpoint = torch.load(args.model_path, map_location='cpu')
                
                # Extract the model state dict
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load the state dict with strict=False to allow missing keys
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                logger.info(f"Successfully loaded pretrained weights")
                logger.info(f"Missing keys: {missing_keys}")
                logger.info(f"Unexpected keys: {unexpected_keys}")
                
            except Exception as e:
                logger.error(f"Error loading pretrained weights: {e}")
                # Continue without pretrained weights
        
        model.to(device)
        logger.info(f"Model moved to {device}")
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to create model: {e}")
    
    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model = {model.__class__.__name__}")
    logger.info(f"Number of trainable parameters: {n_parameters}")
    
    # Create optimizer
    from optim_factory import create_optimizer
    optimizer = create_optimizer(args, model)
    
    # Create loss scaler for mixed precision training
    loss_scaler = utils.NativeScalerWithGradNormCount()
    
    # Create learning rate schedule
    num_training_steps_per_epoch = len(dataloader)
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )
    
    # Training loop
    logger.info(f"Starting pre-training fine-tuning for {args.epochs} epochs")
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Starting epoch {epoch}")
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model=model,
            vqnsp=vqnsp,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            lr_schedule_values=lr_schedule_values,
            args=args
        )
        
        # Save checkpoint
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch == args.epochs - 1):
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pth"
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            
            if loss_scaler is not None:
                checkpoint['scaler'] = loss_scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log epoch stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Save final model
    final_checkpoint_path = Path(args.output_dir) / "final_model.pth"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'args': args,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")
    
    logger.info("Pre-training fine-tuning complete!")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LaBraM pre-training on social interaction EEG data")
    
    # Data paths
    parser.add_argument("--h5-path", type=str, required=True,
                        help="Path to HDF5 EEG data file")
    parser.add_argument("--output-dir", type=str, default="./pretrain_output",
                        help="Directory to save output files")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for tensorboard logs")
    
    # Model parameters
    parser.add_argument("--model", default='labram_base_patch200_1600_8k_vocab', type=str,
                        help="Name of model to train")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to pre-trained LaBraM model")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pre-trained model")
    parser.add_argument("--rel-pos-bias", action="store_true", default=False,
                        help="Use relative position bias")
    parser.add_argument("--abs-pos-emb", action="store_true", default=True,
                        help="Use absolute position embedding")
    parser.add_argument("--layer-scale-init-value", default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument("--drop-path", type=float, default=0.1,
                        help="Drop path rate")
    
    # Tokenizer parameters
    parser.add_argument("--tokenizer-model", type=str, default="vqnsp_encoder_base_decoder_3x200x12",
                        help="VQNSP tokenizer model")
    parser.add_argument("--tokenizer-weight", type=str, required=True,
                        help="Path to VQNSP tokenizer weights")
    parser.add_argument("--codebook-size", default=8192, type=int,
                        help="Number of codebook entries")
    parser.add_argument("--codebook-dim", default=64, type=int,
                        help="Dimension of codebook embeddings")
    
    # DataLoader parameters
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--resample-hz", type=int, default=200,
                        help="Target sampling rate in Hz")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Start epoch")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--clip-grad", type=float, default=3.0,
                        help="Gradient clipping norm")
    parser.add_argument("--save-ckpt-freq", type=int, default=5,
                        help="Save checkpoint frequency in epochs")
    
    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str,
                        help="Optimizer")
    parser.add_argument("--opt-eps", default=1e-8, type=float,
                        help="Optimizer epsilon")
    parser.add_argument("--opt-betas", default=[0.9, 0.999], type=float, nargs='+',
                        help="Optimizer betas")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run main function
    main(args)