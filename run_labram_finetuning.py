# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import pandas as pd
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain
import modeling_vqnsp


def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int, help='Fewer epochs for fine-tuning')
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str, default='/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/vqnsp.pth', 
                        help="Path to pre-trained tokenizer weights")
    parser.add_argument("--tokenizer_model", type=str, default="vqnsp_encoder_base_decoder_3x200x12")
    
    # Model parameters for pre-trained model
    parser.add_argument('--pretrained_model', type=str, default='/zhome/ce/8/186807/Desktop/Labram/LaBraM-MMDTU/checkpoints/labram-base.pth',
                    help='Path to pre-trained LaBraM model weights')
    parser.add_argument('--model', default='labram_base_patch200_1600_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--model_key', default='model|module', type=str)

    parser.add_argument('--input_size', default=1600, type=int,
                        help='EEG input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='number of codebook')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    
    # Dataset parameters
    parser.add_argument('--use_dtu_loader', action='store_true',
                    help='Use the DTU data loader instead of ShockDataset')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size
    )

    return model

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Create model
    model = get_model(args)
    patch_size = model.student.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # Load pretrained model
    if args.pretrained_model:
        if args.pretrained_model.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained_model, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained_model, map_location='cpu')

        print("Load ckpt from %s" % args.pretrained_model)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
            
        # Remove any incompatible keys
        state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if k not in state_dict or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        # Load the checkpoint
        model.load_state_dict(checkpoint_model, strict=False)
        print("Pretrained model loaded successfully!")

    # prepare visual tokenizer
    vqnsp = get_visual_tokenizer(args).to(device)

    # Use DTU loader directly
    if args.use_dtu_loader:
        # Use the DTU data loader directly
        train_dataset, test_dataset, val_dataset = utils.prepare_DTU_data("/work3/s224183/LaBraM_data")
        
        # Get channel names from the proper function
        ch_names = utils.get_channel_names()
        train_ch_names_list = ch_names
        val_ch_names_list = ch_names
        
        # Create the dataset lists needed for the training loop
        dataset_train_list = [train_dataset]
        dataset_train_list.append(val_dataset)
        dataset_val_list = [test_dataset] 
    else:
        # Original ShockDataset loading logic for non-DTU data
        datasets_train = [["/work3/s224183/LaBraM_data/train"]]
        datasets_test = [["/work3/s224183/LaBraM_data/test"]]
        datasets_val = [["/work3/s224183/LaBraM_data/val"]]
        time_window = [4]
    
        dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(
            datasets_train, time_window, stride_size=800, start_percentage=0, end_percentage=1
        )
        
        dataset_val_list, val_ch_names_list = utils.build_pretraining_dataset(
            datasets_val, time_window, stride_size=800, start_percentage=0, end_percentage=1
        )

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
        print("Sampler_train = %s" % str(sampler_train))

        # Set up validation samplers
        sampler_val_list = []
        for dataset in dataset_val_list:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
            )
            sampler_val_list.append(sampler_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)

    data_loader_val_list = []
    for dataset, sampler in zip(dataset_val_list, sampler_val_list):
        data_loader_val = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_val_list.append(data_loader_val)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
            # Remove the problematic key
            if 'logit_scale' in model_dict:
                print("Removing logit_scale from checkpoint")
                del model_dict['logit_scale']
            
            # Load the modified state dict
            model_without_ddp.load_state_dict(model_dict, strict=False)  # Use strict=False for safety
            
        # Load optimizer state, etc. if needed
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint and loss_scaler is not None:
            loss_scaler.load_state_dict(checkpoint['scaler'])
    else:
        # Use the normal auto_load_model function if not resuming
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp, 
            optimizer=optimizer, loss_scaler=loss_scaler)
    
    print(f"Start fine-tuning for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, vqnsp, data_loader_train_list,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args,
        )
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

        # Evaluate on validation data if available
        # Note: You'll need to implement a validation function if needed

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")




    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Fine-tuning time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)