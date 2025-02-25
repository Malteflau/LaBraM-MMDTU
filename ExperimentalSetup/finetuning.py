#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning script for LaBraM on social interaction EEG data.

This script fine-tunes the LaBraM (Large Brain Model) on social interaction
EEG data from the Force Game experiment.

Author: Magnus Evensen, Malte Færgemann Lau
Project: Bachelor's Project - EEG Social Interaction
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import random
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataloader import create_dataloaders


def setup_logging(log_dir):
    """Setup logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
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


def load_model(model_path, model_name, num_classes, device):
    """
    Load LaBraM model.
    
    Args:
        model_path: Path to pre-trained model weights
        model_name: Model name ('labram_base_patch200_200', 'labram_large_patch200_200', 
                              or 'labram_huge_patch200_200')
        num_classes: Number of output classes
        device: Device to load model to
        
    Returns:
        LaBraM model
    """
    # Import necessary modules from LaBraM
    import sys
    import os
    
    # Add the parent directory to sys.path to import LaBraM modules
    labram_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if labram_dir not in sys.path:
        sys.path.append(labram_dir)
    
    # Import the model
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import modeling_finetune
    from functools import partial
    import torch.nn as nn
    
    # Set layer scale init value based on model size
    if model_name == "labram_base_patch200_200":
        layer_scale_init_value = 0.1
    elif model_name in ["labram_large_patch200_200", "labram_huge_patch200_200"]:
        layer_scale_init_value = 1e-5
    else:
        layer_scale_init_value = 0.0
    
    # Create the model with proper init_values
    if model_name == "labram_base_patch200_200":
        model = modeling_finetune.NeuralTransformer(
            patch_size=200, 
            embed_dim=200, 
            depth=12, 
            num_heads=10, 
            mlp_ratio=4, 
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            init_values=layer_scale_init_value
        )
    elif model_name == "labram_large_patch200_200":
        model = modeling_finetune.NeuralTransformer(
            patch_size=200, 
            embed_dim=400, 
            depth=24, 
            num_heads=16, 
            mlp_ratio=4, 
            out_chans=16, 
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            init_values=layer_scale_init_value
        )
    elif model_name == "labram_huge_patch200_200":
        model = modeling_finetune.NeuralTransformer(
            patch_size=200, 
            embed_dim=800, 
            depth=48, 
            num_heads=16, 
            mlp_ratio=4, 
            out_chans=32, 
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            init_values=layer_scale_init_value
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Load pretrained weights
    if model_path:
        print(f"Loading pretrained weights from {model_path}")
        try:
            # First try with weights_only=False (safe if you trust the source)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to load with weights_only=False: {e}")
            try:
                # Try an alternative approach with safe globals
                import torch.serialization
                import numpy as np
                # Add numpy scalar to safe globals
                with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as e2:
                print(f"Failed alternative loading approach: {e2}")
                print("Attempting with a more compatible loading approach...")
                # Try a more compatible approach
                import pickle
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location="cpu", pickle_module=pickle)
        
        # Check for model key patterns based on the LaBraM codebase
        model_state_dict = None
        for key in ['model', 'module', 'state_dict']:
            if key in checkpoint:
                model_state_dict = checkpoint[key]
                print(f"Found model weights under key: {key}")
                break
        
        if model_state_dict is None:
            model_state_dict = checkpoint  # Assume it's directly the state dict
        
        # Handle potential prefixes and create a clean state dict
        prefix_matches = ['', 'student.', 'module.']
        state_dict = {}
        for k, v in model_state_dict.items():
            for prefix in prefix_matches:
                if k.startswith(prefix):
                    clean_k = k[len(prefix):]
                    # Exclude parameters for the head (classification layer)
                    if 'head.' not in clean_k:
                        state_dict[clean_k] = v
                    break
        
        # Load weights, but skip the head layer
        msg = model.load_state_dict(state_dict, strict=False)
        
        # Log missing and unexpected keys
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
    
    # Move model to device
    model = model.to(device)
    
    return model

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc="Validation"):
    """
    Evaluate model with custom forward function to handle channel mismatch.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        desc: Description for progress bar
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    # Define the same custom forward function as in train_epoch
    def custom_forward(model, x, input_chans=None):
        """Custom forward pass that handles channel count mismatches."""
        batch_size, n, a, t = x.shape
        x = model.patch_embed(x)
        
        # Add cls token
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle positional embeddings - IMPORTANT FIX
        if model.pos_embed is not None:
            # Get the actual number of channels in input
            actual_channels = n
            
            # Create a new positional embedding for the actual channels
            if not hasattr(model, 'adapted_pos_embed') or model.adapted_pos_embed.shape[1] != actual_channels + 1:
                # Create a new positional embedding for the actual channels
                # Extract cls token embedding and channel embeddings separately
                cls_pos_embed = model.pos_embed[:, 0:1, :]
                chan_pos_embed = model.pos_embed[:, 1:, :]
                
                # If input has fewer channels than expected, select a subset
                if actual_channels <= chan_pos_embed.shape[1]:
                    # Take first `actual_channels` embeddings
                    new_chan_pos_embed = chan_pos_embed[:, :actual_channels, :]
                else:
                    # If we have more channels than expected, repeat the last embedding
                    last_embed = chan_pos_embed[:, -1:, :]
                    extras = last_embed.repeat(1, actual_channels - chan_pos_embed.shape[1], 1)
                    new_chan_pos_embed = torch.cat([chan_pos_embed, extras], dim=1)
                
                # Combine cls and channel embeddings
                adapted_pos_embed = torch.cat([cls_pos_embed, new_chan_pos_embed], dim=1)
                
                # Store for reuse
                if not hasattr(model, 'adapted_pos_embed'):
                    model.register_buffer('adapted_pos_embed', adapted_pos_embed)
                else:
                    model.adapted_pos_embed = adapted_pos_embed
            
            # Use the adapted positional embedding
            pos_embed = model.adapted_pos_embed
            
            # Apply positional embedding - handle time dimension
            pos_embed_used = pos_embed
            
            # For the time dimension
            input_time_window = a if t == model.patch_size else t
            
            # Create positional embedding for channels and time
            if pos_embed_used is not None:
                pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
                pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
                x = x + pos_embed
                
            # Add time embedding if present
            if model.time_embed is not None:
                nc = n if t == model.patch_size else a
                time_embed = model.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
                x[:, 1:, :] += time_embed
        
        # Rest of the forward pass remains the same
        x = model.pos_drop(x)
        
        for blk in model.blocks:
            x = blk(x, rel_pos_bias=None)
        
        x = model.norm(x)
        if model.fc_norm is not None:
            t = x[:, 1:, :]
            x = model.fc_norm(t.mean(1))
        else:
            x = x[:, 0]
            
        # Apply classification head
        x = model.head(x)
        return x
    
    # Use tqdm for progress visualization
    from tqdm import tqdm
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
    
    # Check if we're using binary classification
    is_binary = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    
    for batch_idx, (patches, labels) in progress_bar:
        try:
            # Move data to device
            patches = patches.to(device)
            
            # Reshape input for LaBraM if needed
            if len(patches.shape) == 3:  # (batch_size, channels, time_points)
                # Assuming patch_size is 200
                patch_size = 200
                # Reshape to (batch_size, channels, num_windows, patch_size)
                b, c, t = patches.shape
                num_windows = t // patch_size
                # Only keep complete windows
                if t % patch_size != 0:
                    patches = patches[:, :, :num_windows*patch_size]
                
                patches = patches.reshape(b, c, num_windows, patch_size)
                print(f"Reshaped patches to: {patches.shape}")
            
            # Ensure labels have the right type and dimension
            if is_binary:
                labels = labels.float().to(device)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # Add extra dimension for BCE loss
            else:
                labels = labels.long().to(device)  # For CrossEntropyLoss
            
            # Forward pass using custom forward function
            outputs = custom_forward(model, patches)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Update running loss
            running_loss += loss.item() * patches.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Convert outputs to predictions
            if is_binary:
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(preds)
            all_targets.append(labels.cpu().numpy())
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # Calculate metrics
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        if is_binary:
            all_predictions = all_predictions.ravel()
            all_targets = all_targets.ravel()
            all_outputs = all_outputs.ravel()
        
        accuracy = accuracy_score(all_targets, all_predictions)
        
        metrics = {
            'loss': running_loss / len(dataloader.dataset),
            'accuracy': accuracy
        }
        
        # Calculate additional metrics
        if len(np.unique(all_targets)) <= 2:  # Binary classification
            try:
                # Only calculate these if we have both positive and negative samples
                if len(np.unique(all_targets)) > 1 and len(np.unique(all_predictions)) > 1:
                    precision = precision_score(all_targets, all_predictions, zero_division=0)
                    recall = recall_score(all_targets, all_predictions, zero_division=0)
                    f1 = f1_score(all_targets, all_predictions, zero_division=0)
                    metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                
                # Calculate AUC if possible
                if len(np.unique(all_targets)) > 1:
                    auc = roc_auc_score(all_targets, all_outputs)
                    metrics['auc'] = auc
                
            except Exception as e:
                print(f"Warning: Error calculating metrics: {e}")
                import traceback
                traceback.print_exc()
        else:  # Multi-class classification
            try:
                precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
                recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
                f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            except Exception as e:
                print(f"Warning: Error calculating metrics: {e}")
                import traceback
                traceback.print_exc()
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'loss': running_loss / len(dataloader.dataset),
            'accuracy': 0.0
        }
    
    return metrics
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with a custom forward function to handle channel mismatch."""
    model.train()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # Define a custom forward function to handle channel mismatch
    def custom_forward(model, x, input_chans=None):
        """Custom forward pass that handles channel count mismatches."""
        batch_size, n, a, t = x.shape
        x = model.patch_embed(x)
        
        # Add cls token
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle positional embeddings - IMPORTANT FIX
        if model.pos_embed is not None:
            # Get the actual number of channels in input
            actual_channels = n
            
            # Create a new positional embedding for the actual channels
            if not hasattr(model, 'adapted_pos_embed') or model.adapted_pos_embed.shape[1] != actual_channels + 1:
                # Create a new positional embedding for the actual channels
                # Extract cls token embedding and channel embeddings separately
                cls_pos_embed = model.pos_embed[:, 0:1, :]
                chan_pos_embed = model.pos_embed[:, 1:, :]
                
                # If input has fewer channels than expected, select a subset
                if actual_channels <= chan_pos_embed.shape[1]:
                    # Take first `actual_channels` embeddings
                    new_chan_pos_embed = chan_pos_embed[:, :actual_channels, :]
                else:
                    # If we have more channels than expected, repeat the last embedding
                    last_embed = chan_pos_embed[:, -1:, :]
                    extras = last_embed.repeat(1, actual_channels - chan_pos_embed.shape[1], 1)
                    new_chan_pos_embed = torch.cat([chan_pos_embed, extras], dim=1)
                
                # Combine cls and channel embeddings
                adapted_pos_embed = torch.cat([cls_pos_embed, new_chan_pos_embed], dim=1)
                
                # Store for reuse
                if not hasattr(model, 'adapted_pos_embed'):
                    model.register_buffer('adapted_pos_embed', adapted_pos_embed)
                else:
                    model.adapted_pos_embed = adapted_pos_embed
            
            # Use the adapted positional embedding
            pos_embed = model.adapted_pos_embed
            
            # Apply positional embedding - handle time dimension
            pos_embed_used = pos_embed
            
            # For the time dimension
            input_time_window = a if t == model.patch_size else t
            
            # Create positional embedding for channels and time
            if pos_embed_used is not None:
                pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
                pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
                x = x + pos_embed
                
            # Add time embedding if present
            if model.time_embed is not None:
                nc = n if t == model.patch_size else a
                time_embed = model.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
                x[:, 1:, :] += time_embed
        
        # Rest of the forward pass remains the same
        x = model.pos_drop(x)
        
        for blk in model.blocks:
            x = blk(x, rel_pos_bias=None)
        
        x = model.norm(x)
        if model.fc_norm is not None:
            t = x[:, 1:, :]
            x = model.fc_norm(t.mean(1))
        else:
            x = x[:, 0]
            
        # Apply classification head
        x = model.head(x)
        return x
    
    # Use tqdm for progress visualization
    from tqdm import tqdm
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]")
    
    # Check if we're using binary classification
    is_binary = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    
    for batch_idx, (patches, labels) in progress_bar:
        try:
            # Move data to device
            patches = patches.to(device)
            
            # Print shape for debugging
            print(f"Original patches shape: {patches.shape}")
            
            # Reshape input for LaBraM if needed
            # LaBraM expects shape (batch_size, channels, num_time_windows, patch_size)
            if len(patches.shape) == 3:  # (batch_size, channels, time_points)
                # Assuming patch_size is 200
                patch_size = 200
                # Reshape to (batch_size, channels, num_windows, patch_size)
                b, c, t = patches.shape
                num_windows = t // patch_size
                # Only keep complete windows
                if t % patch_size != 0:
                    patches = patches[:, :, :num_windows*patch_size]
                
                patches = patches.reshape(b, c, num_windows, patch_size)
                print(f"Reshaped patches to: {patches.shape}")
            
            # Ensure labels have the right type and dimension
            if is_binary:
                labels = labels.float().to(device)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # Add extra dimension for BCE loss
            else:
                labels = labels.long().to(device)  # For CrossEntropyLoss
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass using custom forward function
            try:
                # Use our custom forward function to handle channel mismatch
                outputs = custom_forward(model, patches)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Update running loss
                running_loss += loss.item() * patches.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Convert outputs to predictions
                if is_binary:
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Store predictions and targets for metrics calculation
                all_predictions.append(preds)
                all_targets.append(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error in forward/backward pass: {e}")
                import traceback
                traceback.print_exc()
                raise e
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # Calculate metrics
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        if is_binary:
            all_predictions = all_predictions.ravel()
            all_targets = all_targets.ravel()
        
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Calculate additional metrics
        if len(np.unique(all_targets)) <= 2:  # Binary classification
            try:
                precision = precision_score(all_targets, all_predictions)
                recall = recall_score(all_targets, all_predictions)
                f1 = f1_score(all_targets, all_predictions)
            except Exception as e:
                print(f"Warning: Error calculating metrics: {e}")
                precision, recall, f1 = 0.0, 0.0, 0.0
        else:  # Multi-class classification
            try:
                precision = precision_score(all_targets, all_predictions, average='macro')
                recall = recall_score(all_targets, all_predictions, average='macro')
                f1 = f1_score(all_targets, all_predictions, average='macro')
            except Exception as e:
                print(f"Warning: Error calculating metrics: {e}")
                precision, recall, f1 = 0.0, 0.0, 0.0
                
        metrics = {
            'loss': running_loss / len(dataloader.dataset),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'loss': running_loss / len(dataloader.dataset),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    scheduler=None, 
    num_epochs=10, 
    early_stopping_patience=5,
    checkpoint_dir=None,
    logger=None
):
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        early_stopping_patience: Number of epochs to wait for improvement
        checkpoint_dir: Directory to save checkpoints
        logger: Logger object
        
    Returns:
        Dictionary of training history
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize variables for early stopping
    best_val_metric = 0.0
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch} [Val]")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        # In train_model function in finetuning.py

        # Update history - use get() with default values
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics.get('f1', 0.0))  # Use get with default
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics.get('f1', 0.0))  # Use get with default

        # Safe logging
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics.get('f1', 0.0):.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics.get('f1', 0.0):.4f}")

        # Early stopping - use get with fallback to accuracy
        current_val_metric = val_metrics.get('f1', val_metrics['accuracy'])  # Use accuracy if f1 is missing
        
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            patience_counter = 0
            
            # Save best model
            if checkpoint_dir is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_metric': best_val_metric,
                    'val_metrics': val_metrics,
                }, checkpoint_dir / 'best_model.pt')
                
                logger.info(f"Saved best model with validation F1: {best_val_metric:.4f}")
        else:
            patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save checkpoint
        if checkpoint_dir is not None and epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    if checkpoint_dir is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': history
        }, checkpoint_dir / 'final_model.pt')
    
    return history


def main(args):
    """Main function."""
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting LaBraM fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        metadata_path=args.metadata,
        h5_path=args.h5,
        mapping_name=args.mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resample_to_hz=args.resample_hz,
        patch_size=args.patch_size,
        all_patches=args.all_patches,
        max_patches_per_sample=args.max_patches,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    logger.info(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Load model
    logger.info(f"Loading LaBraM model: {args.model_name}")
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=1 if args.num_classes <= 2 else args.num_classes,  # Use 1 output for binary
        device=device
    )
    logger.info(f"Model loaded: {model.__class__.__name__}")
    
    # Define loss function
    if args.num_classes == 1 or args.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Using BCEWithLogitsLoss for binary classification")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Using CrossEntropyLoss for {args.num_classes}-class classification")
    
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    logger.info(f"Using optimizer: {optimizer.__class__.__name__}")
    
    # Define scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    logger.info(f"Using scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    
    # Train model
    logger.info(f"Starting training for {args.num_epochs} epochs")
    
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        logger=logger
    )
    
    # Save history
    history_path = Path(args.output_dir) / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history to {history_path}")
    
    # Load best model
    best_checkpoint_path = checkpoint_dir / 'best_model.pt'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_checkpoint_path}")
        
        # Get validation metrics
        best_val_metrics = checkpoint['val_metrics']
        logger.info(f"Best validation metrics: {best_val_metrics}")
    
    # Save the model's configuration
    model_config = {
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'training_args': vars(args)
    }
    
    model_config_path = Path(args.output_dir) / 'model_config.pkl'
    with open(model_config_path, 'wb') as f:
        pickle.dump(model_config, f)
    logger.info(f"Saved model configuration to {model_config_path}")
    
    logger.info("Fine-tuning complete")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LaBraM on social interaction EEG data")
    
    # Data paths
    parser.add_argument("--metadata", type=str, default="./DataProcessed/metadata.pkl",
                        help="Path to metadata DataFrame")
    parser.add_argument("--h5", type=str, default="./DataProcessed/eeg_data.h5",
                        help="Path to HDF5 EEG data file")
    parser.add_argument("--mapping", type=str, default="feedback_prediction",
                        help="Name of condition mapping to use")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save output files")
    
    # Model parameters
    parser.add_argument("--model-path", type=str, default="/Users/maltelau/Desktop/LaBraM-MMDTU/LaBraM-MMDTU/checkpoints/labram-base.pth",required=True,
                        help="Path to pre-trained LaBraM model")
    parser.add_argument("--model-name", type=str, default="labram_base_patch200_200",
                        choices=["labram_base_patch200_200", "labram_large_patch200_200", "labram_huge_patch200_200"],
                        help="LaBraM model name")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes")
    
    # DataLoader parameters
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--resample-hz", type=int, default=200,
                        help="Target sampling rate in Hz")
    parser.add_argument("--patch-size", type=int, default=200,
                        help="Number of time samples per patch")
    parser.add_argument("--all-patches", action="store_true", default=True,
                        help="Use all patches per epoch for training")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Maximum number of patches per epoch")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Ratio of data to use for testing")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum (for SGD)")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["step", "cosine", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--step-size", type=int, default=10,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Gamma for StepLR scheduler")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    
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