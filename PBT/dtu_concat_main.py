## PBT extension for DTU dataset with concatenated epochs
from src.utils import *
from src.train import training
from src.model import PBT
from dtu_concat import get_DTU_concatenated_data

import numpy as np
import random
import torch


def train_dtu_concatenated_model(config):
    """
    Train PBT model on DTU dataset with concatenated epochs.
    
    Args:
        config: Dictionary with model configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize datasets
    train_data_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=False,
        augmentation=config["augmentation"],
    )
    test_data_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=False,
    )

    # Get DTU data with concatenated epochs
    print("Loading DTU data with concatenated epochs...")
    data, labels, meta, channels = get_DTU_concatenated_data(
        subject=config.get("subjects", None),
        freq_min=config["freq"][0],
        freq_max=config["freq"][1],
        root=config["data_root"],
        condition=config["condition"],
        filter_feedback_only=config.get("filter_feedback_only", None),
        filter_non_feedback_only=config.get("filter_non_feedback_only", None), 
        filter_non_participant=config.get("filter_non_participant", True),
        concat_same_conditions=config.get("concat_same_conditions", True),
        max_epochs_to_concat=config.get("max_epochs_to_concat", 30)
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Number of samples: {len(data)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Use predefined split or create custom split
    if config.get("use_predefined_split", True):
        train_data = data[meta["session"] == "session_train"]
        train_labels = labels[meta["session"] == "session_train"]
        train_meta = meta[meta["session"] == "session_train"].reset_index(drop=True)
        
        test_data = data[meta["session"] == "session_test"]
        test_labels = labels[meta["session"] == "session_test"]
        test_meta = meta[meta["session"] == "session_test"].reset_index(drop=True)
    else:
        # Custom split
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split = int(len(indices) * 0.8)  # 80% training, 20% testing
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        train_meta = meta.iloc[train_indices].reset_index(drop=True)
        
        test_data = data[test_indices]
        test_labels = labels[test_indices]
        test_meta = meta.iloc[test_indices].reset_index(drop=True)
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Testing set: {len(test_data)} samples")
    
    # Apply normalization
    print("Applying normalization...")
    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    
    # Add data to datasets
    print("Preparing datasets...")
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )
    
    # Prepare datasets
    train_data_set.prepare_data_set()
    test_data_set.prepare_data_set(train_data_set.dict_channels)
    
    # Create model with adjusted parameters for longer sequences
    # We may need to adjust the number of tokens per channel for longer sequences
    print("Creating model...")
    # For concatenated sequences, we might need to adjust these parameters
    if config.get("concat_same_conditions", True):
        # Adjust model parameters for longer sequences if needed
        num_tokens_per_channel = min(config["num_tokens_per_channel"], 32)  # Cap at 32 tokens per channel
        print(f"Using {num_tokens_per_channel} tokens per channel for concatenated data")
    else:
        num_tokens_per_channel = config["num_tokens_per_channel"]
    
    model = PBT(
        d_input=config["d_input"],
        n_classes=len(np.unique(labels)),
        num_embeddings=torch.max(
            torch.cat(list(train_data_set.dict_channels.values()))
        ).item() + 1,
        num_tokens_per_channel=num_tokens_per_channel,
        d_model=config["d_model"],
        n_blocks=config["num_transformer_blocks"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        device=device,
        learnable_cls=config["learnable_cls"],
        bias_transformer=config["bias_transformer"],
        bert=True if config["bert_supervised"] or config["pre_train_bert"] else False,
    )
    
    # Load pre-trained weights if specified
    if config["load"]:
        print(f"Loading pre-trained weights from {config['load']}...")
        checkpoint = torch.load(config["load"])
        # Delete weights that should not be loaded
        checkpoint["model_state_dict"].pop("cls_head.weight")
        checkpoint["model_state_dict"].pop("cls_head.bias")
        if "linear_projection_out.weight" in checkpoint["model_state_dict"]:
            checkpoint["model_state_dict"].pop("linear_projection_out.weight")
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("Starting training...")
    training(
        parameter=config,
        model=model,
        train_data_set=train_data_set,
        test_data_set=test_data_set,
        n_classes=len(np.unique(labels)),
    )


if __name__ == "__main__":
    
    config = {
        # Dataset parameters
        "data_root": "/work3/s224183/LaBraM_data",  # Replace with actual path
        "condition": ["gender"],  # Options: feedback, friendship, sologroup, gender
        "filter_feedback_only": None,
        "filter_non_feedback_only": None,
        "filter_non_participant": True,
        "use_predefined_split": True,  # Use predefined train/test split
        
        # Concatenation parameters
        "concat_same_conditions": True,  # Enable concatenation of same conditions
        "max_epochs_to_concat": 10,     # Maximum number of epochs to concatenate
        
        # Pre-processing
        "freq": [8, 45],
        "pre_train_bert": False,
        "normalization": "zscore",
        
        # Model parameters - these may need adjustment for longer sequences
        "d_input": 64,
        "d_model": 128,
        "dim_feedforward": 128 * 4,
        "num_tokens_per_channel": 16,  # Increased for longer sequences
        "num_transformer_blocks": 4,
        "num_heads": 4,
        "bert_supervised": False,
        "learnable_cls": False,
        "bias_transformer": True,
        
        # Training parameters
        "lr": 3e-4,
        "lr_warm_up_iters": 50,
        "batch_size": 16,  # Reduced batch size due to larger samples
        "num_epochs": 120,
        "betas": (0.9, 0.95),
        "clip_gradient": 1.0,
        
        # Regularization & Augmentation
        "weight_decay": 0.01,
        "weight_decay_pos_embedding": 0.0,
        "weight_decay_cls_head": 0.0,
        "dropout": 0.2,  # Slightly increased dropout for regularization
        "label_smoothing": 0.1,  # Added label smoothing
        "augmentation": ["time_shifts"],
        
        # WandB logging
        "wandb_log": True,
        "wandb_name": "PBT_DTU_Concatenated",
        "wandb_proj": "Patched Brain Transformer",
        "wandb_watch": True,
        
        # Misc
        "save": "./checkpoints",
        "checkpoints": 60,
        "load": False,
        "seed": 42,
        "compile_model": False,
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    
    # Train model
    train_dtu_concatenated_model(config)