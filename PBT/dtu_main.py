from src.utils import *
from src.train import training
from src.model import PBT
from dtu_adapter import get_DTU_data, train_test_split_dtu

import numpy as np
import random
import torch


def train_dtu_model(config):
    """
    Train PBT model on DTU dataset.
    
    Args:
        config: Dictionary with model configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    # Get DTU data
    data, labels, meta, channels = get_DTU_data(
        subject=config.get("subjects", None),
        freq_min=config["freq"][0],
        freq_max=config["freq"][1],
        root=config["data_root"],
        condition=config["condition"],
        filter_feedback_only=config.get("filter_feedback_only", None),
        filter_non_feedback_only=config.get("filter_non_feedback_only", None), 
        filter_non_participant=config.get("filter_non_participant", True)
    )
    
    # If using the predefined train/test split
    if config.get("use_predefined_split", True):
        train_data = data[meta["session"] == "session_train"]
        train_labels = labels[meta["session"] == "session_train"]
        train_meta = meta[meta["session"] == "session_train"].reset_index(drop=True)
        
        test_data = data[meta["session"] == "session_test"]
        test_labels = labels[meta["session"] == "session_test"]
        test_meta = meta[meta["session"] == "session_test"].reset_index(drop=True)
    else:
        # Custom split if predefined split is not desired
        train_data, train_labels, train_meta, test_data, test_labels, test_meta = train_test_split_dtu(
            data, labels, meta, test_size=config.get("test_size", 0.2), random_state=config["seed"]
        )
    
    # Apply normalization
    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    
    # Add data to datasets
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )
    
    # Prepare datasets
    train_data_set.prepare_data_set()
    test_data_set.prepare_data_set(train_data_set.dict_channels)
    
    # Create model
    model = PBT(
        d_input=config["d_input"],
        n_classes=len(np.unique(labels)),
        num_embeddings=torch.max(
            torch.cat(list(train_data_set.dict_channels.values()))
        ).item() + 1,
        num_tokens_per_channel=config["num_tokens_per_channel"],
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
        checkpoint = torch.load(config["load"])
        # Delete weights that should not be loaded
        checkpoint["model_state_dict"].pop("cls_head.weight")
        checkpoint["model_state_dict"].pop("cls_head.bias")
        if "linear_projection_out.weight" in checkpoint["model_state_dict"]:
            checkpoint["model_state_dict"].pop("linear_projection_out.weight")
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
    # Train model
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
        "condition": ["sologroup"],  # Options: feedback, friendship, sologroup, gender
        "filter_feedback_only": False,
        "filter_non_feedback_only": None,
        "filter_non_participant": True,
        "use_predefined_split": True,  # Use predefined train/test split
        
        # Pre-processing
        "freq": [1, 40],
        "pre_train_bert": False,
        "normalization": "zscore",
        
        # Model parameters
        "d_input": 64,
        "d_model": 128,
        "dim_feedforward": 128 * 4,
        "num_tokens_per_channel": 8,
        "num_transformer_blocks": 4,
        "num_heads": 4,
        "bert_supervised": False,
        "learnable_cls": False,
        "bias_transformer": True,
        
        # Training parameters
        "lr": 3e-4,
        "lr_warm_up_iters": 50,
        "batch_size": 64,
        "num_epochs": 120,
        "betas": (0.9, 0.95),
        "clip_gradient": 1.0,
        
        # Regularization & Augmentation
        "weight_decay": 0.01,
        "weight_decay_pos_embedding": 0.0,
        "weight_decay_cls_head": 0.0,
        "dropout": 0.1,
        "label_smoothing": 0,
        "augmentation": ["time_shifts"],
        
        # WandB logging
        "wandb_log": True,
        "wandb_name": "sologroup 75",
        "wandb_proj": "Patched Brain Transformer",
        "wandb_watch": True,
        
        # Misc
        "save": False,
        "checkpoints": 1000,
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
    train_dtu_model(config)