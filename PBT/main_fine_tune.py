from src.utils import *
from src.train import training
from src.model import PBT

import numpy as np
import random
import torch


def fine_tune(config):
    if config["load"]:
        checkpoint = torch.load(config["load"])
        dict_channels = checkpoint["dict_channels"]

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

    if config["data_set"] == "BNCI2014001":
        data, labels, meta, channels = get_BNCI2014001(
            subject=list(range(1, 10)),
            freq_min=config["freq"][0],
            freq_max=config["freq"][1],
        )

        train_data = data[np.where(meta["session"] == "session_T")]
        train_labels = labels[np.where(meta["session"] == "session_T")]
        train_meta = meta.iloc[np.where(meta["session"] == "session_T")]

        test_data = data[np.where(meta["session"] == "session_E")]
        test_labels = labels[np.where(meta["session"] == "session_E")]
        test_meta = meta.iloc[np.where(meta["session"] == "session_E")]

    elif config["data_set"] == "BNCI2014004":
        data, labels, meta, channels = get_BNCI2014004(
            subject=list(range(1, 10)),
            freq_min=config["freq"][0],
            freq_max=config["freq"][1],
        )

        train_data = data[
            np.where(
                (meta["session"] == "session_0")
                | (meta["session"] == "session_1")
                | (meta["session"] == "session_2")
            )
        ]
        train_labels = labels[
            np.where(
                (meta["session"] == "session_0")
                | (meta["session"] == "session_1")
                | (meta["session"] == "session_2")
            )
        ]
        train_meta = meta.iloc[
            np.where(
                (meta["session"] == "session_0")
                | (meta["session"] == "session_1")
                | (meta["session"] == "session_2")
            )
        ]

        test_data = data[
            np.where(
                (meta["session"] == "session_3") | (meta["session"] == "session_4")
            )
        ]
        test_labels = labels[
            np.where(
                (meta["session"] == "session_3") | (meta["session"] == "session_4")
            )
        ]
        test_meta = meta.iloc[
            np.where(
                (meta["session"] == "session_3") | (meta["session"] == "session_4")
            )
        ]

    else:
        raise ValueError("Please choose data_set in {BNCI2014001, BNCI2014004}")

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)

    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    if config["load"]:
        train_data_set.prepare_data_set(dict_channels)
        test_data_set.prepare_data_set(dict_channels)
    else:
        train_data_set.prepare_data_set()
        test_data_set.prepare_data_set(train_data_set.dict_channels)

    model = PBT(
        d_input=config["d_input"],
        n_classes=len(set(test_labels)),
        num_embeddings=torch.max(
            torch.cat(list(train_data_set.dict_channels.values()))
        ).item()
        + 1,
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

    if config["load"]:
        # delete weights that should not be loaded
        # checkpoint['model_state_dict'].pop('pos_embedding.weight')
        checkpoint["model_state_dict"].pop("cls_head.weight")
        checkpoint["model_state_dict"].pop("cls_head.bias")
        checkpoint["model_state_dict"].pop("linear_projection_out.weight")

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    training(
        parameter=config,
        model=model,
        train_data_set=train_data_set,
        test_data_set=test_data_set,
        n_classes=len(set(test_labels)),
    )


if __name__ == "__main__":

    config = {
        "pre_train_bert": False,  # unsupervised pre-training in BERT-style
        # Pre - Processing
        "freq": [8, 45],
        "normalization": "zscore",
        # Model
        "d_input": 64,
        "d_model": 128,  # Input gets expanded in lin. projection
        "dim_feedforward": 128 * 4,
        "num_tokens_per_channel": 8,
        "num_transformer_blocks": 4,
        "num_heads": 4,  # number attention heads transformer
        "bert_supervised": False,  # add a reconstruction task (BERT) as regularisation to loss
        "learnable_cls": False,
        "bias_transformer": True,
        # Train Hyper-Parameters
        "lr": 3e-4,
        "lr_warm_up_iters": 50,
        "batch_size": 64,
        "num_epochs": 120,
        "betas": (0.9, 0.95),  # betas AdamW
        "clip_gradient": 1.0,
        # Regularization & Augmentation
        "weight_decay": 0.01,  # not applied to LayerNorm, self_att and biases
        "weight_decay_pos_embedding": 0.0,  # weight decay applied to learnable pos. embedding
        "weight_decay_cls_head": 0.0,  # cls_head = classification head (linear layer)
        # higher for pre-train may improve few-shot adaptation
        "dropout": 0.1,
        "label_smoothing": 0,
        "augmentation": ["time_shifts"],  # [] for no aug, else list:
        # ['time_shifts', 'DC_shifts', 'amplitude_scaling','noise']
        # WandB
        "wandb_log": True,
        "wandb_name": False,
        "wandb_proj": "Patched Brain Transformer",
        "wandb_watch": True,
        "save": False,  # add path as string where to save
        "checkpoints": False,
        "load": False,
        "seed": 42,  # set random seed
        "compile_model": False,  # compile model with PyTroch to speed up
        "data_set": "BNCI2014001",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(1, 4):
        config["seed"] = i
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

        fine_tune(config)
