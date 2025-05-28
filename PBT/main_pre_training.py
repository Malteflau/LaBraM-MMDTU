import torch
from src.utils import *
from src.model import PBT
from src.train import training


def pre_train(config, reduce_num_chs_to):
    test_size = 0.05
    resample = 250
    train_data_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=reduce_num_chs_to,
        augmentation=config["augmentation"],
    )
    test_data_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=reduce_num_chs_to,
    )

    # AlexMI
    data, labels, meta, channels = get_AlexMI(
        freq_min=config["freq"][0], freq_max=config["freq"][1], resample=resample
    )
    train_data, train_labels, train_meta, test_data, test_labels, test_meta = (
        train_test_split(data, labels, meta, test_size=test_size)
    )

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    # BNCI2015004
    data, labels, meta, channels = get_BNCI2015004(
        freq_min=config["freq"][0], freq_max=config["freq"][1]
    )
    train_data, train_labels, train_meta, test_data, test_labels, test_meta = (
        train_test_split(data, labels, meta, test_size=test_size)
    )

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    # Cho2017
    data, labels, meta, channels = get_Cho2017(
        freq_min=config["freq"][0], freq_max=config["freq"][1], resample=resample
    )
    train_data, train_labels, train_meta, test_data, test_labels, test_meta = (
        train_test_split(data, labels, meta, test_size=test_size)
    )

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    # Lee
    data, labels, meta, channels = get_Lee2019_MI(
        freq_min=config["freq"][0], freq_max=config["freq"][1], resample=resample
    )
    train_data, train_labels, train_meta, test_data, test_labels, test_meta = (
        train_test_split(data, labels, meta, test_size=test_size)
    )

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    # PhysionetMI
    data, labels, meta, channels = get_PhysionetMI(
        freq_min=config["freq"][0], freq_max=config["freq"][1], resample=resample
    )
    train_data, train_labels, train_meta, test_data, test_labels, test_meta = (
        train_test_split(data, labels, meta, test_size=test_size)
    )

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    #DTU data
    
    # create data set
    train_data_set.prepare_data_set()
    test_data_set.prepare_data_set(set_pos_channels=train_data_set.dict_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PBT(
        d_input=config["d_input"],
        n_classes=6,
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

    training(
        parameter=config,
        model=model,
        train_data_set=train_data_set,
        test_data_set=test_data_set,
        n_classes=6,
        num_workers=4,
    )


if __name__ == "__main__":
    config = {
        "pre_train_bert": True,  # unsupervised pre-training in BERT-style
        # if false, supervised pre-training
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
        "lr_warm_up_iters": 100,
        "batch_size": 96,
        "num_epochs": 600,
        "betas": (0.9, 0.95),  # betas AdamW
        "clip_gradient": 1.0,
        # Regularization & Augmentation
        "weight_decay": 0.01,  # not applied to LayerNorm, self_att and biases
        "weight_decay_pos_embedding": 0.0,  # weight decay applied to learnable pos. embedding
        "weight_decay_cls_head": 1,  # cls_head = classification head (linear layer)
        # higher for pre-train may improve few-shot adaptation
        "dropout": 0.1,
        "label_smoothing": 0,
        "augmentation": ["time_shifts"],
        # WandB
        "wandb_log": True,
        "wandb_name": False,
        "wandb_proj": "PatchedBrainTransformer",
        "wandb_watch": True,
        "save": str("/zhome/ce/8/186807/Desktop/PatchedBrainTransformer/checkpoints"),  # add path as string where to save
        "checkpoints": 50,
        "load": False,
        "seed": 42,  # set random seed
        "compile_model": False,  # compile model with PyTroch to speed up
    }

    for i in range(1, 4):
        config["seed"] = i
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

        pre_train(config, reduce_num_chs_to=30)
