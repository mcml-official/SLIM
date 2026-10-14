from argparse import ArgumentParser
from pathlib import Path
import os
import tempfile

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import CustomDataset
from utils.common import instantiate_from_config


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training stage: 1 or 2.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML.",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "Optional model configuration YAML. "
            "Overrides model.config in the training configuration."
        ),
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to the training dataset CSV.",
    )

    parser.add_argument(
        "--sd-ckpt",
        type=str,
        required=True,
        help="Path to the pretrained Stable Diffusion v2.1 checkpoint.",
    )

    parser.add_argument(
        "--stage1-ckpt",
        type=str,
        default=None,
        help="Stage-1 Lightning checkpoint used to resume Stage-2 training.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("HEIGHT", "WIDTH"),
        help="Training crop size. Default: 512 512.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Defaults to 2 for Stage 1 and 1 for Stage 2.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers. Default: 4.",
    )

    parser.add_argument(
        "--final-ckpt",
        type=str,
        default=None,
        help=(
            "Optional path for the final checkpoint. "
            "If omitted, it is saved in the configured checkpoint directory."
        ),
    )

    return parser.parse_args()


def build_dataloader(args):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(tuple(args.patch_size)),
            transforms.ToTensor(),
        ]
    )

    # Stage 1 and Stage 2 use the same training dataset.
    dataset = CustomDataset(
        args.train_csv,
        transform=train_transforms,
    )

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 2 if args.stage == 1 else 1

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"Training dataset: {args.train_csv}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Patch size: {tuple(args.patch_size)}")

    return dataloader


def build_callbacks(config):
    callbacks = []

    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))

    return callbacks


def prepare_stage2_checkpoint(model, stage1_ckpt):
    """
    Convert a Stage-1 checkpoint into a checkpoint compatible with
    the Stage-2 model structure.

    Stage 2 introduces parameters that are absent from the Stage-1
    checkpoint. We first load all compatible Stage-1 parameters with
    strict=False, then replace the checkpoint state_dict with the full
    initialized Stage-2 state_dict.

    This preserves the Stage-1 initialization while providing values
    for newly introduced Stage-2 parameters.
    """

    if stage1_ckpt is None:
        raise ValueError(
            "--stage1-ckpt is required when training with --stage 2."
        )

    stage1_ckpt = Path(stage1_ckpt)

    if not stage1_ckpt.is_file():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {stage1_ckpt}"
        )

    print(f"Loading Stage-1 checkpoint: {stage1_ckpt}")

    checkpoint = torch.load(
        str(stage1_ckpt),
        map_location="cpu",
    )

    if "state_dict" not in checkpoint:
        raise KeyError(
            "The supplied Stage-1 checkpoint does not contain 'state_dict'."
        )

    incompatible = model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )

    print(
        f"Stage-2 initialization: "
        f"{len(incompatible.missing_keys)} missing keys, "
        f"{len(incompatible.unexpected_keys)} unexpected keys."
    )

    # Replace the Stage-1 state_dict with the complete Stage-2 state_dict.
    checkpoint["state_dict"] = model.state_dict()

    # Lightning's older resume interface expects a checkpoint file.
    tmp = tempfile.NamedTemporaryFile(
        prefix="stage2_init_",
        suffix=".ckpt",
        delete=False,
    )
    tmp_path = tmp.name
    tmp.close()

    torch.save(checkpoint, tmp_path)
    del checkpoint

    print(f"Temporary Stage-2 initialization checkpoint: {tmp_path}")

    return tmp_path


def get_final_checkpoint_path(trainer, args):
    if args.final_ckpt is not None:
        path = Path(args.final_ckpt)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    checkpoint_callback = getattr(
        trainer,
        "checkpoint_callback",
        None,
    )

    if (
        checkpoint_callback is not None
        and checkpoint_callback.dirpath is not None
    ):
        ckpt_dir = Path(checkpoint_callback.dirpath)
    else:
        ckpt_dir = Path("./checkpoints")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return ckpt_dir / f"stage{args.stage}_final.ckpt"


def main():
    args = parse_args()

    print(f"Training Stage {args.stage}")

    config = OmegaConf.load(args.config)

    # The model-specific configuration can be selected from the command line.
    model_config_path = args.model_config or config.model.config
    print(f"Model config: {model_config_path}")

    model_config = OmegaConf.load(model_config_path)

    sd_ckpt_path = Path(args.sd_ckpt)
    if not sd_ckpt_path.is_file():
        raise FileNotFoundError(
            f"Stable Diffusion checkpoint not found: {sd_ckpt_path}"
        )

    # Use the user-provided Stable Diffusion v2.1 checkpoint for
    # the same base-model initialization used in the original training.
    model_config.params.sync_path = str(sd_ckpt_path)
    print(f"SD checkpoint: {sd_ckpt_path}")

    model = instantiate_from_config(model_config)

    dataloader = build_dataloader(args)
    callbacks = build_callbacks(config)

    converted_ckpt = None

    try:
        if args.stage == 1:
            trainer = pl.Trainer(
                callbacks=callbacks,
                **config.lightning.trainer,
            )

        else:
            converted_ckpt = prepare_stage2_checkpoint(
                model=model,
                stage1_ckpt=args.stage1_ckpt,
            )

            # Resume Stage 2 from the converted Stage-1 Lightning checkpoint.
            trainer = pl.Trainer(
                callbacks=callbacks,
                **config.lightning.trainer,
                resume_from_checkpoint=converted_ckpt,
            )

        trainer.fit(
            model,
            dataloader,
        )

        final_ckpt_path = get_final_checkpoint_path(
            trainer,
            args,
        )

        trainer.save_checkpoint(str(final_ckpt_path))

        print("Training finished.")
        print(f"Final checkpoint: {final_ckpt_path}")

    finally:
        # The converted checkpoint is needed only for Stage-2 initialization.
        if converted_ckpt is not None and os.path.exists(converted_ckpt):
            os.remove(converted_ckpt)


if __name__ == "__main__":
    main()
