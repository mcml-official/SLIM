from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from utils.common import instantiate_from_config


def parse_args():
    parser = ArgumentParser(
        description="Single-image reconstruction with the Stage-2 model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/stage2/rate1.yaml",
        help="Stage-2 model configuration YAML.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained Stage-2 checkpoint.",
    )

    parser.add_argument(
        "--sd-ckpt",
        type=str,
        required=True,
        help="Path to the pretrained Stable Diffusion v2.1 checkpoint.",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--caption",
        type=str,
        required=True,
        help="Caption conditioning text for the input image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reconstruction.png",
        help="Path for the reconstructed image.",
    )

    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=5,
        help=(
            "DDIM step setting passed to the sampler. "
            "The sampler appends a terminal timestep. Default: 5."
        ),
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )

    return parser.parse_args()


def load_image(path):
    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(path).convert("RGB")
    image = transform(image)

    # Same normalization used by CustomDataset:
    # [0, 1] -> [-1, 1]
    image = image * 2.0 - 1.0

    return image.unsqueeze(0)


def unwrap_condition(value):
    if isinstance(value, list):
        value = value[0]
    return value


def load_model(
    config_path,
    checkpoint_path,
    sd_ckpt_path,
    device,
    strict=True,
):
    print(f"Model config : {config_path}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"SD checkpoint: {sd_ckpt_path}")

    config = OmegaConf.load(config_path)

    # Use the pretrained Stable Diffusion v2.1 checkpoint.
    config.params.sync_path = sd_ckpt_path

    model = instantiate_from_config(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(
        state_dict,
        strict=strict,
    )

    if not strict:
        print(
            f"Checkpoint loading: "
            f"{len(incompatible.missing_keys)} missing keys, "
            f"{len(incompatible.unexpected_keys)} unexpected keys."
        )

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def reconstruct(model, image, caption, ddim_steps, device):
    image = image.to(device)

    # Stage-2 get_input() expects these four keys.
    #
    # mask and class_id are required by the original training model's
    # input interface, but they are not used by the reconstruction
    # sampling path below.
    dummy_mask = torch.zeros(
        (image.shape[0], 1, image.shape[2], image.shape[3]),
        dtype=image.dtype,
        device=device,
    )

    batch = {
        "jpg": image,
        "txt": [caption],
        "mask": dummy_mask,
        "class_id": ["dummy"],
    }

    _, cond = model.get_input(batch, "jpg")

    c_latent = unwrap_condition(cond["c_latent"])
    c_cross = unwrap_condition(cond["c_crossattn"])

    bpp = cond["q_bpp"]

    # Text conditioning for classifier-free guidance.
    uc_cross = model.get_unconditional_conditioning(
        c_cross.shape[0]
    )

    if isinstance(uc_cross, list):
        uc_cross = uc_cross[0]

    # Keep the conditioning dimensions compatible with the reconstruction sampling path.
    if (
        torch.is_tensor(uc_cross)
        and uc_cross.shape[1] != c_cross.shape[1]
    ):
        uc_cross = torch.zeros_like(c_cross)

    samples = model.sample_log(
        cond={
            "c_crossattn": [c_cross],
            "c_latent": [c_latent],
        },
        un_cond={
            "c_crossattn": [uc_cross],
            "c_latent": [c_latent],
        },
        steps=ddim_steps,
    )

    reconstruction = model.decode_first_stage(samples)

    # [-1, 1] -> [0, 1]
    reconstruction = (reconstruction + 1.0) / 2.0
    reconstruction = reconstruction.clamp(0.0, 1.0)

    return reconstruction, bpp


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference.")

    device = torch.device("cuda")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {input_path}"
        )

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not Path(args.sd_ckpt).is_file():
        raise FileNotFoundError(
            f"Stable Diffusion checkpoint not found: {args.sd_ckpt}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        sd_ckpt_path=args.sd_ckpt,
        device=device,
        strict=not args.non_strict,
    )

    image = load_image(input_path)

    reconstruction, bpp = reconstruct(
        model=model,
        image=image,
        caption=args.caption,
        ddim_steps=args.ddim_steps,
        device=device,
    )

    output_image = to_pil_image(
        reconstruction[0].detach().cpu()
    )
    output_image.save(output_path)

    if torch.is_tensor(bpp):
        bpp = bpp.detach().float().mean().item()

    print()
    print("Reconstruction finished.")
    print(f"Input       : {input_path}")
    print(f"Caption     : {args.caption}")
    print(f"Output      : {output_path}")
    print(f"Estimated BPP: {float(bpp):.6f}")


if __name__ == "__main__":
    main()
