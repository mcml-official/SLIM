# SLIM

Official implementation of **"SLIM: Semantic-based Low-bitrate Image Compression for Machines"**.

This repository contains the training and inference code for our generative learned image compression method. The public release focuses on the main method, including Stage 1 training, Stage 2 training, dataset preparation, and image reconstruction.


\n
\n


## Overview

The model is trained in two stages.

Multiple rate configurations are provided under:

```text
configs/model/stage1/
configs/model/stage2/
```

The available settings are:

```text
rate1
rate2
rate4
rate10
```




## Environment

The release was tested with Python 3.10 and CUDA-enabled PyTorch.

A recommended setup is:

```bash
conda create -n slim python=3.10 -y
conda activate slim

pip install -r requirements.txt
```

The tested dependency versions are listed in `requirements.txt`.

A CUDA-capable GPU is required for training and inference.





## Stable Diffusion v2.1 Checkpoint

This project uses the pretrained **Stable Diffusion v2.1 base** checkpoint:

```text
v2-1_512-ema-pruned.ckpt
```

The Stable Diffusion checkpoint is **not distributed with this repository**. Please obtain it separately and comply with its applicable license.

The checkpoint path is supplied explicitly using:

```text
--sd-ckpt /path/to/v2-1_512-ema-pruned.ckpt
```





## Dataset Preparation

Training uses ImageNet images together with captions, class labels, and activation-based ROI masks.

The expected ImageNet directory layout is:

```text
/path/to/imagenet/train/
├── n01440764/
│   ├── *.JPEG
│   └── ...
├── n01443537/
│   ├── *.JPEG
│   └── ...
└── ...
```

ImageNet itself is not distributed with this repository.

Prepare the training metadata and ROI masks using:

```bash
python scripts/prepare_imagenet.py \
    --imagenet-root /path/to/imagenet/train \
    --output-csv ./data/train.csv \
    --mask-dir ./data/masks
```

For a small subset, for example:

```bash
python scripts/prepare_imagenet.py \
    --imagenet-root /path/to/imagenet/train \
    --output-csv ./data/train_subset.csv \
    --mask-dir ./data/masks_subset \
    --num-images 100
```

The resulting CSV contains:

```text
jpg
txt
class_id
mask
```

where `class_id` is the ImageNet synset identifier, such as `n02093754`.

The preparation script uses a pretrained ImageNet ResNet-50 to generate activation-based ROI masks and BLIP-2 to generate image captions.

Additional options are available with:

```bash
python scripts/prepare_imagenet.py --help
```





## Stage 1 Training

The following example trains the `rate1` configuration:

```bash
python train.py \
    --stage 1 \
    --config configs/train_stage1.yaml \
    --model-config configs/model/stage1/rate1.yaml \
    --train-csv ./data/train.csv \
    --sd-ckpt /path/to/v2-1_512-ema-pruned.ckpt \
    --final-ckpt ./checkpoints/stage1_rate1.ckpt
```

To train another rate point, replace:

```text
configs/model/stage1/rate1.yaml
```

with one of:

```text
configs/model/stage1/rate2.yaml
configs/model/stage1/rate4.yaml
configs/model/stage1/rate10.yaml
```

The default Stage 1 training configuration uses:

```text
max_steps: 100001
precision: 32
```





## Stage 2 Training

Stage 2 resumes from a Stage 1 Lightning checkpoint.

For `rate1`:

```bash
python train.py \
    --stage 2 \
    --config configs/train_stage2.yaml \
    --model-config configs/model/stage2/rate1.yaml \
    --train-csv ./data/train.csv \
    --sd-ckpt /path/to/v2-1_512-ema-pruned.ckpt \
    --stage1-ckpt ./checkpoints/stage1_rate1.ckpt \
    --final-ckpt ./checkpoints/stage2_rate1.ckpt
```

Use the corresponding Stage 2 configuration for the other rate points:

```text
configs/model/stage2/rate2.yaml
configs/model/stage2/rate4.yaml
configs/model/stage2/rate10.yaml
```

The default Stage 2 configuration uses:

```text
max_steps: 200001
precision: 32
```

Because Stage 2 resumes from the Stage 1 Lightning checkpoint, the training step state is continued rather than restarted from zero.





## Inference

The inference script reconstructs a single image using a trained Stage 2 checkpoint and the Stable Diffusion v2.1 base checkpoint.

For example, using the `rate1` model:

```bash
python inference.py \
    --config configs/model/stage2/rate1.yaml \
    --checkpoint ./checkpoints/stage2_rate1.ckpt \
    --sd-ckpt /path/to/v2-1_512-ema-pruned.ckpt \
    --input /path/to/input.jpg \
    --caption "a photo" \
    --output ./outputs/reconstruction.png \
    --ddim-steps 5
```

Replace the caption with text describing the input image.

The required inference inputs are:

- `--checkpoint`: trained Stage 2 checkpoint
- `--sd-ckpt`: pretrained Stable Diffusion v2.1 checkpoint
- `--input`: input image
- `--caption`: text conditioning for the input image

The model configuration can be selected with `--config`. Available configurations are:

```text
configs/model/stage2/rate1.yaml
configs/model/stage2/rate2.yaml
configs/model/stage2/rate4.yaml
configs/model/stage2/rate10.yaml
```

By default, `--ddim-steps` is set to `5`. The current sampler internally appends a terminal timestep.





## Rate Configurations

The bitrate-loss weights used by the provided configurations are:

| Configuration | Stage 1 `l_bpp_weight` | Stage 2 `l_bpp_weight` |
| --- | ---: | ---: |
| rate1 | 3.0 | 3.5 |
| rate2 | 6.0 | 7.0 |
| rate4 | 12.0 | 14.0 |
| rate10 | 30.0 | 35.0 |

Other core optimization settings are shared across the rate configurations unless specified otherwise in the YAML files.





## Citation

If you find this work useful, please cite:

```bibtex
<FINAL_BIBTEX>
```





## Acknowledgements

This project builds upon several open-source projects, including Latent Diffusion, Stable Diffusion, ELIC, LPIPS, CompressAI, PyTorch, and related libraries.

Please see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party acknowledgements and license information.





## License

License information for this repository is provided in the root `LICENSE` file. Third-party components remain subject to their respective licenses as described in `THIRD_PARTY_NOTICES.md`.

Third-party components remain subject to their respective licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.
