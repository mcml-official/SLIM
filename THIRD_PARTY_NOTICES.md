# Third-Party Notices

This repository incorporates or adapts components from several open-source projects. We gratefully acknowledge the original authors and contributors.

## Latent Diffusion

Parts of the diffusion model implementation are based on:

**Latent Diffusion Models**  
CompVis / Machine Vision and Learning Group, LMU Munich  
https://github.com/CompVis/latent-diffusion

The original code is distributed under the MIT License.

## Stable Diffusion

The diffusion backbone and pretrained initialization used by this project are based on Stable Diffusion v2.1.

Stable Diffusion model weights are not distributed in this repository. Users are responsible for obtaining the required pretrained checkpoint separately and complying with the applicable model license.

## ELIC

Parts of the learned image compression implementation are based on or adapted from:

**Unofficial ELIC Implementation**  
Wei Jiang  
https://github.com/JiangWeibeta/ELIC

The original implementation is distributed under the Apache License 2.0.

Related work:

D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang,  
"ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding,"  
CVPR 2022.

## LPIPS

The perceptual loss implementation and pretrained LPIPS weights are based on:

**PerceptualSimilarity / LPIPS**  
Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang  
https://github.com/richzhang/PerceptualSimilarity

The original code is distributed under the BSD 2-Clause License.

## Other Dependencies

This project also relies on third-party libraries including PyTorch, torchvision, CompressAI, PyTorch Lightning, OpenCLIP, Transformers, xFormers, and other packages listed in `requirements.txt`.

Each dependency remains subject to its respective license.

## License Texts

Copies of the licenses for third-party source code incorporated or adapted in
this repository are provided in the `third_party_licenses/` directory:

- `third_party_licenses/latent-diffusion-LICENSE`
- `third_party_licenses/ELIC-LICENSE`
- `third_party_licenses/LPIPS-LICENSE`

The root `LICENSE` applies to the original code and modifications provided by
this project, except where third-party components are subject to their
respective upstream licenses and notices.
