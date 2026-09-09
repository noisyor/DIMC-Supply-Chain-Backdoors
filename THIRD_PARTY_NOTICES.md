# Third-party notices

The model implementation contains code derived from Meta's DiT project. Original copyright notices and source references are retained. The upstream DiT license is Creative Commons Attribution-NonCommercial 4.0 International, reproduced in `licenses/DiT-CC-BY-NC-4.0.txt`.

- DiT source: https://github.com/facebookresearch/DiT
- DiT license: https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt
- This repository adapts DiT to generate CIFAR-10-sized images, including the DiT-N/2 model. The original DiT authors retain credit for their implementation.
- External runtime dependencies (PyTorch, torchvision, timm, NumPy, safetensors, Pillow) are installed separately and retain their respective upstream licenses.

The exact version of the original DiT code used for this adaptation was not recorded. This repository does not claim to be an official upstream release. Original contributions owned by the repository contributors are released under the root MIT license, including original evaluation helpers, documentation, data, RTL, and checkpoints. Third-party code, weights, data, and derivative portions remain subject to their applicable upstream terms. The MIT grant applies only to rights held by the contributors and does not relicense third-party material.

The classifier uses the archived VGG quantization implementation and clean checkpoint from the ROBBIN experiment code, with AT/CT and white-patch fine-tuning. This package uses that model for saved input-trigger experiments; it does not require a Rowhammer implementation. Original source notices are preserved.
