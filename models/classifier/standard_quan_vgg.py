"""
Standard quantized VGG models for CIFAR-10.

These variants avoid the SDN multi-branch outputs used by `quan_vgg_cifar.py`
and match the archived ROBBIN VGG-16 INT8 checkpoint layout.
"""

import math

import torch.nn as nn
import torch.nn.init as init

from .quantization import quan_Conv2d, quan_Linear

__all__ = [
    "VGGStandard",
    "vgg11_quan_standard",
    "vgg11_bn_quan_standard",
    "vgg13_quan_standard",
    "vgg13_bn_quan_standard",
    "vgg16_quan_standard",
    "vgg16_bn_quan_standard",
    "vgg19_quan_standard",
    "vgg19_bn_quan_standard",
]


class VGGStandard(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            quan_Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            quan_Linear(512, num_classes),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def make_layers_standard(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        conv = quan_Conv2d(in_channels, value, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        in_channels = value

    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg11_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["A"]), num_classes)


def vgg11_bn_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["A"], batch_norm=True), num_classes)


def vgg13_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["B"]), num_classes)


def vgg13_bn_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["B"], batch_norm=True), num_classes)


def vgg16_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["D"]), num_classes)


def vgg16_bn_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["D"], batch_norm=True), num_classes)


def vgg19_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["E"]), num_classes)


def vgg19_bn_quan_standard(num_classes=10):
    return VGGStandard(make_layers_standard(cfg["E"], batch_norm=True), num_classes)
