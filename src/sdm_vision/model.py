"""The ResNet-18 of notebooks 04 and 08."""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # Match the shape of the input so it can be added to the output.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ImageNetResNet(nn.Module):
    # Notebook 04's CifarResNet. stem="imagenet" uses the ImageNet input layer (7 x 7 convolution, stride 2, then a 3 x 3 max-pool);
    # stem="cifar" keeps notebook 04's 3 x 3 convolution, for 32 x 32 images.
    def __init__(self, num_classes=10, widths=(64, 128, 256, 512), stem="imagenet"):
        super().__init__()
        if stem == "imagenet":
            self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        stages, in_channels = [], widths[0]
        for i, width in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(nn.Sequential(ResidualBlock(in_channels, width, stride), ResidualBlock(width, width)))
            in_channels = width
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(widths[-1], num_classes)

    def embed(self, x):
        # The 512-dimensional feature vector before the final layer.
        return F.adaptive_avg_pool2d(self.stages(self.stem(x)), 1).flatten(1)

    def forward(self, x):
        return self.classifier(self.embed(x))
