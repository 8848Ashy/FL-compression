"""ResNet-9 for CIFAR (davidcpage-style true 9-layer), GroupNorm federated-safe.

Structure (d ~ 6.57M, lambda_had = 2^23/d ~ 1.277):
  prep:  Conv3x3(3->64)   GN ReLU          32x32
  layer1: Conv3x3(64->128,s2) GN ReLU + 1 residual block(128)   16x16
  layer2: Conv3x3(128->256,s2) GN ReLU + MaxPool2                4x4
  layer3: Conv3x3(256->512,s2) GN ReLU + 1 residual block(512)   2x2
  classifier: GAP -> Linear(512->num_classes)

Normalization is GroupNorm(8, C) everywhere: BN's running_mean/var are buffers
that our FedAvg pipeline does NOT aggregate (only parameters() are transmitted),
so BatchNorm would desync train/test statistics under federated training.
"""
import torch.nn as nn


class _Block(nn.Module):
    """Same-channel double-conv residual block (identity skip)."""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return self.relu(x + out)


def _gn(c):
    return nn.GroupNorm(8, c)


class CIFARResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), _gn(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), _gn(128),
            nn.ReLU(inplace=True), _Block(128))
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), _gn(256),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), _gn(512),
            nn.ReLU(inplace=True), _Block(512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(self.pool(x).flatten(1))


if __name__ == "__main__":
    m = CIFARResNet9()
    d = sum(p.numel() for p in m.parameters())
    lam = (1 << (d - 1).bit_length()) / d
    print("params d =", d)
    print("hadamard padded D =", 1 << (d - 1).bit_length(), " lambda_had = %.4f" % lam)
