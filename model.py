# A YOLOv13-style (DS-based blocks) image classifier implemented in pure PyTorch.
# No pretrained weights are used.

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None) -> int:
    """Auto padding to keep spatial size for stride=1."""
    return k // 2 if p is None else p


class ConvBNAct(nn.Module):
    """Standard Conv2d + BN + SiLU (YOLO-style)."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """
    Depthwise Separable Conv:
      depthwise(kxk, groups=in_ch) + pointwise(1x1)
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k=k, s=s, g=c1)
        self.pw = ConvBNAct(c1, c2, k=1, s=1, g=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSBottleneck(nn.Module):
    """
    YOLO-style bottleneck using DSConv as the 3x3 operator.
    """
    def __init__(self, c: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(round(c * expansion))
        self.cv1 = ConvBNAct(c, hidden, k=1, s=1)
        self.cv2 = DSConv(hidden, c, k=3, s=1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k2(nn.Module):
    """
    CSP-style block (C3-family) built with DSBottleneck.
    - "k2" here is treated as a lightweight variant; we keep the DS bottlenecks.
    """
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(round(c2 * expansion))
        self.cv1 = ConvBNAct(c1, hidden, k=1, s=1)
        self.cv2 = ConvBNAct(c1, hidden, k=1, s=1)
        self.m = nn.Sequential(*[DSBottleneck(hidden, shortcut=shortcut, expansion=1.0) for _ in range(n)])
        self.cv3 = ConvBNAct(hidden * 2, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPF(nn.Module):
    """SPPF block (fast spatial pyramid pooling) used widely in YOLO backbones."""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = ConvBNAct(c1, hidden, k=1, s=1)
        self.cv2 = ConvBNAct(hidden * 4, c2, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class FireClassifier(nn.Module):
    """
    YOLOv13-style image classifier (DS-based blocks).
    Args:
        num_classes: number of classes, default 3
        input_size: expected input resolution (H=W), default 640
    """
    def __init__(self, num_classes: int = 3, input_size: int = 640):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)

        # A compact backbone similar in spirit to YOLO-family downsampling (x2 each stage).
        # You can scale base channels if you want a larger/smaller model.
        base = 64

        self.stem = ConvBNAct(3, base, k=3, s=2)            # 640 -> 320
        self.stage1 = nn.Sequential(
            DSConv(base, base * 2, k=3, s=2),               # 320 -> 160
            DSC3k2(base * 2, base * 2, n=2),
        )
        self.stage2 = nn.Sequential(
            DSConv(base * 2, base * 4, k=3, s=2),           # 160 -> 80
            DSC3k2(base * 4, base * 4, n=4),
        )
        self.stage3 = nn.Sequential(
            DSConv(base * 4, base * 8, k=3, s=2),           # 80 -> 40
            DSC3k2(base * 8, base * 8, n=4),
        )
        self.stage4 = nn.Sequential(
            DSConv(base * 8, base * 16, k=3, s=2),          # 40 -> 20
            DSC3k2(base * 16, base * 16, n=2),
        )
        self.sppf = SPPF(base * 16, base * 16)

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(base * 16, self.num_classes)

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.sppf(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional sanity check (kept lightweight)
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input shape [B,3,H,W], got {tuple(x.shape)}")

        feat = self.forward_features(x)
        v = self.pool(feat).flatten(1)   # [B, C]
        v = self.dropout(v)
        logits = self.fc(v)              # [B, num_classes]
        return logits

    def _init_weights(self) -> None:
        # No pretrained weights: initialize from scratch.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Quick smoke test
    model = FireClassifier(num_classes=3, input_size=640)
    x = torch.randn(2, 3, 640, 640)
    y = model(x)
    print("logits:", y.shape)  # [2, 3]
