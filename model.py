import torch
import torch.nn as nn
import torch.nn.functional as F

# YOLOv11的基础组件
class Conv(nn.Module):
    """标准卷积层：卷积 + 批归一化 + 激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, activation=nn.SiLU(inplace=True)):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """瓶颈层：1x1卷积降维 -> 3x3卷积 -> 1x1卷积升维"""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """C2f模块：YOLOv11的核心组件，改进的CSP结构"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels * 2, 1, 1)
        self.cv2 = Conv(hidden_channels * (2 + n), out_channels, 1, 1)
        self.m = nn.ModuleList(Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """空间金字塔池化快速版"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

# YOLOv11分类模型
class FireClassifier(nn.Module):
    def __init__(self, num_classes=3, input_size=640):
        super(FireClassifier, self).__init__()
        self.num_classes = num_classes
        
        # YOLOv11 Backbone
        self.backbone = nn.Sequential(
            # 初始卷积层
            Conv(3, 16, 3, 2),  # 640 -> 320
            Conv(16, 32, 3, 2),  # 320 -> 160
            C2f(32, 32, 1),
            Conv(32, 64, 3, 2),  # 160 -> 80
            C2f(64, 64, 2),
            Conv(64, 128, 3, 2),  # 80 -> 40
            C2f(128, 128, 2),
            Conv(128, 256, 3, 2),  # 40 -> 20
            C2f(256, 256, 4),
            Conv(256, 512, 3, 2),  # 20 -> 10
            C2f(512, 512, 2),
            SPPF(512, 512, 5),  # 空间金字塔池化
        )
        
        # 分类头
        self.head = nn.Sequential(
            Conv(512, 1024, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 前向传播通过backbone
        x = self.backbone(x)
        # 通过分类头
        x = self.head(x)
        return x

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 将silu改为relu以确保兼容性
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)