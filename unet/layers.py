import torch.nn as nn
from torch import cat
import torch.nn.functional as F


class ConvX2(nn.Module):

    def __init__(self, channels):
        super(ConvX2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.act2 = nn.ReLU()
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x


class DownSample(nn.Module):

    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_x2 = ConvX2(channels)
        return

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_x2(x)
        return x


class UpSample(nn.Module):

    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_x2 = ConvX2(channels)
        pass

    def forward(self, x, residual):
        x1 = self.upsample(x)

        y = (residual.size()[2] - x1.size()[2]) // 2
        x = (residual.size()[3] - x1.size()[3]) // 2
        x2 = F.pad(residual, [-x, -x, -y, -y])

        x = cat([x2, x1], dim=1)
        x = self.conv_x2(x)
        return x
