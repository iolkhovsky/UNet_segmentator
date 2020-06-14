import torch.nn as nn


class ConvX2(nn.Module):

    def __init__(self, channels):
        super(ConvX2, self).__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=0),
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=0),
        self.bn2 = nn.BatchNorm2d(channels[2])
        return

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(self.bn1(x))
        x = self.conv2(x)
        x = nn.ReLU(self.bn2(x))
        return x


class DownSample(nn.Module):

    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvX2(channels)
        return

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):

    def __init__(self, channels):
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvX2(channels)
        pass

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
