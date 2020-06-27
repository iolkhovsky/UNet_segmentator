import numpy as np
import torch
from unet.layers import *
from unet.loss import compute_loss
from torch.nn.init import kaiming_normal_ as he_normal
import unittest


class UNet(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.conv1 = ConvX2(channels=[in_channels, 64, 64])

        self.downsample1 = DownSample(channels=[64, 128, 128])
        self.downsample2 = DownSample(channels=[128, 256, 256])
        self.downsample3 = DownSample(channels=[256, 512, 512])
        self.downsample4 = DownSample(channels=[512, 512, 512])

        self.upsample5 = UpSample(channels=[1024, 512, 256])
        self.upsample6 = UpSample(channels=[512, 256, 128])
        self.upsample7 = UpSample(channels=[256, 128, 64])
        self.upsample8 = UpSample(channels=[128, 64, 64])

        self.conv9 = nn.Conv2d(64, out_classes, kernel_size=1)

        he_normal(self.conv1)
        he_normal(self.conv9)
        return

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)
        x6 = self.upsample5(x5, x4)
        x7 = self.upsample6(x6, x3)
        x8 = self.upsample7(x7, x2)
        x9 = self.upsample8(x8, x1)
        out = self.conv9(x9)
        return out

    def __str__(self):
        return "UNet"


class TestSSDBasics(unittest.TestCase):

    def test_forward_pass(self):
        batch_sz = 2
        model = UNet(3, 2)
        test_in = torch.from_numpy(np.arange(572*572*3*batch_sz).reshape(batch_sz, 3, 572, 572).astype(np.float32))
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 2, 388, 388))
        return

    def test_back_prop(self):
        batch_sz = 1
        model = UNet(3, 2)

        input = torch.randn(batch_sz, 3, 572, 572)
        out = model.forward(input)
        target = torch.empty(batch_sz, 1, 388, 388, dtype=torch.long).random_(2)
        weights = torch.randn(batch_sz, 2)

        loss = compute_loss(prediction_tensor=out, target_maps=target, weights=weights, classes_cnt=2)
        loss.backward()
        return


if __name__ == "__main__":
    unittest.main()

