import torch.nn as nn
import numpy as np
import torch
import unittest
from transform_utils import generate_random_tensor
from unet.layers import *


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
        batch_sz = 4
        model = UNet(3, 2)
        test_in = torch.from_numpy(np.arange(572*572*3*batch_sz).reshape(batch_sz, 3, 572, 572).astype(np.float32))
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 388, 388, 2))
        return

    def test_back_prop(self):
        '''batch_sz = 8
        priors = 6
        classes = 21
        model = SSD(priors_cnt=priors, classes_cnt=classes)
        test_in = torch.from_numpy(np.arange(300*300*3*batch_sz).reshape(batch_sz, 3, 300, 300).astype(np.float32))
        clf0_tgt = generate_random_tensor(batch_sz, priors * classes, 38, 38)
        reg0_tgt = generate_random_tensor(batch_sz, priors * 4, 38, 38)
        clf1_tgt = generate_random_tensor(batch_sz, priors * classes, 19, 19)
        reg1_tgt = generate_random_tensor(batch_sz, priors * 4, 19, 19)
        clf2_tgt = generate_random_tensor(batch_sz, priors * classes, 10, 10)
        reg2_tgt = generate_random_tensor(batch_sz, priors * 4, 10, 10)
        clf3_tgt = generate_random_tensor(batch_sz, priors * classes, 5, 5)
        reg3_tgt = generate_random_tensor(batch_sz, priors * 4, 5, 5)
        clf4_tgt = generate_random_tensor(batch_sz, priors * classes, 3, 3)
        reg4_tgt = generate_random_tensor(batch_sz, priors * 4, 3, 3)
        clf5_tgt = generate_random_tensor(batch_sz, priors * classes, 1, 1)
        reg5_tgt = generate_random_tensor(batch_sz, priors * 4, 1, 1)

        model.train()
        net_out = model.forward(test_in)
        target_out = (clf0_tgt, reg0_tgt, clf1_tgt, reg1_tgt, clf2_tgt, reg2_tgt, clf3_tgt, reg3_tgt,
                      clf4_tgt, reg4_tgt, clf5_tgt, reg5_tgt)
        l0 = torch.nn.functional.mse_loss(net_out[0], target_out[0])
        l1 = torch.nn.functional.mse_loss(net_out[1], target_out[1])
        l2 = torch.nn.functional.mse_loss(net_out[2], target_out[2])
        l3 = torch.nn.functional.mse_loss(net_out[3], target_out[3])
        l4 = torch.nn.functional.mse_loss(net_out[4], target_out[4])
        l5 = torch.nn.functional.mse_loss(net_out[5], target_out[5])
        l6 = torch.nn.functional.mse_loss(net_out[6], target_out[6])
        l7 = torch.nn.functional.mse_loss(net_out[7], target_out[7])
        l8 = torch.nn.functional.mse_loss(net_out[8], target_out[8])
        l9 = torch.nn.functional.mse_loss(net_out[9], target_out[9])
        l10 = torch.nn.functional.mse_loss(net_out[10], target_out[10])
        l11 = torch.nn.functional.mse_loss(net_out[11], target_out[11])

        total = l0+l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11
        total.backward()'''
        return


if __name__ == "__main__":
    unittest.main()
