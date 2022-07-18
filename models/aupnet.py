import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import *


class AUPNet(nn.Module):
    def __init__(self, bins):
        super(AUPNet, self).__init__()

        self.conv_image = nn.Sequential(
            ConvBlock(in_channels=4, out_channels=16, kernel_size=3, stride=1),
            ConvBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )
        self.conv_alpha = ConvBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)  # ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.upsample = nn.Sequential(
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv_output = nn.Sequential(
            ConvBlock(in_channels=48, out_channels=32, kernel_size=3, stride=1),
            ConvBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            ConvNxN(in_channels=16, out_channels=1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )
        self.bins = bins

        self.img_mean = torch.tensor([0.485, 0.456, 0.406])
        self.img_mean = self.img_mean.view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225])
        self.img_std = self.img_std.view(1, 3, 1, 1)

    def forward(self, image, disp, mpi_alpha):
        image_norm = (image - self.img_mean.to(image.device)) / self.img_std.to(image.device)

        B, C_image, H_image, W_image = image_norm.shape
        _, S, C_alpha, H_alpha, W_alpha = mpi_alpha.shape
        assert H_image == 2 * H_alpha
        assert W_image == 2 * W_alpha
        assert S == self.bins

        mpi_alpha_up = torch.zeros(B, S, C_alpha, H_image, W_image, dtype=torch.float32, device=image.device)
        for i in range(S):
            # feat_image = self.conv_image(image_norm)
            feat_image = self.conv_image(torch.cat((image_norm, disp), dim=1))
            feat_alpha = self.conv_alpha(mpi_alpha[:, i])
            x = torch.cat((self.downsample(feat_image), feat_alpha), dim=1)
            x = self.upsample(x)
            mpi_alpha_up[:, i] = self.conv_output(torch.cat((x, feat_image), dim=1))

        return mpi_alpha_up
