# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if dist.get_rank() == 0 and pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


def replace(layers, pretrained):
    for name, module in layers.named_children():
        if isinstance(module, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=0,
                bias=True if module.bias else False,
            )
            if pretrained:
                new_conv.weight.data.copy_(module.weight.clone())
            new_conv = nn.Sequential(nn.ReplicationPad2d(module.padding[0]), new_conv)
            layers.add_module(name, new_conv)
        else:
            replace(module, pretrained)

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        old_conv1 = self.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels=old_conv1.in_channels + 1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=True if old_conv1.bias else False,
        )
        if pretrained:
            new_conv1.weight[:, :3, :, :].data.copy_(old_conv1.weight.clone())
        self.encoder.conv1 = new_conv1

        replace(self.encoder, pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.img_mean = torch.tensor([0.485, 0.456, 0.406])
        self.img_mean = self.img_mean.view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225])
        self.img_std = self.img_std.view(1, 3, 1, 1)

    def forward(self, input_image, input_disp):
        # normalize before going into network
        ref_images_normalized = (input_image - self.img_mean.to(input_image.device)) / self.img_std.to(input_image.device)

        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = torch.cat((ref_images_normalized, input_disp), dim=1)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        conv1_out = self.encoder.relu(x)
        block1_out = self.encoder.layer1(self.encoder.maxpool(conv1_out))
        block2_out = self.encoder.layer2(block1_out)
        block3_out = self.encoder.layer3(block2_out)
        block4_out = self.encoder.layer4(block3_out)

        return conv1_out, block1_out, block2_out, block3_out, block4_out


class ShufflenetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(ShufflenetEncoder, self).__init__()

        shufflenets = {0.5: models.shufflenet_v2_x0_5,
                       1.0: models.shufflenet_v2_x1_0,
                       1.5: models.shufflenet_v2_x1_5,
                       2.0: models.shufflenet_v2_x2_0}

        if num_layers not in shufflenets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_layers == 0.5:
            self.num_ch_enc = np.array([24, 24, 48, 96, 192])
        elif num_layers == 1.0:
            self.num_ch_enc = np.array([24, 24, 116, 232, 464])
        elif num_layers == 1.5:
            self.num_ch_enc = np.array([24, 24, 176, 352, 704])
        else:
            self.num_ch_enc = np.array([24, 24, 244, 488, 976])


        self.encoder = shufflenets[num_layers](pretrained)

        old_conv1 = self.encoder.conv1[0]
        new_conv1 = nn.Conv2d(
            in_channels=old_conv1.in_channels + 1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=True if old_conv1.bias else False,
        )
        if pretrained:
            new_conv1.weight[:, :3, :, :].data.copy_(old_conv1.weight.clone())
        self.encoder.conv1[0] = new_conv1

        replace(self.encoder, pretrained)

        self.img_mean = torch.tensor([0.485, 0.456, 0.406])
        self.img_mean = self.img_mean.view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225])
        self.img_std = self.img_std.view(1, 3, 1, 1)

    def forward(self, input_image, input_disp):
        # normalize before going into network
        ref_images_normalized = (input_image - self.img_mean.to(input_image.device)) / self.img_std.to(input_image.device)

        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = torch.cat((ref_images_normalized, input_disp), dim=1)
        conv1_out = self.encoder.conv1(x)
        block1_out = self.encoder.maxpool(conv1_out)
        block2_out = self.encoder.stage2(block1_out)
        block3_out = self.encoder.stage3(block2_out)
        block4_out = self.encoder.stage4(block3_out)

        return conv1_out, block1_out, block2_out, block3_out, block4_out