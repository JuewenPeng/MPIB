#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import numpy as np
import yaml
import argparse
from munch import DefaultMunch

import torch
import torch.nn.functional as F

from models import AUPNet, MPINet, MPIBokehRenderer, FFCResNetGenerator
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--config_path', type=str, default='configs/default.yaml')
    parser.add_argument('--gpus', type=str, default='0')

    # Checkpoint
    parser.add_argument('--mpinet_checkpoint_path', type=str, default='./checkpoints/mpinet.pth')
    parser.add_argument('--aupnet_checkpoint_path', type=str, default='./checkpoints/aupnet.pth')
    parser.add_argument('--inpnet_model', type=str, default='lama', help='choose from "edge_connect", "madf", "lama"')
    parser.add_argument('--inpnet_checkpoint_path', type=str, default='./checkpoints/inpnet.pth')

    # Input
    parser.add_argument('--image_path', type=str, default='./inputs/new_11_all-in-focus.jpg')
    parser.add_argument('--disp_path', type=str, default='./inputs/new_11_disparity.jpg')
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--K', type=float, default=60, help='blur parameter (<200)')
    parser.add_argument('--disp_focus', type=float, default=207/255, help='refocused disparity (0~1)')
    parser.add_argument('--gamma', type=float, default=2.2, help='gamma value')
    parser.add_argument('--inpaint_mask_dilate_size', type=int, default=5, help='dilation kernel size of inpaint mask')
    parser.add_argument('--blend_weight_dilate_size', type=int, default=5, help='dilation kernel size of blend weight')
    parser.add_argument('--inpaint_size', type=int, default=1024, help='image size for inpainting model')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = DefaultMunch.fromDict(cfg, 'BokehConfig')

    checkpoint = torch.load(args.mpinet_checkpoint_path)
    new_checkpoint = dict()
    new_checkpoint['model'] = checkpoint['model']
    torch.save(new_checkpoint, 'mpinet.pth')

    checkpoint = torch.load(args.aupnet_checkpoint_path)
    new_checkpoint = dict()
    new_checkpoint['model'] = checkpoint['model']
    torch.save(new_checkpoint, 'aupnet.pth')

if __name__ == '__main__':
    main()
