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

    inpnet = FFCResNetGenerator(**cfg.inpnet).to(device)
    checkpoint = torch.load('/data4/pengjuewen/Code/lama/LaMa_models/lama-places/lama-fourier/models/best_resave_pjw.pth')
    # checkpoint = torch.load(args.inpnet_checkpoint_path)
    inpnet.load_state_dict(checkpoint['model'])
    inpnet = torch.nn.DataParallel(inpnet)

    mpinet = MPINet(**cfg.mpinet).to(device)
    mpinet = torch.nn.DataParallel(mpinet)
    checkpoint = torch.load(args.mpinet_checkpoint_path)
    mpinet.load_state_dict(checkpoint['model'])

    aupnet = AUPNet(**cfg.aupnet).to(device)
    aupnet = torch.nn.DataParallel(aupnet)
    checkpoint = torch.load(args.aupnet_checkpoint_path)
    aupnet.load_state_dict(checkpoint['model'])

    renderer = MPIBokehRenderer(**cfg.renderer).to(device)
    renderer = torch.nn.DataParallel(renderer)

    inpnet.eval()
    mpinet.eval()
    aupnet.eval()

    K = args.K
    disp_focus = args.disp_focus
    gamma = args.gamma

    blend_weight_dilate_size = args.blend_weight_dilate_size
    inpaint_mask_dilate_size = args.inpaint_mask_dilate_size
    inpaint_size = args.inpaint_size


    image = cv2.imread(args.image_path).astype(np.float32) / 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    disp = cv2.imread(args.disp_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    disp_re = resize(disp, short_size=inpaint_size, ensure_multiple_of=8)
    disp_grad_y = cv2.Sobel(disp_re, ddepth=-1, dx=0, dy=1)
    disp_grad_x = cv2.Sobel(disp_re, ddepth=-1, dx=1, dy=0)
    disp_grad_y = cv2.GaussianBlur(disp_grad_y, (7, 7), 0)
    disp_grad_x = cv2.GaussianBlur(disp_grad_x, (7, 7), 0)
    disp_grad = np.stack((disp_grad_x, disp_grad_y), axis=-1)
    mask = (np.abs(disp_grad).sum(axis=-1, keepdims=True) > 0.2).astype(np.float32)

    _, label_mask = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for i in range(1, label_mask.max() + 1):
        if (label_mask == i).sum() < 1 / 10000 * label_mask.shape[0] * label_mask.shape[1]:
            mask[label_mask == i] = 0

    with torch.no_grad():
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
        disp = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).to(device)
        disp_grad_re = torch.from_numpy(disp_grad).permute(2, 0, 1).unsqueeze(0).to(device)
        inpaint_mask_re = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(device)

        _, _, H, W = image.shape

        image_re = resize(image, short_size=inpaint_size, ensure_multiple_of=8)
        inpaint_mask_re = mask_generator(
            disp_grad_re, inpaint_mask_re,
            iteration=int(2 * K * min(1, inpaint_size/min(H, W))),
            dilate_kernel_size=inpaint_mask_dilate_size
        )
        inpaint_bg_re = inpnet(image_re * (1 - inpaint_mask_re), inpaint_mask_re)
        inpaint_mask = F.interpolate(inpaint_mask_re, size=(H, W), mode='bilinear', align_corners=True)
        inpaint_bg = F.interpolate(inpaint_bg_re, size=(H, W), mode='bilinear', align_corners=True)
        bg = inpaint_mask * inpaint_bg + (1 - inpaint_mask) * image

        base_size = min(512, int(min(H, W) * 30 / K))
        image_re = resize(image, short_size=base_size, ensure_multiple_of=32)
        disp_re = resize(disp, short_size=base_size, ensure_multiple_of=32)
        mpi_blend_weight_re, mpi_alpha_re, mpi_disp_mask_re = mpinet(image_re, disp_re)
        mpi_blend_weight_re = dilate(mpi_blend_weight_re[0], kernel_size=blend_weight_dilate_size).unsqueeze(0)
        mpi_blend_weight = F.interpolate(mpi_blend_weight_re[0], size=(H, W), mode='bilinear', align_corners=True).unsqueeze(0)

        mpi_rgb = mpi_blend_weight * bg.unsqueeze(1) + (1 - mpi_blend_weight) * image.unsqueeze(1)

        scale = max(H / image_re.shape[2], 1)

        for _ in range(int(np.log2(scale))):
            image_re = F.interpolate(image, size=(image_re.shape[2]*2, image_re.shape[3]*2), mode='bilinear', align_corners=True)
            disp_re = F.interpolate(disp, size=(disp_re.shape[2]*2, disp_re.shape[3]*2), mode='bilinear', align_corners=True)
            mpi_alpha_re = aupnet(image_re.clamp(0, 1), disp_re, mpi_alpha_re)

        image_re = resize(image, short_size=min(H, W), ensure_multiple_of=2)
        disp_re = resize(disp, short_size=min(H, W), ensure_multiple_of=2)
        mpi_alpha_re = F.interpolate(mpi_alpha_re[0], size=(image_re.shape[2]//2, image_re.shape[3]//2), mode='bilinear', align_corners=True).unsqueeze(0)
        mpi_alpha_re = aupnet(image_re.clamp(0, 1), disp_re, mpi_alpha_re)
        mpi_alpha = F.interpolate(mpi_alpha_re[0], size=(H, W), mode='bilinear', align_corners=True).unsqueeze(0)

        bokeh_pred = renderer(mpi_rgb, mpi_alpha, K, disp_focus, gamma, norm=True, increase_focus_range=True)

        b, s, _, h, w = mpi_alpha.shape
        mpi_disp = torch.linspace(0.5 / s, 1 - 0.5 / s, s).reshape(1, s, 1, 1, 1).expand(b, s, 1, h, w).to(device)
        disp_reconstruct = renderer(mpi_disp, mpi_alpha, 0, disp_focus, 1, True)

    image = image[0].cpu().clone().permute(1, 2, 0).numpy()
    disp = disp[0][0].cpu().clone().numpy()
    disp_reconstruct = disp_reconstruct[0][0].cpu().clone().numpy()
    inpaint_mask = inpaint_mask[0][0].cpu().clone().numpy()
    bokeh_pred = bokeh_pred[0].cpu().clone().permute(1, 2, 0).detach().numpy()
    bg = bg[0].cpu().clone().permute(1, 2, 0).detach().numpy()

    cv2.imwrite(os.path.join(save_dir, 'image.jpg'), image[..., ::-1] * 255)
    cv2.imwrite(os.path.join(save_dir, 'bg.jpg'), bg[..., ::-1] * 255)
    cv2.imwrite(os.path.join(save_dir, 'disp.jpg'), disp * 255)
    cv2.imwrite(os.path.join(save_dir, 'disp_reconstruct.jpg'), disp_reconstruct * 255)
    cv2.imwrite(os.path.join(save_dir, 'inpaint_mask.jpg'), inpaint_mask * 255)
    cv2.imwrite(os.path.join(save_dir, 'bokeh_pred.jpg'), bokeh_pred[..., ::-1] * 255)


if __name__ == '__main__':
    main()
