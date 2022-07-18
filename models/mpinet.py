import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ResnetEncoder, ShufflenetEncoder
from models.decoder import DepthDecoder
from models.blur_kernel import GaussianBlur


class MPINet(nn.Module):
    def __init__(self, encoder_layers=50, encoder_pretrained=True, decoder_use_alpha=True, num_output_channels=5, bins=32):
        super(MPINet, self).__init__()
        self.num_output_channels = num_output_channels
        if encoder_layers > 10:
            self.encoder = ResnetEncoder(num_layers=encoder_layers, pretrained=encoder_pretrained)
        else:
            self.encoder = ShufflenetEncoder(num_layers=encoder_layers, pretrained=encoder_pretrained)
        self.decoder = DepthDecoder(
            # Common params
            num_ch_enc=self.encoder.num_ch_enc,
            use_alpha=decoder_use_alpha,
            num_output_channels=self.num_output_channels,
            scales=range(1),
            use_skips=True,
        )
        self.bins = bins
        self.gaussian = GaussianBlur(5)

    def forward(self, image, disp):
        disp_discrete = torch.linspace(0.5/self.bins, 1-0.5/self.bins, self.bins, dtype=torch.float32, device=image.device)  # S
        disp_discrete = disp_discrete[None, :, None, None].repeat(image.shape[0], 1, 1, 1)  # BXSx1x1

        disp_mask = 1.0 * (disp_discrete-0.5/self.bins <= disp) * (disp <= disp_discrete+0.5/self.bins)
        mpi_disp_mask = disp_mask.unsqueeze(2)

        conv1_out, block1_out, block2_out, block3_out, block4_out = self.encoder(image, disp)

        # outputs = self.decoder([conv1_out, block1_out, block2_out, block3_out, block4_out], disp_mask)
        # ########## use the codes below if CUDA out of memory ##########
        B, C, H, W = image.shape
        outputs = torch.zeros(B, self.bins, self.num_output_channels, H, W, dtype=torch.float32, device=image.device)
        for i in range(self.bins):
            outputs[:, i, :, :, :] = self.decoder([conv1_out, block1_out, block2_out, block3_out, block4_out], disp_mask[:, i:i+1, :, :])[:, 0]
        # ################################################################

        if self.num_output_channels == 4:
            mpi_rgb = outputs[:, :, :3, :, :]  # BxSx3xHxW
            mpi_alpha = outputs[:, :, 3:, :, :]  # BxSx1xHxW
            return mpi_rgb, mpi_alpha, mpi_disp_mask
        elif self.num_output_channels == 5:
            mpi_inpaint_rgb = outputs[:, :, :3, :, :]
            mpi_blend_weight = outputs[:, :, 3:4, :, :]
            mpi_alpha = outputs[:, :, 4:, :, :]
            return mpi_inpaint_rgb, mpi_blend_weight, mpi_alpha, mpi_disp_mask
        elif self.num_output_channels == 3:
            mpi_displacement = outputs[:, :, :2, :, :]
            mpi_alpha = outputs[:, :, 2:, :, :]
            return mpi_displacement, mpi_alpha, mpi_disp_mask
        elif self.num_output_channels == 2:
            mpi_blend_weight = outputs[:, :, :1, :, :]
            mpi_alpha = outputs[:, :, 1:, :, :]
            return mpi_blend_weight, mpi_alpha, mpi_disp_mask
        elif self.num_output_channels == 1:
            mpi_alpha = outputs
            return mpi_alpha, mpi_disp_mask
