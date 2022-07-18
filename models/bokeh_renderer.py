import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blur_kernel import DiskBlur, SoftDiskBlur


class MPIBokehRenderer(nn.Module):
    def __init__(self, K_max=100, bins=32):
        super(MPIBokehRenderer, self).__init__()
        blur_kernels = nn.ModuleList()
        for i in range(K_max+1):
            if i == 0:
                blur_kernels.append(nn.Identity())
            else:
                blur_kernels.append(SoftDiskBlur(int(2 * i + 1)))
        self.blur_kernels = blur_kernels
        self.bins = bins

    def forward(self, mpi_image, mpi_alpha, K, disp_focus, gamma, norm=True, increase_focus_range=False, return_aux=False):
        if isinstance(gamma, torch.Tensor):
            mpi_image = mpi_image ** gamma.unsqueeze(1)
        else:
            mpi_image = mpi_image ** gamma

        mpi_image = torch.cat((mpi_image, torch.ones_like(mpi_alpha)), dim=2)

        bokeh = torch.zeros_like(mpi_image[:, 0])

        for i in range(self.bins):
            image = mpi_image[:, i]
            alpha = mpi_alpha[:, i]
            if isinstance(disp_focus, torch.Tensor) or isinstance(K, torch.Tensor):
                for b in range(image.shape[0]):
                    disp_focus_i = disp_focus[b] if isinstance(disp_focus, torch.Tensor) else disp_focus
                    K_i = K[b] if isinstance(K, torch.Tensor) else K
                    signed_disp_i = float((i+0.5)/self.bins - disp_focus_i)

                    defocus_i = abs(signed_disp_i)
                    if increase_focus_range:
                        if increase_focus_range is True:
                            defocus_i = defocus_i - 0.5 / self.bins
                        else:
                            defocus_i = defocus_i - increase_focus_range
                        if defocus_i < 0:
                            defocus_i = 0
                    kernel = self.blur_kernels[int(round(float(defocus_i * K_i)))]

                    bokeh[b:b+1] = bokeh[b:b+1].clone() * (1 - kernel(alpha[b:b+1])) + kernel(image[b:b+1] * alpha[b:b+1])
            else:
                signed_disp = float((i+0.5)/self.bins - disp_focus)

                defocus = abs(signed_disp)
                if increase_focus_range:
                    if increase_focus_range is True:
                        defocus = defocus - 0.5 / self.bins
                    else:
                        defocus = defocus - increase_focus_range  # 0.5 / self.bins
                    if defocus < 0:
                        defocus = 0
                kernel = self.blur_kernels[int(round(float(defocus * K)))]

                bokeh = bokeh * (1 - kernel(alpha)) + kernel(image * alpha)

        if norm is True:
            bokeh_final = bokeh[:, :-1] / bokeh[:, -1:]
        else:
            bokeh_final = bokeh[:, :-1]

        bokeh_final = bokeh_final.clamp(1e-10, 1e10) ** (1 / gamma)

        if return_aux:
            return bokeh_final, bokeh[:, -1:]
        else:
            return bokeh_final