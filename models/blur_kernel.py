import torch
import torch.nn as nn
import torch.nn.functional as F


class DiskBlur(nn.Module):
    def __init__(self, kernel_size):
        super(DiskBlur, self).__init__()
        r = kernel_size // 2
        x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r)+1), torch.arange(-int(r), int(r)+1))
        kernel = torch.le(x_grid**2 + y_grid**2, r**2).float()
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.pad = nn.ReplicationPad2d(r)
        self.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        out = self.pad(x)
        ch = x.shape[1]
        out = F.conv2d(out, self.weight.expand(ch, 1, self.kernel_size, self.kernel_size), padding=0, groups=ch)
        return out


# class DiskBlur(nn.Module):
#     def __init__(self, kernel_size_max=101):
#         super(DiskBlur, self).__init__()
#         r = kernel_size_max // 2
#         x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r)+1), torch.arange(-int(r), int(r)+1))
#         kernel = torch.le(x_grid**2 + y_grid**2, r**2).float()
#         # kernel = kernel / kernel.sum()
#         kernel = kernel.expand(1, 1, kernel_size_max, kernel_size_max)
#         self.kernel_size_max = kernel_size_max
#         # self.pad = nn.ReplicationPad2d(r)
#         self.weight = nn.Parameter(kernel, requires_grad=False)
#
#     def forward(self, x, r):
#         out = F.pad(x, pad=(r, r, r, r), mode='replicate')
#         size = 2 * r + 1
#         weight = F.interpolate(self.weight, size=(size, size), mode='bilinear', align_corners=True)
#         weight = weight / weight.sum()
#         ch = x.shape[1]
#         out = F.conv2d(out, weight.expand(ch, 1, size, size), padding=0, groups=ch)
#         return out


class SoftDiskBlur(nn.Module):
    def __init__(self, kernel_size):
        super(SoftDiskBlur, self).__init__()
        r = kernel_size // 2
        x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r)+1), torch.arange(-int(r), int(r)+1))
        kernel = 0.5 + 0.5 * torch.tanh(0.25 * (r**2 - x_grid**2 - y_grid**2) + 0.5)
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.pad = nn.ReplicationPad2d(r)
        self.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        out = self.pad(x)
        ch = x.shape[1]
        out = F.conv2d(out, self.weight.expand(ch, 1, self.kernel_size, self.kernel_size), padding=0, groups=ch)
        return out


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma=None):
        super(GaussianBlur, self).__init__()
        r = kernel_size // 2
        if sigma is None:
            sigma = 0.3 * (r - 1) + 0.8
        x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r)+1), torch.arange(-int(r), int(r)+1))
        kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.pad = nn.ReplicationPad2d(r)
        self.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        out = self.pad(x)
        ch = x.shape[1]
        out = F.conv2d(out, self.weight.expand(ch, 1, self.kernel_size, self.kernel_size), padding=0, groups=ch)
        return out
