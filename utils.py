#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
from numba import jit, float32

import torch
import torch.nn as nn
import torch.nn.functional as F

import cupy
import re


kernel_Softsplat_updateOutput = '''
    extern "C" __global__ void kernel_Softsplat_updateOutput(
        const int n,
        const float* image,
        const float* flow,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX = ( intIndex                                                    ) % SIZE_3(output);
        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);
        assert(isfinite(fltOutputX) == true);
        assert(isfinite(fltOutputY) == true);
        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;
        float fltNorthwest = ((float) (intSoutheastX) - fltOutputX) * ((float) (intSoutheastY) - fltOutputY);
        float fltNortheast = (fltOutputX - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY);
        float fltSouthwest = ((float) (intNortheastX) - fltOutputX) * (fltOutputY - (float) (intNortheastY));
        float fltSoutheast = (fltOutputX - (float) (intNorthwestX)) * (fltOutputY - (float) (intNorthwestY));
        if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(image, intN, intC, intY, intX) * fltNorthwest);
        }
        if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(image, intN, intC, intY, intX) * fltNortheast);
        }
        if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(image, intN, intC, intY, intX) * fltSouthwest);
        }
        if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(image, intN, intC, intY, intX) * fltSoutheast);
        }
    } }
'''

def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end


# @cupy.util.memoize(for_each_device=True)
@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    # return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
    return cupy.RawModule(code=strKernel).get_function(strFunction)
# end


class _FunctionSoftsplat(torch.autograd.Function):
    @staticmethod
    def forward(self, image, flow):
        output = torch.zeros_like(image)

        if image.is_cuda == True:
            n = output.nelement()
            cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {
                'image': image,
                'flow': flow,
                'output': output
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=(
                    cupy.int(n),
                    image.data_ptr(),
                    flow.data_ptr(),
                    output.data_ptr()
                )
            )

        elif image.is_cuda == False:
            raise NotImplementedError()

        # end

        return output
    # end
# end


def FunctionSoftsplat(image, flow):
    output = _FunctionSoftsplat.apply(image, flow)

    return output
# end


class ModuleSoftsplat(nn.Module):
    def __init__(self):
        super(ModuleSoftsplat, self).__init__()

    def forward(self, image, flow):
        return FunctionSoftsplat(image, flow)


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        kernel_x = torch.Tensor([[-1.,  0.,  1.],
                                 [-2.,  0.,  2.],
                                 [-1.,  0.,  1.]]).expand(1, 1, 3, 3)
        kernel_y = torch.Tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).expand(1, 1, 3, 3)
        self.kernel_x = nn.Parameter(kernel_x, requires_grad=False)
        self.kernel_y = nn.Parameter(kernel_y, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, inputs):
        inputs = self.pad(inputs)
        ch = inputs.shape[1]
        grad_x = F.conv2d(inputs, self.kernel_x.expand(ch, 1, 3, 3), padding=0, groups=ch)
        grad_y = F.conv2d(inputs, self.kernel_y.expand(ch, 1, 3, 3), padding=0, groups=ch)
        return torch.cat((grad_x, grad_y), dim=1)


def dilate(inputs, kernel_size=3):
    return F.max_pool2d(inputs, kernel_size=kernel_size, stride=1, padding=kernel_size//2)


def erode(inputs, kernel_size=3):
    return -F.max_pool2d(-inputs, kernel_size=kernel_size, stride=1, padding=kernel_size//2)


def resize(inputs, short_size=640, ensure_multiple_of=32):
    if isinstance(inputs, np.ndarray):
        h, w, = inputs.shape[:2]
        h_re = min(h, max(short_size, short_size * h / w))
        w_re = min(w, max(short_size, short_size * w / h))
        h_re = int((h_re + ensure_multiple_of - 1) // ensure_multiple_of * ensure_multiple_of)
        w_re = int((w_re + ensure_multiple_of - 1) // ensure_multiple_of * ensure_multiple_of)
        return cv2.resize(inputs, dsize=(w_re, h_re), interpolation=cv2.INTER_LINEAR)
    elif isinstance(inputs, torch.Tensor):
        h, w = inputs.shape[-2:]
        h_re = min(h, max(short_size, short_size * h / w))
        w_re = min(w, max(short_size, short_size * w / h))
        h_re = int((h_re + ensure_multiple_of - 1) // ensure_multiple_of * ensure_multiple_of)
        w_re = int((w_re + ensure_multiple_of - 1) // ensure_multiple_of * ensure_multiple_of)
        return F.interpolate(inputs, size=(h_re, w_re), mode='bilinear', align_corners=True)


def mask_generator(disp_grad, inpaint_mask, iteration=60, dilate_kernel_size=5):
    disp_grad = disp_grad * inpaint_mask
    for _ in range(iteration):
        disp_grad = disp_grad / (disp_grad.norm(dim=1, keepdim=True) + 1e-10)
        disp_grad = disp_grad * inpaint_mask + FunctionSoftsplat(disp_grad, disp_grad) * (1 - inpaint_mask)
        inpaint_mask = (disp_grad.abs().sum(dim=1, keepdims=True) > 0).float()
    inpaint_mask = dilate(inpaint_mask, kernel_size=dilate_kernel_size)

    return inpaint_mask


@jit(float32[:, :](float32[:, :], float32[:, :], float32, float32), nopython=True)
def generate_boundary_mask(boundary_mask, disp, K, df):
    result = np.zeros_like(boundary_mask)
    defocus = K * np.abs(disp - df)
    h, w = boundary_mask.shape
    for y in range(h):
        for x in range(w):
            if boundary_mask[y, x] > 0:
                # dilate_radius = max([defocus[y-1, x-1], defocus[y-1, x], defocus[y-1, x+1],
                #                      defocus[y, x-1], defocus[y, x], defocus[y, x+1],
                #                      defocus[y+1, x-1], defocus[y+1, x], defocus[y+1, x+1]]) + 1
                dilate_radius = 0
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if defocus[ny, nx] > dilate_radius:
                                dilate_radius = defocus[ny, nx]
                for dy in range(-int(dilate_radius), int(dilate_radius)+1):
                    for dx in range(-int(dilate_radius), int(dilate_radius)+1):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w and dy * dy + dx * dx <= dilate_radius * dilate_radius:
                            result[ny, nx] = 1
    return result
