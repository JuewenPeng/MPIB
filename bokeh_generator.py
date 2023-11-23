#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy
import re

kernel_Render_updateOutput = '''

    extern "C" __global__ void kernel_Render_updateOutput(
        const int n,
        const int samples,
        const float* xs,
        const float* ys,
        const float* K,
        const float* depth_focus,
        const float* images,
        const float* alphas,
        const float* coffs,
        float* bokeh
    )
    {
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int b = ( intIndex / SIZE_3(bokeh) / SIZE_2(bokeh) / 1 ) % SIZE_0(bokeh);
            // const int c = ( intIndex / SIZE_3(bokeh) / SIZE_2(bokeh)                 ) % SIZE_1(bokeh);
            const int y = ( intIndex / SIZE_3(bokeh)                                 ) % SIZE_2(bokeh);
            const int x = ( intIndex                                                 ) % SIZE_3(bokeh);

            float num = 0;
            float sumR = 0;
            float sumG = 0;
            float sumB = 0;

            float radius = K[b];
            float df = depth_focus[b];

            for (int i = 0; i < samples; ++i) {
                float u = xs[i] * radius;
                float v = ys[i] * radius;
                
                float alpha_prev = 1;
                
                for (int l = 0; l < SIZE_1(images); ++l) {
                    float A = coffs[OFFSET_3(coffs, b, l, 0)];
                    float B = coffs[OFFSET_3(coffs, b, l, 1)];
                    float C = coffs[OFFSET_3(coffs, b, l, 2)];
                    float t = (A * u + B * v + C) / (df - A * (df * x - u) - B * (df * y - v) + 1e-5);
                    
                    float x_intersect = t * (df * x - u) + u;
                    float y_intersect = t * (df * y - v) + v;
                    float z_intersect = t * df;
                    
                    float x_map = x_intersect / (z_intersect + 1e-5);
                    float y_map = y_intersect / (z_intersect + 1e-5);
                    
                    if (0 <= y_map && y_map < SIZE_2(bokeh) && 0 <= x_map && x_map < SIZE_3(bokeh)) {
                        int x1 = int(x_map);
                        int x2 = x1 + 1;
                        int y1 = int(y_map);
                        int y2 = y1 + 1;
                        
                        int x2_ = x2;
                        int y2_ = y2;
                        
                        if (x2 >= SIZE_3(bokeh)) {
                            x2_ = SIZE_3(bokeh) - 1;
                        }
                        if (y2 >= SIZE_2(bokeh)) {
                            y2_ = SIZE_2(bokeh) - 1;
                        }
                        
                        float alpha_curr = 1;
                        
                        // if (-0.4 <= x - x_map && x - x_map <= 0.4 && -0.4 <= y - y_map <= 0.4) {
                        //     alpha_curr = alphas[OFFSET_5(alphas, b, l, 0, y, x)];
                        //     if (alpha_curr < 0.01) {
                        //         continue;
                        //     }
                        //     sumR += images[OFFSET_5(images, b, l, 0, y, x)] * alpha_curr * alpha_prev;
                        //     sumG += images[OFFSET_5(images, b, l, 1, y, x)] * alpha_curr * alpha_prev;
                        //     sumB += images[OFFSET_5(images, b, l, 2, y, x)] * alpha_curr * alpha_prev;
                        // }
                        // else {
                        float f1 = (x2 - x_map) * alphas[OFFSET_5(alphas, b, l, 0, y1, x1)];
                        float f2 = (x_map - x1) * alphas[OFFSET_5(alphas, b, l, 0, y1, x2_)];
                        float f3 = (x2 - x_map) * alphas[OFFSET_5(alphas, b, l, 0, y2_, x1)];
                        float f4 = (x_map - x1) * alphas[OFFSET_5(alphas, b, l, 0, y2_, x2_)];
                        float f12 = f1 + f2;
                        float f34 = f3 + f4;
                        alpha_curr = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        if (alpha_curr < 0.01) {
                            continue;
                        }
                        float f;
                        f12 = f1 * images[OFFSET_5(images, b, l, 0, y1, x1)] + f2 * images[OFFSET_5(images, b, l, 0, y1, x2_)];
                        f34 = f3 * images[OFFSET_5(images, b, l, 0, y2_, x1)] + f4 * images[OFFSET_5(images, b, l, 0, y2_, x2_)];
                        f = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        sumR += f * alpha_prev;

                        f12 = f1 * images[OFFSET_5(images, b, l, 1, y1, x1)] + f2 * images[OFFSET_5(images, b, l, 1, y1, x2_)];
                        f34 = f3 * images[OFFSET_5(images, b, l, 1, y2_, x1)] + f4 * images[OFFSET_5(images, b, l, 1, y2_, x2_)];
                        f = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        sumG += f * alpha_prev;

                        f12 = f1 * images[OFFSET_5(images, b, l, 2, y1, x1)] + f2 * images[OFFSET_5(images, b, l, 2, y1, x2_)];
                        f34 = f3 * images[OFFSET_5(images, b, l, 2, y2_, x1)] + f4 * images[OFFSET_5(images, b, l, 2, y2_, x2_)];
                        f = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        sumB += f * alpha_prev;
                        // }
                        
                        num += alpha_curr * alpha_prev;

                        if (alpha_curr > 0.99) {
                            // num += 1;
                            break;
                        }
                        else {
                            alpha_prev *= (1 - alpha_curr); 
                        }
                    }
                }
            }
            bokeh[OFFSET_4(bokeh, b, 0, y, x)] = sumR / num;
            bokeh[OFFSET_4(bokeh, b, 1, y, x)] = sumG / num;
            bokeh[OFFSET_4(bokeh, b, 2, y, x)] = sumB / num;
        }
    }

'''


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-5])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-5])(\()([^\)]+)(\))', strKernel)

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
        objMatch = re.search('(VALUE_)([0-5])(\()([^\)]+)(\))', strKernel)

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


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    def forward(self, images, alphas, coffs, K, depth_focus, samples_per_side):
        # self.save_for_backward(image, defocus)

        # defocus_dilate = -10000 * torch.ones_like(defocus).int()
        bokeh = torch.zeros_like(images[:, 0])
        if isinstance(depth_focus, (int, float)):
            depth_focus = torch.tensor(depth_focus, device=bokeh.device).repeat(images.shape[0]).float()
        elif isinstance(depth_focus, torch.Tensor):
            depth_focus = depth_focus[:, 0, 0, 0].float()
        if isinstance(K, (int, float)):
            K = torch.tensor(K, device=bokeh.device).repeat(images.shape[0]).float()
        elif isinstance(K, torch.Tensor):
            K = K[:, 0, 0, 0].float()

        size = samples_per_side

        xs = []
        ys = []
        for i in range(size):
            for j in range(size):
                x = i * 2/(size-1) - 1
                y = j * 2/(size-1) - 1
                if x**2 + y**2 <= 1:
                    xs.append(x)
                    ys.append(y)
        samples = len(xs)
        xs = torch.tensor(xs, device=bokeh.device)
        ys = torch.tensor(ys, device=bokeh.device)

        if images.is_cuda == True:
            # n = bokeh_cum.nelement()
            n = bokeh.nelement() // 3
            cupy_launch('kernel_Render_updateOutput', cupy_kernel('kernel_Render_updateOutput', {
                'samples': samples,
                'xs': xs,
                'ys': ys,
                'K': K,
                'depth_focus': depth_focus,
                'images': images,
                'alphas': alphas,
                'coffs': coffs,
                'bokeh': bokeh
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=(
                    cupy.int(n),
                    cupy.int(samples),
                    xs.data_ptr(),
                    ys.data_ptr(),
                    K.data_ptr(),
                    depth_focus.data_ptr(),
                    images.data_ptr(),
                    alphas.data_ptr(),
                    coffs.data_ptr(),
                    bokeh.data_ptr()
                )
            )

        elif bokeh.is_cuda == False:
            raise NotImplementedError()

        # end

        return bokeh
    # end


# end


def FunctionRender(images, alphas, coffs, K, depth_focus, samples_per_side):
    bokeh = _FunctionRender.apply(images, alphas, coffs, K, depth_focus, samples_per_side)

    return bokeh


# end


class ModuleRenderRT(torch.nn.Module):
    def __init__(self):
        super(ModuleRenderRT, self).__init__()
        # self.gaussian_blur = GaussianBlur(gaussian_kernel)

    # end

    def forward(self, images, alphas, coffs, K, depth_focus, samples_per_side=51):
        bokeh = FunctionRender(images, alphas, coffs, K, depth_focus, samples_per_side)
        return bokeh
    # end
# end


if __name__ == '__main__':
    import cv2
    
    module = ModuleRenderRT().cuda()
    # In practice, make sure that the lower index corresponds to the nearer object in the second dimension
    # of the following three tensors, and do not let the disparities of different objects overlap with each other
    # by cautiously setting tensor "coffs"
    images = torch.zeros((1, 3, 3, 256, 256)).cuda()  # batch, num_object, C, H, W
    alphas = torch.zeros((1, 3, 1, 256, 256)).cuda()  # batch, num_object, C, H, W
    coffs = torch.zeros((1, 3, 3)).cuda()  # batch, num_object, num_param (a, b, c) (refer to Eq.7 of the paper)

    images[0, 0] = 1
    images[0, 1] = 0.5
    images[0, 2] = 0

    alphas[0, 0, 0, 50:100, 50:100] = 1
    alphas[0, 1, 0, 75:200, 75:200] = 1
    alphas[0, 2] = 1    
    
    coffs[0, 0, 0], coffs[0, 0, 1], coffs[0, 0, 2] = -1e-4/0.9, -4e-4/0.9, 1/0.9
    coffs[0, 1, 0], coffs[0, 1, 1], coffs[0, 1, 2] = 2e-4/0.5, -2e-4/0.5, 1/0.5
    coffs[0, 2, 0], coffs[0, 2, 1], coffs[0, 2, 2] = -2e-4/0.01, 1e-4/0.01, 1/0.01

    aif = torch.zeros((1, 3, 256, 256)).cuda()
    disp = torch.zeros((1, 1, 256, 256)).cuda()
    grid_y, grid_x = torch.meshgrid(torch.arange(256), torch.arange(256))
    grid_y = grid_y[None, None].cuda()
    grid_x = grid_x[None, None].cuda()
    for i in range(images.shape[1]-1, -1, -1):
        image_i = images[:, i]
        alpha_i = alphas[:, i]
        coff_i = coffs[:, i]
        disp_i = (1 - coff_i[:, 0] * grid_x - coff_i[:, 1] * grid_y) / coff_i[:, 2]
        aif = aif * (1 - alpha_i) + image_i * alpha_i
        disp = disp * (1 - alpha_i) + disp_i * alpha_i
    
    cv2.imwrite('aif.jpg', aif[0].detach().clone().permute(1, 2, 0).cpu().numpy() * 255)
    cv2.imwrite('disp.jpg', disp[0][0].detach().clone().cpu().numpy() * 255)
    
    K = 50      # blur radius
    zf = 1/0.5  # depth of focus
    samples_per_side = 101
    import time
    torch.cuda.synchronize()
    start = time.time()
    for i in range(10):
        bokeh = module(images, alphas, coffs, K, zf, samples_per_side)
    torch.cuda.synchronize()
    print(time.time() - start)

    cv2.imwrite('bokeh.jpg', bokeh[0].detach().clone().permute(1, 2, 0).cpu().numpy() * 255)
