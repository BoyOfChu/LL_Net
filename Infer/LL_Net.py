import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.normal import Normal


class Conv_1x1(nn.Module):
    """
    一个简单的1x1卷积
    """

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=1, padding=0, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv_3x3(nn.Module):
    """
    一个简单的3x3卷积
    """

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                              stride=1, padding=1, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv_Block(nn.Module):
    """
    一个简单的卷积
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out


class H_Conv(nn.Module):
    """
    一个简单的单向卷积
    """

    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        kernel_size = (large_kernel, small_kernel, small_kernel)
        padding = ((large_kernel - 1) // 2, (small_kernel - 1) // 2, (small_kernel - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        out = self.conv1(x)
        return out


class W_Conv(nn.Module):
    """
    一个简单的单向卷积
    """

    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        kernel_size = (small_kernel, large_kernel, small_kernel)
        padding = ((small_kernel - 1) // 2, (large_kernel - 1) // 2, (small_kernel - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        out = self.conv1(x)
        return out

class D_Conv(nn.Module):
    """
    一个简单的单向卷积
    """

    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        kernel_size = (small_kernel, small_kernel, large_kernel)
        padding = ((small_kernel - 1) // 2, (small_kernel - 1) // 2, (large_kernel - 1) // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        out = self.conv1(x)
        return out

class Attention_Block(nn.Module):
    def __init__(self, in_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        self.local = Conv_Block(in_channels, in_channels, kernel_size=small_kernel, stride=1,
                                padding=(small_kernel - 1) // 2, groups=groups)

        self.AP = nn.AvgPool3d(kernel_size=2, stride=2)
        self.MP = nn.MaxPool3d(kernel_size=2, stride=2)
        self.global_3x3 = Conv_Block(in_channels, in_channels, kernel_size=3, stride=1,
                                padding=1, groups=groups)
        self.global_h = H_Conv(in_channels, in_channels, large_kernel, small_kernel, groups=groups)
        self.global_w = W_Conv(in_channels, in_channels, large_kernel, small_kernel, groups=groups)
        self.global_d = D_Conv(in_channels, in_channels, large_kernel, small_kernel, groups=groups)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')


        self.linear = Conv_1x1(in_channels, in_channels, groups=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        local = self.local(x)
        x_d = self.AP(x) + self.MP(x)
        global_f = self.global_h(x_d) + self.global_w(x_d) + self.global_d(x_d) + self.global_3x3(x_d)
        attention = self.act(self.linear(local + self.up(global_f)))
        return attention * x


class MSA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.linear_1 = Conv_1x1(in_channels, out_channels)
        self.act_1 = nn.PReLU()
        self.att = Attention_Block(out_channels, large_kernel, small_kernel, groups=groups)
        self.linear_2 = Conv_1x1(out_channels, out_channels)
        self.re_channels = Conv_1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear_1(out)
        out = self.act_1(out)
        out = self.att(out)
        out = self.linear_2(out)
        out += self.re_channels(x)
        return out


class SSA_Block(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.channel = Conv_1x1(in_channels, in_channels, groups=1)
        self.spatial = Conv_Block(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                                  padding=(kernel_size - 1) // 2, groups=in_channels)
        self.fusion = Conv_1x1(in_channels, in_channels, groups=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        channel = self.channel(x)
        spatial = self.spatial(x)
        out = self.act(self.fusion(channel * spatial))
        out = out * x
        return out


class FFN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.linear_1 = Conv_1x1(in_channels, out_channels)
        self.act = nn.PReLU()
        self.att = SSA_Block(out_channels, kernel_size)
        self.linear_2 = Conv_1x1(out_channels, out_channels)
        self.re_channels = Conv_1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear_1(out)
        out = self.act(out)
        out = self.att(out)
        out = self.linear_2(out)
        out += self.re_channels(x)
        return out


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        self.msa = MSA_Block(in_channels, out_channels, large_kernel, small_kernel, groups=groups)

    def forward(self, x):
        out = self.msa(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.ffn = FFN_Block(in_channels, out_channels, kernel_size)

    def forward(self, x):
        out = self.ffn(x)
        return out


class Net(nn.Module):
    def __init__(self, start_channels, large_kernel, small_kernel, in_channels=2, out_channels=3):
        super().__init__()
        encoder_channels = [i * start_channels for i in [1, 1, 2, 4, 8, 16]]
        decoder_channels = [i * start_channels for i in [8, 4, 2, 1, 1, 1]]

        self.start_conv = nn.Conv3d(in_channels=in_channels, out_channels=encoder_channels[0], kernel_size=large_kernel, stride=1,
                                    padding=(large_kernel-1)//2)
        self.start_act = nn.PReLU()
        self.AP = nn.AvgPool3d(kernel_size=2, stride=2)
        self.MP = nn.MaxPool3d(kernel_size=2, stride=2)

        '''编码——下采样路径'''
        self.encoder_stage_1 = Stage(encoder_channels[0], encoder_channels[1], large_kernel, small_kernel, groups=encoder_channels[1])
        self.encoder_stage_2 = Stage(encoder_channels[1], encoder_channels[2], large_kernel, small_kernel, groups=encoder_channels[2])
        self.encoder_stage_3 = Stage(encoder_channels[2], encoder_channels[3], large_kernel, small_kernel, groups=encoder_channels[3])
        self.encoder_stage_4 = Stage(encoder_channels[3], encoder_channels[4], large_kernel, small_kernel, groups=encoder_channels[4])

        '''最底层的卷积'''
        self.mid_stage = Stage(encoder_channels[4], encoder_channels[5], large_kernel, small_kernel, groups=encoder_channels[5])

        '''解码——上采样路径'''
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.decoder_stage_4 = Decoder(encoder_channels[5] + encoder_channels[4], decoder_channels[0],
                                       kernel_size=small_kernel)
        self.decoder_stage_3 = Decoder(decoder_channels[0] + encoder_channels[3], decoder_channels[1],
                                       kernel_size=small_kernel)
        self.decoder_stage_2 = Decoder(decoder_channels[1] + encoder_channels[2], decoder_channels[2],
                                       kernel_size=small_kernel)
        self.decoder_stage_1 = Decoder(decoder_channels[2] + encoder_channels[1], decoder_channels[3],
                                       kernel_size=small_kernel)

        '''最后的几层卷积'''
        # self.final_stage = nn.Sequential(
        #     Conv_Block(decoder_channels[3], decoder_channels[4], kernel_size=small_kernel, stride=1,
        #                padding=(small_kernel - 1) // 2, groups=1),
        #     Conv_Block(decoder_channels[4], decoder_channels[5], kernel_size=small_kernel, stride=1,
        #                padding=(small_kernel - 1) // 2, groups=1),
        #     nn.PReLU()
        # )

        '''生成形变场的卷积'''
        self.FlowConv = nn.Conv3d(decoder_channels[3], out_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.FlowConv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.FlowConv.weight.shape))
        self.FlowConv.bias = nn.Parameter(torch.zeros(self.FlowConv.bias.shape))
        self.final_act = nn.Softsign()

    def forward(self, m, f):
        x = torch.cat([m, f], dim=1)
        x = self.start_act(self.start_conv(x))

        encoder_feature = []
        '''编码路径'''
        # stage 1
        x = self.encoder_stage_1(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        # stage 2
        x = self.encoder_stage_2(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        # stage 3
        x = self.encoder_stage_3(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        # stage 4
        x = self.encoder_stage_4(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        '''中间路径'''
        x = self.mid_stage(x)

        '''解码路径'''
        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_4(x)

        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_3(x)

        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_2(x)

        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_1(x)

        '''最后的卷积层+形变场'''
        # x = self.final_stage(x)
        flow = self.final_act(self.FlowConv(x))
        return flow


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, mov_image, flow, mod='bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]

        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode=mod, align_corners=True)

        return warped


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)

            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border",
                                                          align_corners=True)
        return flow


def smoothloss(y_pred):
    d2, h2, w2 = y_pred.shape[-3:]
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device,
                            requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

if __name__ == "__main__":
    for i in [8]:
        model = Net(start_channels=i, large_kernel=5, small_kernel=3, in_channels=2, out_channels=3)
        m = torch.rand((1, 1, 160, 192, 224))
        f = torch.rand((1, 1, 160, 192, 224))
        y = model(m, f)

        total = sum([param.nelement() for param in model.parameters()]) / 1e6

        print('Number of parameter: % .6fM; C  = %d' % (total, i))

    # import time
    # model = Net(start_channels=4, large_kernel=5, small_kernel=3, in_channels=2, out_channels=3).cuda()
    # m = torch.rand((1, 1, 160, 192, 224)).cuda()
    # f = torch.rand((1, 1, 160, 192, 224)).cuda()
    # model.eval()
    # start = time.process_time_ns()
    # for i in range(10):
    #     y = model(m,f)
    # end = time.process_time_ns()
    # dif = (end - start) / 1e+7
    # print(dif)