import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexConvTranspose2d, ComplexReLU
from complex_enhance import NaiveComplexGroupNorm
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_avg_pool2d, complex_upsample
from scipy.stats import spearmanr


def complex_to_polar(complex_RI):
    # Magnitude
    mag = torch.sqrt((torch.real(complex_RI))**2 + (torch.imag(complex_RI))**2)
    # Phase
    phase = torch.atan2(torch.real(complex_RI), torch.imag(complex_RI))

    return mag, phase


def polar_to_complex(magnitude, phase):
    # Real
    real_part = magnitude * torch.cos(phase)
    # Imaginary
    imag_part = magnitude * torch.sin(phase)
    complex_h = torch.complex(real_part, imag_part)

    return complex_h


class BasicCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False,
                 map_build=False):
        super(BasicCConv2d, self).__init__()
        self.act_norm = act_norm
        self.map_build = map_build
        if not transpose:
            self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = ComplexConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = ComplexBatchNorm2d(out_channels)
        # self.bn = ComplexBatchNorm2d(out_channels)
        # self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = complex_relu(self.norm(y))
        if self.map_build:
            y = complex_relu(y)
        return y


class GroupCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, transpose=False,
                 act_norm=False):
        super(GroupCConv2d, self).__init__()
        self.act_norm = act_norm
        self.transpose = transpose
        if in_channels % groups != 0:
            groups = 1

        if not transpose:
            self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
        else:
            self.conv = ComplexConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, groups=groups, output_padding=stride // 2)
        self.norm = NaiveComplexGroupNorm(4, out_channels)

    def forward(self, x):

        y = self.conv(x)
        if self.act_norm:
            y = complex_relu(self.norm(y))
        return y


class CConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True, groupConv=False):
        super(CConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.groups = 4
        if groupConv:
            self.Cconv = GroupCConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                    padding=1, transpose=transpose, groups=self.groups, act_norm=True)
        else:
            self.Cconv = BasicCConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                    padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.Cconv(x)
        return y


class CConvSC_K(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, transpose=False, act_norm=True, groupConv=False):
        super(CConvSC_K, self).__init__()
        if stride == 1:
            transpose = False
        self.groups = 4
        self.kernel = kernel_size
        self.padding = padding
        if groupConv:
            self.Cconv = GroupCConv2d(C_in, C_out, kernel_size=self.kernel, stride=stride,
                                    padding=self.padding, transpose=transpose, groups=self.groups, act_norm=True)
        else:
            self.Cconv = BasicCConv2d(C_in, C_out, kernel_size=self.kernel, stride=stride,
                                    padding=self.padding, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.Cconv(x)
        return y


# class CInception(nn.Module):
#     def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
#         super(CInception, self).__init__()
#         self.Cconv1 = ComplexConv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
#         layers = []
#         for ker in incep_ker:
#             layers.append(
#                 GroupCConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker // 2, groups=groups, act_norm=True))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.Cconv1(x)
#         y = 0
#         for layer in self.layers:
#             y += layer(x)
#         return y


class CInceptionA(nn.Module):
    def __init__(self, C_in, C_out):
        super(CInceptionA, self).__init__()
        self.groupConv = False
        self.branch1x1 = CConvSC_K(C_in, 64, kernel_size=1, stride=1, padding=0, transpose=False, act_norm=True, groupConv=self.groupConv)

        self.branch5x5_1 = CConvSC_K(C_in, 48, kernel_size=1, stride=1, padding=0, transpose=False, act_norm=True, groupConv=self.groupConv)
        self.branch5x5_2 = CConvSC_K(48, 64, kernel_size=5, stride=1, padding=2, transpose=False, act_norm=True, groupConv=self.groupConv)

        self.branch3x3db1_1 = CConvSC_K(C_in, 64, kernel_size=1, stride=1, padding=0, transpose=False, act_norm=True, groupConv=self.groupConv)
        self.branch3x3db1_2 = CConvSC_K(64, 96, kernel_size=3, stride=1, padding=1, transpose=False, act_norm=True, groupConv=self.groupConv)
        self.branch3x3db1_3 = CConvSC_K(96, 96, kernel_size=3, stride=1, padding=1, transpose=False, act_norm=True, groupConv=self.groupConv)

        self.branch_pool = CConvSC_K(C_in, C_out, kernel_size=1, stride=1, padding=0, transpose=False, act_norm=True, groupConv=self.groupConv)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db1_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = complex_avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3db1, branch_pool]
        return torch.cat(outputs, 1)


class PNFBlock(nn.Module):
    # How to isolate the PN out
    def __init__(self, device):
        super(PNFBlock, self).__init__()
        self.device = device
        self.group = 4

    def forward(self, hid):
        _B, _C, H, W = hid.shape
        hids = torch.chunk(hid, chunks=self.group, dim=1)
        pnExtactor = torch.empty(_B, self.group-1, _C//self.group, H, W, dtype=torch.complex64, device=self.device)
        for i in range(self.group-1):
            PNC_B = torch.div(hids[i], hids[i+1])
            PNC_B = PNC_B.squeeze()
            # if torch.isnan(PNC_B).any():
            #     print("There are nan values in the tensor.")
            #     break
            pnExtactor[:, i, :] = PNC_B

        hid_s = torch.empty(_B, self.group, _C // self.group, H, W, dtype=torch.complex64, device=self.device)
        for i in range(self.group):
            if i == 0:
                hid = hids[i]
            elif i == 1:
                hid = torch.mul(hids[i], pnExtactor[:, i-1, :])
            elif i == 2:
                hid = torch.mul(torch.mul(hids[i], pnExtactor[:, i-1, :]), pnExtactor[:, i-2, :])
            elif i == 3:
                hid = torch.mul(torch.mul(torch.mul(hids[i], pnExtactor[:, i-1, :]), pnExtactor[:, i-2, :]), pnExtactor[:, i-3, :])
            # if torch.isnan(hid).any():
            #     print("There are nan values in the tensor.")
            #     break
            hid_s[:, i, :] = hid

        # print(hid_s.shape)
        return hid_s


def spearman_rank_correlation_matrix(matrix1_t, matrix2_t):
    """
    计算两个高维数组之间的斯皮尔曼秩相关系数矩阵。

    参数：
    matrix1: 第一个高维数组，形状为 (m, n)
    matrix2: 第二个高维数组，形状为 (m, n)

    返回：
    correlation_matrix: 两个高维数组的斯皮尔曼秩相关系数矩阵，形状为 (n, n)
    """
    B, T, H, W = matrix1_t.shape
    matrix1 = matrix1_t.cpu().detach().numpy()
    matrix2 = matrix2_t.cpu().detach().numpy()
    s = matrix1_t.shape[2]  # 获取列数
    a = matrix1_t.shape[3]  # 获取行数

    # 初始化斯皮尔曼秩相关系数矩阵
    correlation_matrix_a = np.zeros((B, 1, a, a))
    correlation_matrix_s = np.zeros((B, 1, s, s))

    # 逐列计算斯皮尔曼秩相关系数
    for index in range(B):
        processing_matrix_1 = matrix1[index, 0, :, :]
        processing_matrix_2 = matrix2[index, 0, :, :]
        for i in range(a):
            for j in range(a):
                correlation_matrix_a[index, :, i, j], _ = spearmanr(processing_matrix_1[:, i], processing_matrix_2[:, j])

    for index in range(B):
        processing_matrix_1 = matrix1[index, 0, :, :]
        processing_matrix_1 = np.transpose(processing_matrix_1)
        processing_matrix_2 = matrix2[index, 0, :, :]
        processing_matrix_2 = np.transpose(processing_matrix_2)
        for i in range(s):
            for j in range(s):
                correlation_matrix_s[index, :, i, j], _ = spearmanr(processing_matrix_1[:, i], processing_matrix_2[:, j])

    return correlation_matrix_a, correlation_matrix_s


class PNlevel_Estimate(nn.Module):
    # Finding out the Magnitude of Channel
    # And Finding the Magnitude-to-Phase Feature （Too Implicit）
    def __init__(self, device):
        super(PNlevel_Estimate, self).__init__()
        self.device = device
        self.group = 4
        # self.feture_embed = CConvSC_K(self.group, self.group, kernel_size=3, stride=1, padding=1, transpose=False, groupConv=True)

    def forward(self, sample, input_rho, input_rho_T):
        B, T, H, W = sample.shape
        hids = torch.chunk(sample, chunks=self.group, dim=1)
        # Magnitude = torch.empty(B, self.group, H, W, device=self.device)
        # Phase = torch.empty(B, self.group, H, W, device=self.device)
        # PaM_correlation_a = torch.empty(B, self.group, W, W, dtype=torch.complex64, device=self.device)
        # PaM_correlation_s = torch.empty(B, self.group, H, H, dtype=torch.complex64, device=self.device)
        PaM_correlation = torch.empty(B, self.group, H, W, dtype=torch.complex64, device=self.device)

        for i in range(self.group):
            mag, phase = complex_to_polar(hids[i])
            MtoP_feature = torch.div(mag, hids[i])

            # MtoP_feature_a, MtoP_feature_s = spearman_rank_correlation_matrix(mag, phase)
            # MtoP_feature_a, MtoP_feature_s = torch.tensor(MtoP_feature_a), torch.tensor(MtoP_feature_s)
            # MtoP_feature = torch.matmul(MtoP_feature_a, MtoP_feature_s)
            # print(MtoP_feature.shape)
            # print(Check)
            # Mag_feature = hids[i]*torch.conj(hids[i])
            # MtoP_feature = torch.div(Mag_feature, hids[i])
            # MtoP_feature.squeeze()
            # Mag, PHPN = complex_to_polar(hids[i])
            # Mag, PHPN = Mag.squeeze(), PHPN.squeeze()
            # Magnitude[:, i, :, :] = Mag
            # Phase[:, i, :, :] = PHPN
            # PaM_correlation_a[:, i, :, :] = MtoP_feature_a.squeeze()
            # PaM_correlation_s[:, i, :, :] = MtoP_feature_s.squeeze()
            PaM_correlation[:, i, :, :] = MtoP_feature.squeeze()

        # print(PaM_correlation.shape)
        # print(Check)
        # PaM_correlation = self.feture_embed(PaM_correlation)
        # return PaM_correlation_a, PaM_correlation_s
        return PaM_correlation


class PNFBlock_sample(nn.Module):
    # How to isolate the PN out
    def __init__(self, device):
        super(PNFBlock_sample, self).__init__()
        self.device = device
        self.group = 4

    def forward(self, sample):
        B, T, H, W = sample.shape
        hids = torch.chunk(sample, chunks=self.group, dim=1)
        pnExtactor = torch.empty(B, self.group-1, H, W, dtype=torch.complex64, device=self.device)
        for i in range(self.group-1):
            # PNC_B_head = torch.matmul(torch.transpose(torch.conj(hids[i + 1]), 2, 3), hids[i])
            PNC_B_head = hids[i]
            # PNC_B_sub = torch.abs(PNC_B_head)
            PNC_B_sub = hids[i+1]
            # PNC_B = torch.div(torch.real(PNC_B_head), PNC_B_sub) + 1j * (
            #     torch.div(torch.imag(PNC_B_head), PNC_B_sub))
            PNC_B = torch.div(PNC_B_head, PNC_B_sub)
            PNC_B = PNC_B.squeeze()
            if torch.isnan(PNC_B).any():
                print("There are values in the tensor.")
                break
            pnExtactor[:, i, :, :] = PNC_B

        hid_s = torch.empty(B, self.group, H, W, dtype=torch.complex64, device=self.device)
        for i in range(self.group):
            if i == 0:
                hid = hids[i]
            elif i == 1:
                # hid = torch.matmul(hids[i], pnExtactor[:, i - 1, :].unsqueeze(1))
                hid = torch.mul(hids[i], pnExtactor[:, i - 1, :].unsqueeze(1))
            elif i == 2:
                # hid = torch.matmul(torch.matmul(hids[i], pnExtactor[:, i - 1, :].unsqueeze(1)), pnExtactor[:, i - 2, :].unsqueeze(1))
                hid = torch.mul(torch.mul(hids[i], pnExtactor[:, i - 1, :].unsqueeze(1)), pnExtactor[:, i - 2, :].unsqueeze(1))
            elif i == 3:
                hid = torch.mul(torch.mul(torch.mul(hids[i], pnExtactor[:, i - 1, :].unsqueeze(1)), pnExtactor[:, i - 2, :].unsqueeze(1)),
                    pnExtactor[:, i - 3, :].unsqueeze(1))
            hid = hid.squeeze()
            if torch.isnan(hid).any():
                print("There are values in the tensor.")
                break
            hid_s[:, i, :] = hid

        return hid_s


# class CSPADE(nn.Module):
#     def __init__(self, label_nc, norm_nc, step):
#         super().__init__()
#         # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
#         if step == 1:
#             self.mlp_shared = nn.Sequential(BasicCConv2d(label_nc, norm_nc, kernel_size=3, stride=2,
#                              padding=1, transpose=True, act_norm=True),)
#         else:
#             self.mlp_shared = nn.Sequential(BasicCConv2d(label_nc, norm_nc, kernel_size=3, stride=1,
#                                                          padding=1, transpose=False, act_norm=True),)
#
#         self.mlp_gamma = BasicCConv2d(norm_nc, norm_nc, 1, stride=1,
#                                     padding=0, transpose=False, act_norm=True)
#         self.mlp_beta = BasicCConv2d(norm_nc, norm_nc, 1, stride=1,
#                                     padding=0, transpose=False, act_norm=True)
#
#     def forward(self, hid, segmap):
#         # Part 1. generate parameter-free normalized activations
#         # hid has been normalized in act_norm
#
#         # Part 2. produce scaling and bias conditioned on semantic map
#         # segmap has multiple_features
#         actv = self.mlp_shared(segmap)
#         # actv = complex_relu(actv)   # Introduce the Map for PN reproduce
#
#         gamma = self.mlp_gamma(actv)
#         beta = self.mlp_beta(actv)
#
#         # apply scale and bias
#         out = hid * (1 + gamma) + beta
#         return out, actv


class CSPADE(nn.Module):
    def __init__(self, label_nc, norm_nc, step):
        super().__init__()
        # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # self.Cconv = GroupCConv2d(C_in, C_out, kernel_size=3, stride=stride,
        #                           padding=1, transpose=transpose, groups=self.groups, act_norm=True)
        self.groups = 4
        if step == 1:
            self.mlp_shared = nn.Sequential(GroupCConv2d(label_nc, norm_nc, kernel_size=3, stride=2,
                             padding=1, transpose=True, groups=self.groups, act_norm=True),)
        else:
            self.mlp_shared = nn.Sequential(GroupCConv2d(label_nc, norm_nc, kernel_size=3, stride=1,
                                padding=1, transpose=False, groups=self.groups, act_norm=True),)

        self.mlp_gamma = GroupCConv2d(norm_nc, norm_nc, 1, stride=1,
                                    padding=0, transpose=False, groups=self.groups, act_norm=True)
        self.mlp_beta = GroupCConv2d(norm_nc, norm_nc, 1, stride=1,
                                    padding=0, transpose=False, groups=self.groups, act_norm=True)

    def forward(self, hid, segmap):
        # Part 1. generate parameter-free normalized activations
        # hid has been normalized in act_norm

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap has multiple_features
        actv = self.mlp_shared(segmap)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = hid * (1 + gamma) + beta
        return out, actv


# class PNFBlock_Over_one(nn.Module):
#     def __init__(self, c_mag, c_out, step):
#         super(PNFBlock_Over_one, self).__init__()
#         self.spade = CSPADE(c_mag, c_out, step)
#         self.step = step
#
#     def forward(self, Y_EF, Seg_Multiple):
#         Y_EF, actMap = self.spade(Y_EF, Seg_Multiple)
#         if self.step == 1:
#             actMap = complex_max_pool2d(actMap, kernel_size=(2, 2), stride=2)
#         return Y_EF, actMap


class PNFBlock_Over(nn.Module):
    def __init__(self, c_mag, c_out, step):
        super(PNFBlock_Over, self).__init__()
        self.spade = CSPADE(c_mag, c_out, step)
        self.step = step

    def forward(self, Y_EF, Seg_Multiple):
        if self.step == 4:
            Seg_Multiple = complex_upsample2(Seg_Multiple, scale_factor=2)
        Y_EF, actMap = self.spade(Y_EF, Seg_Multiple)
        if self.step == 1 or self.step == 3:
            actMap = complex_max_pool2d(actMap, kernel_size=(2, 2), stride=2)
        return Y_EF, actMap


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, ks):
        super().__init__()
        pw = ks // 2
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Conv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)
        self.mlp_gamma = nn.Conv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)

    def forward(self, Phase_Noise, Phase_feature):
        # Phase_Noise = self.mlp_shared(Phase_Noise)
        Phase_feature = self.param_free_norm(Phase_feature)
        gamma = self.mlp_gamma(Phase_feature)
        beta = self.mlp_beta(Phase_feature)
        # apply scale and bias
        out = Phase_Noise * (1 + gamma) + beta

        return out


class Phase_Mapping(nn.Module):
    def __init__(self, c_mag, c_out, device):
        super(Phase_Mapping, self).__init__()
        self.device = device
        self.group = 4
        self.input_dim = 9216
        self.hidden_dim = 4608
        self.output_dim = 9216
        self.ForwardLayer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            )
        # The connection here is mass
        # self.ForwardLayer =
        self.Phase_spade = SPADE(c_mag, c_mag, 7)
        self.upper_layer = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, Mag_PNH, Pha_PNH):
        B, T, H, W = Mag_PNH.shape

        Phase_Feature = torch.empty(B, self.group, H, W, device=self.device)
        for i in range(self.group):
            Mag_PNH_p = Mag_PNH[:, i, :].reshape(Mag_PNH.size(0), -1)
            Mag_PNH_p = self.ForwardLayer(Mag_PNH_p)
            Phase_Feature[:, i, :] = Mag_PNH_p.reshape(B, H, W)

        # Phase_Feature = self.upper_layer(Phase_Feature)
        Phase_Feature = self.Phase_spade(Pha_PNH, Phase_Feature)
        x = polar_to_complex(Mag_PNH, Phase_Feature)

        return x


# class ADD_PNPM(nn.Module):
#     def __init__(self, c_in, c_out, device):
#         super(ADD_PNPM, self).__init__()
#         self.seq_len = 4
#         self.device = device
#         self.groups = 3
#         self.conv_UP = GroupConv2d(c_in, c_out, kernel_size=1, stride=1,
#                                    padding=0, transpose=False, groups=self.groups, act_norm=True)
#         self.conv_Down = GroupConv2d(c_out, c_in, kernel_size=1, stride=1,
#                                      padding=0, transpose=False, groups=self.groups, act_norm=True)
#
#     def forward(self, x_raw):
#         B, T, C, H, W = x_raw.shape
#         #   Unroll over time Steps
#         AdditiveModel = torch.zeros(B, T - 1, C, H, W, device=self.device)
#         for time_step in range(self.seq_len - 1):
#             X_processing_p = x_raw[:, time_step, :]
#             X_processing_f = x_raw[:, time_step + 1, :]
#             AdditiveModel[:, time_step, :] = X_processing_f - X_processing_p
#
#         AdditiveModel = AdditiveModel.reshape(B, 3 * C, H, W)
#         AdditiveModel = self.conv_UP(AdditiveModel)
#         AdditiveModel = self.conv_Down(AdditiveModel)
#
#         return AdditiveModel
#
#
# class ADD_PNPM_Over(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(ADD_PNPM_Over, self).__init__()
#         self.seq_len = 4
#         self.groups = 3
#         self.conv_UP = GroupConv2d(c_in, c_out, kernel_size=1, stride=1,
#                                    padding=0, transpose=False, groups=self.groups, act_norm=True)
#         self.conv_Down = GroupConv2d(c_out, c_in, kernel_size=1, stride=1,
#                                      padding=0, transpose=False, groups=self.groups, act_norm=True)
#
#     def forward(self, Y_EF, Seg_additive):
#         Seg_additive = self.conv_UP(Seg_additive)
#         Seg_additive = self.conv_Down(Seg_additive)
#         #   Unroll over time Steps
#         # AdditiveModel = torch.zeros(B, T-1, C, H, W, device=self.device)
#         for time_step in range(self.seq_len - 1):
#             ADD_Correlation = Seg_additive[:, time_step: time_step + 2, :]
#             Y_EF[:, time_step + 1, :] = Y_EF[:, time_step, :] + ADD_Correlation
#
#         return Y_EF