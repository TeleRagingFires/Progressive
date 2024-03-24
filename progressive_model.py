## This is a Pytorch Version Code for A Progressive-learning-based Channel Prediction within Phase Noise of Independent Oscillators
## Haozhen Li


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import scipy.io as sio
from scipy import io
from einops import rearrange



##########################################################################
## Layer Norm
def involve_CtoB(x):
    return rearrange(x, 'b c h w -> (b c) h w')


def decompose_CfromB(x, c):
    return rearrange(x, '(b c) t h w -> b (t c) h w', c=c)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, ks):
        super(FeedForward, self).__init__()
        self.ffn_expansion_factor = 2
        self.pw = ks//2
        self.groups = 2
        hidden_features = int(dim * self.ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconvI = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=ks, stride=1, padding=self.pw,
                                groups=hidden_features, bias=bias)
        self.dwconvIII = nn.Sequential(nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=ks, stride=1, padding=self.pw,
                                groups=hidden_features, bias=bias),
                                        nn.Conv2d(hidden_features, hidden_features, kernel_size=ks, stride=1, padding=self.pw,
                                                  groups=hidden_features, bias=bias))

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias, groups=self.groups)

    def forward(self, x):
        x = self.project_in(x)
        x1_a, x1_b = self.dwconvI(x).chunk(2, dim=1)
        x2_a, x2_b = self.dwconvIII(x).chunk(2, dim=1)
        x1 = torch.cat([x1_a, x2_a], 1)
        x2 = torch.cat([x1_b, x2_b], 1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Gated-Dconv Feed-Forward Network Denosie (GDFN-D)
class FeedForward_denoise(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_denoise, self).__init__()
        self.ffn_expansion_factor = 2
        self.groups = 2
        hidden_features = int(dim * self.ffn_expansion_factor)

        self.project_in_pn = nn.Conv2d(2, hidden_features, kernel_size=1, bias=bias)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconvI = nn.Conv2d(hidden_features * (2+1), hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.dwconvIII = nn.Sequential(nn.Conv2d(hidden_features * (2+1), hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias),
                                        nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                                  groups=hidden_features, bias=bias))

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias, groups=self.groups)

    def forward(self, x, pneffectiveness):
        x = self.project_in(x)
        pn_x = self.project_in_pn(pneffectiveness)
        x = torch.cat([pn_x, x], 1)
        x1_a, x1_b = self.dwconvI(x).chunk(2, dim=1)
        x2_a, x2_b = self.dwconvIII(x).chunk(2, dim=1)
        x1 = torch.cat([x1_a, x2_a], 1)
        x2 = torch.cat([x1_b, x2_b], 1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv_dwconv_I = nn.Sequential(nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
                                          nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
        self.qkv_dwconv_II = nn.Conv2d(dim * 3, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_I, k_I, v_I = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q_II, k_II, v_II = self.qkv_dwconv_I(self.qkv(x)).chunk(3, dim=1)
        q_III, k_III, v_III = self.qkv_dwconv_II(self.qkv(x)).chunk(3, dim=1)

        q = torch.cat([q_I, q_II, q_III], 1)
        k = torch.cat([k_I, k_II, k_III], 1)
        v = torch.cat([v_I, v_II, v_III], 1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, ks):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, ks)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
### Denoiser Presentation
class TransformerBlock_denoise(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_denoise, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_denoise(dim, ffn_expansion_factor, bias)

    def forward(self, x, PNeffectiveness):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x), PNeffectiveness)

        return x


##########################################################################
class Inception_L(nn.Module):
    def __init__(self, feature_in, feature_out, ks):
        super(Inception_L, self).__init__()
        self.pw = ks // 2
        self.groups = 2
        self.branch1 = nn.Conv2d(feature_in, feature_out, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(nn.Conv2d(feature_in, feature_out, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(feature_out, feature_out, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Conv2d(feature_in, feature_out, kernel_size=3, stride=1, padding=1)
        self.end = nn.Conv2d(feature_out*3, feature_out, kernel_size=ks, stride=1, padding=self.pw, groups=self.groups)

    def forward(self, x):
        x = self.end(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))
        return x


##########################################################################
### Time Envolver
### To evolve the His_S_1 to Future_1
class Time_Envolver(nn.Module):
    def __init__(self, ks):
        super(Time_Envolver, self).__init__()
        self.time_Intervel = 4      # Input Time Correlation
        self.short_Intervel = 2     # Amount of Correlation
        self.inception = Inception_L(feature_in=self.time_Intervel*2, feature_out=self.short_Intervel, ks=ks)  # To Sum Long-Corr

    def forward(self, x):
        _B, _C, H, W = x.shape  # Here _B involves [B*T]*C'*H'*W'
        latent_t1, latent_t2, latent_t3, latent_t4 = torch.chunk(x, chunks=self.time_Intervel, dim=0)   # B*C'*H'*W'
        latent_t1 = involve_CtoB(latent_t1).unsqueeze(1)
        latent_t2 = involve_CtoB(latent_t2).unsqueeze(1)
        latent_t3 = involve_CtoB(latent_t3).unsqueeze(1)
        latent_t4 = involve_CtoB(latent_t4).unsqueeze(1)
        time_evlvo1 = torch.cat([latent_t1, latent_t3], 1)
        time_evlvo2 = torch.cat([latent_t2, latent_t4], 1)

        latent_l = torch.cat([time_evlvo1, time_evlvo2], 1)    # Finding [B*C'*T_sc(2)*T_lc(2)]*H'*W'
        latent_s = torch.cat([latent_t1, latent_t2, latent_t3, latent_t4], 1)
        latent = torch.cat([latent_l, latent_s], 1)
        latent = self.inception(latent)
        latent = decompose_CfromB(latent, _C)
        # latent_t1, latent_t2 = torch.chunk(latent, chunks=self.short_Intervel, dim=0)
        # latent = torch.cat([decompose_CfromB(latent_t1, _C), decompose_CfromB(latent_t2, _C)], dim=1)
        return latent


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Angular Delay Domain Representation
class FourierTransformModule(nn.Module):
    def forward(self, x):
        x_ifft = torch.fft.ifft(torch.fft.fft(x, dim=(-2)), dim=(-1))

        PN_effectiveIV = torch.cat([x_ifft[:, :, 32:41, 4:6], x_ifft[:, :, 32:41, 27:29]], 3)
        PN_effectiveIII = torch.cat([x_ifft[:, :, 27:45, 2:6], x_ifft[:, :, 27:45, 27:31]], 3)
        PN_effectiveII = torch.cat([x_ifft[:, :, 18:54, 2:10], x_ifft[:, :, 18:54, 23:31]], 3)
        return x_ifft, PN_effectiveIV, PN_effectiveIII, PN_effectiveII


# class SPADE_Type2(nn.Module):
#     # How to make the PN effective AD representation to be Feature Level Indication
#     def __init__(self, norm_nc, label_nc, ks, device):
#         super().__init__()
#         pw = ks // 2
#         self.group = 4
#         self.device = device
#         self.mlpC_shared = ComplexConv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
#         self.mlpC_shared_II = ComplexConv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw)
#         self.mlpC_gamma = ComplexConv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)
#         self.mlpC_beta = ComplexConv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)
#
#     def forward(self, PNeffective_feature, inp_feature):
#         _B, _C, H, W = inp_feature.shape
#         PNeffective_feature = complex_interpolate(PNeffective_feature, scale_factor=2)
#         PNeffective_common_I = self.mlpC_shared(PNeffective_feature)
#         PNeffective_common = complex_interpolate(PNeffective_common_I, scale_factor=2)
#         PNeffective_common_II = self.mlpC_shared_II(PNeffective_common)
#
#         Gamma_C = self.mlpC_gamma(PNeffective_common_II)
#         Beta_C = self.mlpC_beta(PNeffective_common_II)
#
#         # # apply scale and bias
#         PN_C = inp_feature * (1 + Gamma_C) + Beta_C
#         return PN_C


# class SPADE_Type1(nn.Module):
#     # How to make the PN effective AD representation to be Feature Level Indication
#     def __init__(self, norm_nc, label_nc, ks, device):
#         super().__init__()
#         pw = ks // 2
#         self.group = 4
#         self.device = device
#         # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
#         self.param_free_normC = NaiveComplexInstanceNorm2d(norm_nc, affine=False)
#         self.param_free_normC_I = NaiveComplexInstanceNorm2d(norm_nc, affine=False)
#         self.mlpC_shared_I = ComplexConv2d(label_nc, norm_nc//2, kernel_size=ks, padding=pw)
#         self.mlpC_shared_II = nn.Sequential(ComplexConv2d(label_nc, norm_nc//2, kernel_size=ks, padding=pw),
#                                          ComplexConv2d(norm_nc//2, norm_nc//2, kernel_size=ks, padding=pw))
#         self.mlpC_gamma = ComplexConv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)
#         self.mlpC_beta = ComplexConv2d(norm_nc, label_nc, kernel_size=ks, padding=pw)
#
#     def forward(self, PNeffective_feature, inp_feature):
#         _B, _C, H, W = inp_feature.shape
#         inp_features = torch.chunk(inp_feature, chunks=self.group, dim=1)
#         PNeffective_feature = complex_interpolate(PNeffective_feature, scale_factor=2)
#         PNeffective_features = torch.chunk(PNeffective_feature, chunks=self.group, dim=1)
#         PN_C = torch.empty(_B, 0, H, W, dtype=torch.complex64, device=self.device)
#         for i in range(self.group):
#             # processing_inp = torch.cat([torch.real(inp_features[i]), torch.imag(inp_features[i])], 1)
#             processing_inp = inp_features[i]
#             # processing = torch.cat([torch.real(PNeffective_features[i]), torch.imag(PNeffective_features[i])], 1)
#             processing = PNeffective_features[i]
#
#             processing_shared = torch.cat([self.mlpC_shared_I(processing), self.mlpC_shared_II(processing)],1)
#             processing_shared = self.param_free_normC(processing_shared)
#             processing_shared = complex_softplus(processing_shared)
#             processing_shared = complex_interpolate(processing_shared, scale_factor=2)
#
#             processing_gamma = self.mlpC_gamma(processing_shared)
#             processing_gamma = self.param_free_normC_I(processing_gamma)
#             processing_gamma = complex_softplus(processing_gamma)
#
#             processing_beta = self.mlpC_beta(processing_shared)
#             processing_beta = self.param_free_normC_I(processing_beta)
#             processing_beta = complex_softplus(processing_beta)
#
#             processing_out = processing_inp * (1 + processing_gamma) + processing_beta
#             PN_C = torch.cat((PN_C, processing_out), dim=1)
#         return PN_C


########################################################################################
##  Angular Sparde
class Angular_Select(nn.Module):
    # Finding the Angular representation that is unreasonable
    def __init__(self, Angular_representive_Threshold, device):
        super().__init__()
        self.threshold = Angular_representive_Threshold
        self.device = device
        self.groups = 2

    def forward(self, Inp_AD_C):
        _B, _C, H, W = Inp_AD_C.shape
        Inp_AD_LR, Inp_AD_SR = torch.chunk(Inp_AD_C, chunks=self.groups, dim=1)
        Inp_AD_Cs = torch.cat([Inp_AD_LR.unsqueeze(1), Inp_AD_SR.unsqueeze(1)], dim=1)
        Inp_Cs = torch.empty(_B, 0, H, W, dtype=torch.complex64, device=self.device)
        for group in range(self.groups):
            Inp_AD_C = Inp_AD_Cs[:, group, :, :, :]
            His1_AD_REALenergy = torch.sqrt(torch.abs(torch.real(Inp_AD_C[:, 0, :, :])) ** 2).unsqueeze(1)
            His1_AD_IMAGenergy = torch.sqrt(torch.abs(torch.imag(Inp_AD_C[:, 0, :, :])) ** 2).unsqueeze(1)
            His2_AD_REALenergy = torch.sqrt(torch.abs(torch.real(Inp_AD_C[:, 1, :, :])) ** 2).unsqueeze(1)
            His2_AD_IMAGenergy = torch.sqrt(torch.abs(torch.imag(Inp_AD_C[:, 1, :, :])) ** 2).unsqueeze(1)

            # His1R2I_CS = nn.functional.cosine_similarity(His1_AD_REALenergy, His2_AD_IMAGenergy, dim=2)
            # His2R1I_CS = nn.functional.cosine_similarity(His2_AD_REALenergy, His1_AD_IMAGenergy, dim=2)
            # CS_differ = His1R2I_CS+His2R1I_CS
            # print(CS_differ.shape)

            His1R2I_differ = torch.sum(torch.abs(His1_AD_REALenergy - His2_AD_IMAGenergy), dim=2)
            His2R1I_differ = torch.sum(torch.abs(His2_AD_REALenergy - His1_AD_IMAGenergy), dim=2)
            Energy_differ = His1R2I_differ+His2R1I_differ

            _, Energy_Indices = torch.sort(Energy_differ)
            angular_index = Energy_Indices[:, :, 0:self.threshold]
            MASK = torch.zeros(_B, 1, H, W, dtype=torch.float32, device=self.device)
            for i in range(_B):
                MASK_p = MASK[i, :, :, :]
                angular_index_p = angular_index[i, :, :]
                for a in range(self.threshold):
                    MASK_p[:, :, angular_index_p[:, a]] = 1
                    MASK[i, :, :, :] = MASK_p.unsqueeze(0)

            Inp_AD_C = Inp_AD_C*MASK
            Inp_C = torch.fft.fft(torch.fft.ifft(Inp_AD_C, dim=(-2)), dim=(-1))
            # print(Inp_C.shape)
            # Inp_C = torch.fft.fftn(Inp_AD_C, dim=(-2, -1))
            Inp_C = Inp_C+0.5
            Inp_Cs = torch.cat([Inp_Cs, Inp_C], dim=1)
        return Inp_Cs


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Downsample_EX(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_EX, self).__init__()

        self.downlayer = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.downlayer(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.branch_1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch_2 = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.Shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x = self.Shuffle(torch.cat([x1, x2], 1))
        return x


class Upsample_EX(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_EX, self).__init__()

        self.uplayer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer1 = nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False, groups=2)

    def forward(self, x):
        x = self.layer1(self.uplayer(x))
        return x


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
                 device,
                 inp_channels=2,  # Sample_wise
                 out_channels=2,
                 dim=6,
                 num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=1,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()
        self.input_Intervel = 2   # number of correlation
        self.fourier_module = FourierTransformModule()
        self.angular_selector = Angular_Select(Angular_representive_Threshold=15, device=device)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, ks=3) for i in range(num_blocks[0])])
        self.time_evlove_1 = Time_Envolver(ks=9)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock_denoise(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.time_evlove_2 = Time_Envolver(ks=7)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock_denoise(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.time_evlove_3 = Time_Envolver(ks=5)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = TransformerBlock_denoise(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.time_evlove_4 = Time_Envolver(ks=3)

        self.up4_3 = Upsample(int(dim * 2 ** 4))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 4), int(dim * 2 ** 3), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, ks=3) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 3))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, ks=3) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, ks=3) for i in range(num_blocks[0])])

        self.up1_0 = Upsample_EX(int(dim * 2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level0 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, ks=7) for i in range(num_blocks[0])])
        self.down1_0 = Downsample_EX(int(dim * 2))  ## From Level 1 to Level 0

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 3), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, ks=5) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################
        self.output = nn.Conv2d(int(dim * 3), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output = nn.Conv2d(int(dim * 1.5), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.pid_output = nn.Conv2d(int(dim * 1.5), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.activation = nn.Sigmoid()

    def forward(self, inp_img):

        B, T, C, H, W = inp_img.shape
        inp_c = torch.complex(inp_img[:, :, 0, :, :]-0.5, inp_img[:, :, 1, :, :]-0.5)
        x_AD_C, PN_effectiveIV, PN_effectiveIII, PN_effectiveII = self.fourier_module(inp_c)

        X = self.angular_selector(x_AD_C)
        inp_img = torch.cat([torch.real(X).unsqueeze(1), torch.imag(X).unsqueeze(1)], 1)

        PN_effectiveII = torch.cat([torch.real(PN_effectiveII).unsqueeze(1), torch.imag(PN_effectiveII).unsqueeze(1)], 1)
        PN_effectiveIII = torch.cat([torch.real(PN_effectiveIII).unsqueeze(1), torch.imag(PN_effectiveIII).unsqueeze(1)], 1)
        PN_effectiveIV = torch.cat([torch.real(PN_effectiveIV).unsqueeze(1), torch.imag(PN_effectiveIV).unsqueeze(1)], 1)
        PN_effectiveII = torch.cat([PN_effectiveII[:,:,0,:,:]+0.5, PN_effectiveII[:,:,1,:,:]+0.5, PN_effectiveII[:,:,2,:,:]+0.5, PN_effectiveII[:,:,3,:,:]+0.5], dim=0)
        PN_effectiveIII = torch.cat([PN_effectiveIII[:,:,0,:,:]+0.5, PN_effectiveIII[:,:,1,:,:]+0.5, PN_effectiveIII[:,:,2,:,:]+0.5, PN_effectiveIII[:,:,3,:,:]+0.5], dim=0)
        PN_effectiveIV = torch.cat([PN_effectiveIV[:,:,0,:,:]+0.5, PN_effectiveIV[:,:,1,:,:]+0.5, PN_effectiveIV[:,:,2,:,:]+0.5, PN_effectiveIV[:,:,3,:,:]+0.5], dim=0)

        inp_least = inp_img[:, :, 3, :, :]
        inp_img = torch.cat([inp_img[:, :, 0, :, :], inp_img[:, :, 1, :, :], inp_img[:, :, 2, :, :], inp_img[:, :, 3, :, :]], dim=0)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        level1_shortcut_pid, level1_shortcut_den = torch.chunk(self.time_evlove_1(out_enc_level1), chunks=2, dim=1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, PN_effectiveII)
        level2_shortcut_pid, level2_shortcut_den = torch.chunk(self.time_evlove_2(out_enc_level2), chunks=2, dim=1)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, PN_effectiveIII)
        level3_shortcut_pid, level3_shortcut_den = torch.chunk(self.time_evlove_3(out_enc_level3), chunks=2, dim=1)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, PN_effectiveIV)
        latent = self.time_evlove_4(latent)

        latent_pid, latent_den = torch.chunk(self.up4_3(latent), chunks=2, dim=1)
        latent = torch.cat([latent_pid, level3_shortcut_pid, latent_den, level3_shortcut_den], 1)
        latent = self.reduce_chan_level3(latent)
        latent = self.decoder_level3(latent)

        latent_pid, latent_den = torch.chunk(self.up3_2(latent), chunks=2, dim=1)
        latent = torch.cat([latent_pid, level2_shortcut_pid, latent_den, level2_shortcut_den], 1)
        latent = self.reduce_chan_level2(latent)
        latent = self.decoder_level2(latent)

        latent_pid, latent_den = torch.chunk(self.up2_1(latent), chunks=2, dim=1)
        latent = torch.cat([latent_pid, level1_shortcut_pid, latent_den, level1_shortcut_den], 1)
        latent = self.reduce_chan_level1(latent)
        latent_pid, latent_den = torch.chunk(self.decoder_level1(latent), chunks=2, dim=1)

        latent_e = self.up1_0(latent)
        latent_e = self.decoder_level0(latent_e)
        latent_e_pid, latent_e_den = torch.chunk(self.down1_0(latent_e), chunks=2, dim=1)

        latent_e = torch.cat([latent_pid, latent_e_pid, latent_den, latent_e_den], 1)
        out_dec_level1 = self.refinement(latent_e)
        # out_dec_level1_pid, out_dec_level1_den = torch.chunk(out_dec_level1, chunks=2, dim=1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            # out_dec_level1 = self.output(out_dec_level1) + inp_least
            out_dec_level1 = self.activation(self.output(out_dec_level1))
            # out_dec_level1_pid, out_dec_level1_den = torch.chunk(self.output(out_dec_level1), chunks=2, dim=1)
            # out_dec_level1_den = self.activation(out_dec_level1_den)
            # out_dec_level1_pid = self.activation(out_dec_level1_pid)

        # out_dec_level1_1, out_dec_level1_2 = torch.chunk(out_dec_level1, chunks=2, dim=1)
        # out_dec_level1 = torch.cat([out_dec_level1_den.unsqueeze(1), out_dec_level1_pid.unsqueeze(1)], dim=1)
        out_dec_level1 = out_dec_level1.unsqueeze(1)
        return out_dec_level1

