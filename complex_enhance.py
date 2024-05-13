import torch
from torch.nn import Module, Parameter
from torch.nn import LayerNorm, GroupNorm, Upsample, InstanceNorm2d
from torch.nn.functional import gelu, softplus, softmax, normalize


class NaiveComplexLayerNorm(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(NaiveComplexLayerNorm, self).__init__()
        self.bn_r = LayerNorm(normalized_shape, eps, elementwise_affine)
        self.bn_i = LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self,input):
        return self.bn_r(input.real).type(torch.complex64) +1j*self.bn_i(input.imag).type(torch.complex64)


class NaiveComplexGroupNorm(Module):
    '''
    Naive approach to complex Group norm, perform group norm independently on real and imaginary part.
    '''
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super(NaiveComplexGroupNorm, self).__init__()
        self.GN_r = GroupNorm(num_groups, num_channels)
        self.GN_i = GroupNorm(num_groups, num_channels)

    def forward(self, input):
        return self.GN_r(input.real).type(torch.complex64) +1j*self.GN_i(input.imag).type(torch.complex64)


class NaiveComplexInstanceNorm2d(Module):
    '''
    Naive approach to complex Group norm, perform group norm independently on real and imaginary part.
    '''
    def __init__(self, num_channels, affine=False):
        super(NaiveComplexInstanceNorm2d, self).__init__()
        self.IN_r = InstanceNorm2d(num_channels, affine=affine)
        self.IN_i = InstanceNorm2d(num_channels, affine=affine)

    def forward(self, input):
        return self.IN_r(input.real).type(torch.complex64) +1j*self.IN_i(input.imag).type(torch.complex64)


def complex_gelu(input):
    return gelu(input.real).type(torch.complex64) + 1j*gelu(input.imag).type(torch.complex64)


def complex_softmax(input, dim):
    return softmax(input.real, dim=dim).type(torch.complex64) + 1j*softmax(input.imag, dim=dim).type(torch.complex64)


def complex_softplus(input):
    return softplus(input.real).type(torch.complex64) + 1j*softplus(input.imag).type(torch.complex64)


def complex_normalize(input, dim):
    return normalize(input.real, dim=dim).type(torch.complex64) + 1j*normalize(input.imag, dim=dim).type(torch.complex64)


class ComplexGeLU(Module):

    def forward(self, input):
        return complex_gelu(input)


class NaiveComplexUpsampling(Module):
    '''
    Naive approach to complex Group norm, perform group norm independently on real and imaginary part.
    '''
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super(NaiveComplexUpsampling, self).__init__()
        self.UP_r = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.UP_i = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        return self.UP_r(input.real).type(torch.complex64) +1j*self.UP_i(input.imag).type(torch.complex64)