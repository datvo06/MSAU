from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import math
import torch.nn.functional as F


class CustomAttentConv(torch.nn.Module):
    def __init__(self, in_channels, channels, kernel=4, stride=2, pad=0,
                 pad_type='zero', use_bias=True, sn=False):
        super(CustomAttentConv, self).__init__()
        paddings = [pad, pad]
        self.pad = torch.nn.ConstantPad2d(paddings, 0)\
            if pad_type == 'zero' else torch.nn.ReflectionPad2d(paddings)
        if sn:
            self.conv = torch.nn.Conv2d(in_channels, channels, kernel,
                                        stride, bias=use_bias)
            self.conv.weight.data.normal_(0.0, 0.02)
        else:
            self.conv = torch.nn.Conv2d(in_channels, channels, kernel,
                                        stride, bias=use_bias)

    def forward(self, x):
        return self.conv(self.pad(x))


def hw_flatten(x):
    batch_size = x.size()[0]
    return x.view(batch_size, x.size()[1], -1)


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e3):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, channels, d1 ... dn]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.size()
    num_dims = len(static_shape) - 2
    channels = static_shape[1]
    # Numer of time scales = number of channels / (2 * number of dimmension)
    num_timescales = channels // (num_dims * 2)
    # rescaling time step on each
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        torch.from_numpy(np.array(num_timescales)) - 1
    )
    # Inversion of time scales
    inv_timescales = min_timescale * torch.exp(
        torch.from_numpy(np.array(num_timescales)).float() * -log_timescale_increment.float())
    # Loop through each dimmensions, could be H, W,..
    for dim in range(num_dims):
        # The size of the dimmension
        length = x.size()[dim + 2]
        # Annotating index of each element along the dimmension's axes
        position = torch.from_numpy(np.arange(length))
        # position size is [dim_length]
        # using broad casting to get the scaled time along dimmension
        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
            inv_timescales, 0)
        # Scaled time: [dim_length, 1]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        # signal: [dim_length, 2]
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 2) * 2 * num_timescales
        signal = F.pad(signal, (prepad, postpad))
        signal = torch.transpose(signal, 0, 1)
        # signal: [channels, dim_lengths]
        for _ in range(1 + dim):
            signal = torch.unsqueeze(signal, 1)
        # signal: [channels, 1, 1, ..., dim_length]
        for _ in range(num_dims - 1 - dim):
            signal = torch.unsqueeze(signal, -2)
        # signal: [channels, 1, 1, ..., dim_length, 1, 1, ..., 1]
        x += signal
    return x


def add_positional_embedding_nd(x, max_length):
    """Add n-dimensional positional embedding.
    Adds embeddings to represent the positional dimensions of the tensor.
    The input tensor has n positional dimensions - i.e. 1 for text, 2 for images,
    3 for video, etc.
    Args:
      x: a Tensor with shape [batch, depth, p1 ... pn]
      max_length: an integer.  static maximum size of any dimension.
      name: a name for this layer.
    Returns:
      a Tensor the same shape as x.
    """
    '''
    static_shape = x.size()
    num_dims = len(static_shape) - 2
    depth = num_dims[1]
    base_shape = [1] + [depth] + [1] * (num_dims)
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [depth] + [1] * num_dims
    for i in range(num_dims):
        shape = base_shape[:]
        start = base_start[:]
        size = base_size[:]
        shape[i + 2] = max_length
        size[i + 2] = dynamic_shape[i+2]
        var = torch.nn.Parameter(shape)
        var.data.normal_(0, depth ** -0.5)
        x += (var * (depth ** 0.5)).narrow(0, 0, size[i+2])
    '''
    return x


class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, input_channels, num_heads=8, sn=False):
        super(SelfAttentionBlock, self).__init__()
        ch = input_channels
        self.f = CustomAttentConv(
            input_channels, ch//num_heads, kernel=1, stride=1,
            sn=sn)
        self.g = CustomAttentConv(
            input_channels, ch // num_heads, kernel=1, stride=1
        )
        self.h = CustomAttentConv(
            input_channels, ch, kernel=1, stride=1, sn=sn)
        self.softmax = torch.nn.Softmax(dim=-1)     # relation/attention map

    def forward(self, x):
        out_f = self.f(x)  # F: [batch, bottle_depth, H, W]
        out_g = self.g(x)  # G: [batch, bottle_depth, H, W]
        out_h = self.h(x)  # H: [batch, depth, H, W]
        s = torch.matmul(torch.transpose(hw_flatten(out_g), 1, 2),
                         hw_flatten(out_f))
        beta = self.softmax(s)
        o = torch.matmul(hw_flatten(out_h), beta)
        o = o.view(x.size())
        x = o + x
        return x



class SEBlock(torch.nn.Module):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, input_channels, ratio):
        self.squeeze_func = torch.nn.Linear(input_channels,
                                            input_channels//ratio)
        self.excitation_func = torch.nn.Linear(input_channels//ratio,
                                               input_channels)

    def forward(self, x):
        squeeze_statistic = x.mean(-1).mean(-1)
        excitation = self.squeeze_func(squeeze_statistic)
        excitation = self.excitation_func(excitation)
        scale = x * excitation
        return scale


class ChannelAttentionBlock(torch.nn.Module):

    def __init__(self, input_channels, ratio):
        self.squeeze1 = torch.nn.Linear(input_channels,
                                        input_channels//ratio)
        self.excite1 = torch.nn.Linear(input_channels//ratio,
                                       input_channels)
        self.squeeze2 = torch.nn.Linear(input_channels,
                                        input_channels//ratio)
        self.excite2 = torch.nn.Linear(input_channels//ratio,
                                       input_channels)
        self.activation = torch.nn.Sigmoid()



    def forward(self, x):
        avg_pool = x.mean(-1).mean(-1)
        squeeze_avg = self.squeeze1(avg_pool)
        excite_avg = self.excite1(squeeze_avg)
        max_pool = x.max(-1).max(-1)
        squeeze_max = self.squeeze2(max_pool)
        excite_max = self.max_timescale(squeeze_max)
        scale = self.activation(excite_avg + excite_max)
        return scale


class SpatialAttentionBlock(torch.nn.Module):
    def __init__(self, input_channels, kernel_size=7):
        self.conv = torch.nn.Conv2d(input_channels,
                                    1, kernel_size, 1, 3, bias=False)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        spatial_avg_pool = x.mean(1, keepdim=True)
        spatial_max_pool = x.max(1, keepdim=True)
        concat = torch.cat([spatial_avg_pool, spatial_max_pool], dim=1)
        return x * self.activation(self.conv(x))


class CBAMBlock(torch.nn.Module):
    """Contains the implementation of Convolutional Block
    Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, input_channels, ratio=8):
        self.channel_attent = ChannelAttentionBlock(input_channels, ratio)
        self.spatial_attent = SpatialAttentionBlock(input_channels)

    def forward(self, x):
        return self.spatial_attent(self.channel_attent(x))


class SAWrapperBlock(torch.nn.Module):
    def __init__(self, input_channels):
        super(SAWrapperBlock, self).__init__()
        self.attention_block = SelfAttentionBlock(input_channels, num_heads=8)

    def forward(self, x):
        # attention_feature = add_timing_signal_nd(x, 1e3)
        return self.attention_block(x)
