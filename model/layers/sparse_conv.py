import torch
from torch.nn.parameter import Parameter
from .utils import pad_2d


class SparseConv(torch.nn.Module):
    """
    Arguments
        tensor: Tensor input.
        binary_mask: Tensor, a mask with the same size as tensor,
                     channel size = 1
        out_channels: Integer, the dimensionality of the output space (i.e.
                 the number of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
                 specifying the strides of the convolution along
                 the height and width.
        l2_scale: float, A scalar multiplier Tensor.
                  0.0 disables the regularizer.

    Returns:
        Output tensor, binary mask.
    """
    def __init__(self, input_channels, out_channels=32, kernel_size=3,
                 strides=2, paddings='SAME', l2_scale=0.0):
        self.conv2d = torch.nn.Conv2d(input_channels, out_channels,
                                      kernel_size, strides)

        self.conv2d_ones = torch.nn.Conv2d(
            input_channels, out_channels,
            kernel_size, strides)
        with torch.no_grad:
            self.conv2d_ones.W = torch.ones_like(self.conv2d_ones.W)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.custom_padding = lambda inp: pad_2d(
            inp, paddings, 'conv2d',
            kernel_size[0],
            kernel_size[1], strides[0], strides[1])

        self.bias = Parameter(torch.Tensor(out_channels))
        for param in self.conv2d_ones.parameters():
            param.requires_grad = False
        self.maxpool2d = torch.nn.MaxPool2d(out_channels,
                                            strides=strides
                                            )

    def forward(self, x, binary_mask=None):
        x = self.custom_padding(x)
        if binary_mask is None:
            # Assume that if any channel has no information, all channels has
            # no information
            binary_mask = torch.where(x[:, [0], :, :] == 0,
                                      torch.zeros_like(x[:, [0], :, :]),
                                      torch.ones_like(x[:, [0], :, :]))
            # mask should have the size of (B, 1, H, W)
        else:
            binary_mask = self.custom_padding(binary_mask)
        features = torch.mul(x, binary_mask)
        features = self.conv2d(x)
        norm = self.conv2d_ones(binary_mask)
        norm = torch.where(norm == 0, torch.zeros_like(norm),
                           torch.reciprocal(norm))
        features = torch.mul(features, norm) + self.bias
        mask = self.maxpool2d(binary_mask)
        return features, mask
