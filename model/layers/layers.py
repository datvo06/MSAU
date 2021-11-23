from __future__ import print_function, division
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from .utils import pad_2d
from .sparse_conv import SparseConv


class Conv2dBnLrnDrop(torch.nn.Module):
    """
    Conv2d with optional BatchNorm, Local Response Norm and Dropout
    Note: is_training no longer needed, simply call model.eval()
    Args:
        kernel: [kernel_width, kernel_height, input_channels, out_channels
        strides: list of `ints`, length 2, the stride of the sliding window for
            each dimmension of `inputs`.
        activation: actvation function to be used
        use_bn: `bool`, wheter or not to use the batch normalization
        use_lrn: `bool`, whether or not to include local response normalization
                 in the layer.
        keep_prob: `double`, dropout keep prob.
        dropout_maps: `bool`, If true whole maps are dropped or not, otherwise
                      single elements.
        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm
                 used in the convolution.
    """
    def __init__(self, kernel_shape, use_sparse_conv=False,
                 strides=[1, 1], activation=torch.nn.ReLU,
                 use_bn=False, use_mvn=False, use_lrn=False, keep_prob=1.0,
                 dropout_maps=False, padding='SAME', initOpt=0, biasInit=0.1):
        super(Conv2dBnLrnDrop, self).__init__()
        if initOpt == 0:
            self.stddev = np.sqrt(2.0 / (
                kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
                + kernel_shape[3]))
        if initOpt == 1:
            self.stddev = 5e-2
        if initOpt == 2:
            self.stddev = min(np.sqrt(2.0 / (
                kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
        self.use_sparse_conv = use_sparse_conv
        self.custom_padding = lambda inp: pad_2d(inp,
                                                 padding,
                                                 'conv2d',
                                                 kernel_shape[0],
                                                 kernel_shape[1],
                                                 strides[0],
                                                 strides[1])
        if use_sparse_conv:
            self.custom_conv = SparseConv(kernel_shape[2],
                                          kernel_shape[3],
                                          kernel_shape[:2],
                                          strides=strides[0])
        else:
            self.custom_conv = torch.nn.Conv2d(
                kernel_shape[2], kernel_shape[3],
                kernel_shape[:2], stride=strides, padding=0)
            self.custom_conv.weight.data.normal_(0.0, self.stddev)
            self.custom_conv.bias.data.normal_(biasInit, 0.00001)

        self.use_bn = use_bn
        self.use_mvn = use_mvn
        self.use_lrn = use_lrn
        self.dropout_maps = dropout_maps
        if use_bn:
            self.bn = torch.nn.BatchNorm2d(kernel_shape[3])
        if use_mvn:
            # The smae as bn right now...
            self.mvn = torch.nn.BatchNorm2d(kernel_shape[3])
        self.activation = None
        if activation:
            self.activation = activation()
        if use_lrn:
            self.lrn = torch.nn.modules.normalization.LocalResponseNorm(kernel_shape[3])
        if dropout_maps:
            self.dropout = torch.nn.modules.Dropout2d(p=(1.0-keep_prob))
        else:
            self.dropout = torch.nn.modules.Dropout2d(p=(1.0-keep_prob))
        self.keep_prob = keep_prob

    def forward(self, x, binary_mask=None):
        x = self.custom_padding(x)
        if binary_mask is not None:
            binary_mask = self.custom_padding(binary_mask)
        if self.use_sparse_conv:
            outputs, binary_mask = self.custom_conv(x, binary_mask)
        else:
            outputs = self.custom_conv(x)
        if self.use_bn:
            outputs = self.bn(outputs)
        if self.use_mvn:
            outputs = self.mvn(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.use_lrn:
            outputs = self.lrn(outputs)
        outputs = self.dropout(outputs)
        if self.use_sparse_conv:
            return outputs, binary_mask
        else:
            return outputs


class DilConv2dBnLrnDrop(torch.nn.Module):
    def __init__(self, kernel_shape, rate,
                 activation=torch.nn.ReLU, use_bn=False, use_mvn=False,
                 use_lrn=True, padding='SAME', keep_prob=1.0, dropout_maps=False,
                 initOpt=0):
        super(DilConv2dBnLrnDrop, self).__init__()
        if initOpt == 0:
            self.stddev = np.sqrt(2.0 / (
                kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
                + kernel_shape[3]))
        if initOpt == 1:
            self.stddev = 5e-2
        if initOpt == 2:
            self.stddev = min(
                np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1]
                               * kernel_shape[2])
                        ), 5e-2)
        self.custom_padding = lambda inp: pad_2d(
            inp, padding, 'atrous_conv2d',
            kernel_shape[0], kernel_shape[1], dilation=rate
        )

        self.conv = torch.nn.Conv2d(kernel_shape[2], kernel_shape[3],
                                    kernel_shape[:2],
                                    dilation=rate)
        self.conv.weight.data.normal_(0.0, self.stddev)
        self.conv.bias.data.normal_(0.1, 0.00001)
        self.use_bn = use_bn
        self.use_mvn = use_mvn
        self.activation = activation
        self.dropout_maps = dropout_maps
        self.use_lrn = use_lrn
        if use_bn:
            self.bn = torch.nn.BatchNorm2d(kernel_shape[3])
        if use_mvn:
            # The smae as bn right now...
            self.mvn = torch.nn.BatchNorm2d(kernel_shape[3])
        if activation:
            self.activation = activation()
        if use_lrn:
            self.lrn = torch.nn.modules.normalization.LocalResponseNorm(kernel_shape[3])
        if dropout_maps:
            self.dropout = torch.nn.modules.Dropout2d(p=(1.0-keep_prob))
        else:
            self.dropout = torch.nn.modules.Dropout2d(p=(1.0-keep_prob))
        self.keep_prob = keep_prob

    def forward(self, x):
        x = self.custom_padding(x)
        outputs = self.conv(x)
        if self.use_bn:
            outputs = self.bn(outputs)
        if self.use_mvn:
            outputs = self.mvn(outputs)
        if self.activation:
            outputs = self.activation(outputs)
        if self.use_lrn:
            outputs = self.lrn(outputs)
        outputs = self.dropout(outputs)
        return outputs


class SeparableRNNBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filter_out, keep_prob=1.0,
                 cell_type='LSTM'):
        super(SeparableRNNBlock, self).__init__()

    def forward(self, x):
        return x


class DownSampleResNet(torch.nn.Module):
    def __init__(self, channel_in, channel_out, filter_size, res_depth,
                 pool_size, activation):
        super(DownSampleResNet, self).__init__()
        self.conv_res_list = torch.nn.ModuleList()
        for a_res in range(res_depth):
            if a_res < res_depth - 1:
                self.conv_res_list.append(Conv2dBnLrnDrop(
                    [filter_size, filter_size, channel_in, channel_in],
                    activation=activation))
            else:
                self.conv_res_list.append(Conv2dBnLrnDrop(
                    [filter_size, filter_size, channel_in, channel_in]))
        self.activation = activation
        if activation is not None:
            self.activation = activation()
        self.max_pool = torch.nn.MaxPool2d(pool_size, pool_size, pool_size//2)
        self.class_aux = Conv2dBnLrnDrop([4, 4, channel_in, channel_out], False,
                                         1, 2, activation=torch.nn.ReLU)

    def forward(self, x):
        orig_x = x
        for conv_op in self.conv_res_list:
            x = conv_op(x)
        x += orig_x
        x = self.activation(x)
        x = self.max_pool(x)
        x = self.class_aux(x)
        return x


class Deconv2DBnLrnDrop(torch.nn.Module):
    def __init__(self, kernel_shape, sub_s=2,
                 activation=torch.nn.ReLU,
                 use_bn=False, use_mvn=False, use_lrn=False,
                 keep_prob=1.0,
                 dropout_maps=False,
                 initOpt=0):
        super(Deconv2DBnLrnDrop, self).__init__()
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
        self.conv = torch.nn.ConvTranspose2d(
            kernel_shape[3], kernel_shape[2],
            kernel_shape[:2],
            stride=sub_s,
            padding=(int(kernel_shape[0]// 2), kernel_shape[1]//2),
            bias=True)
        self.conv.weight.data.normal_(0.0, stddev)
        self.conv.bias.data.normal_(0.1, 0.00001)
        self.use_bn = use_bn
        self.use_mvn = use_mvn
        self.activation = activation
        self.dropout_maps = dropout_maps
        self.use_lrn = use_lrn
        if use_bn:
            self.bn = torch.nn.BatchNorm2d(kernel_shape[2])
        if use_mvn:
            # The smae as bn right now...
            self.mvn = torch.nn.BatchNorm2d(kernel_shape[2])
        if activation:
            self.activation = activation()
        if use_lrn:
            self.lrn = torch.nn.modules.normalization.LocalResponseNorm(kernel_shape[2])
        if dropout_maps:
            self.dropout = torch.nn.modules.Dropout2d(p=(1.0-keep_prob))
        else:
            self.dropout = torch.nn.modules.Dropout2d(p=(1.0-keep_prob))
        self.keep_prob = keep_prob

    def forward(self, x, output_size=None):
        outputs = self.conv(x, output_size=output_size)
        if self.use_bn:
            outputs = self.bn(outputs)
        if self.use_mvn:
            outputs = self.mvn(outputs)
        if self.activation:
            outputs = self.activation(outputs)
        if self.use_lrn:
            outputs = self.lrn(outputs)
        outputs = self.dropout(outputs)
        return outputs
