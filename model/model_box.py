from __future__ import print_function, division
import torch
from .layers import layers
from .layers import attention
from collections import OrderedDict
from box_convolution import BoxConv2d


class MultiBoxConvBlock(torch.nn.Module):
    '''
    Args:
        channels: number of in channels (also out channels)
        num_convs: number of BoxConv followed by 1x1Conv
        num_boxs: number of Box Filters per channels
        max_box_h_w: maximum box size (h, w)
        activation: last activation
    '''
    def __init__(self, channels, num_convs, num_boxs, max_box_h_w,
                 activation):
        super(MultiBoxConvBlock, self).__init__()
        self.conv_list = torch.nn.ModuleList()
        self.num_convs = num_convs
        if isinstance(max_box_h_w, int):
            max_box_h_w = (max_box_h_w, max_box_h_w)
        self.max_box_size = max_box_h_w
        self.num_boxs = num_boxs
        self.activation = None
        if activation is not None:
            self.activation = activation()
        for i_conv in range(self.num_convs):
            self.conv_list.append(
                    BoxConv2d(channels, num_boxs, max_box_h_w[0], max_box_h_w[1])

            )
            if i_conv < self.num_convs-1:
                self.conv_list.append(layers.Conv2dBnLrnDrop(
                        [1, 1, num_boxs*channels, channels],
                        activation=activation,
                        use_sparse_conv=False
                ))
            else:
                self.conv_list.append(
                    layers.Conv2dBnLrnDrop(
                        [1, 1, num_boxs*channels, channels],
                        activation=None,
                        use_sparse_conv=False
                    )
                )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        orig_x = x
        x = self.relu(x)
        for i_conv in range(self.num_convs*2):
            x = self.conv_list[i_conv](x)
        x += orig_x
        if self.activation is not None:
            x = self.activation(x)
        return x


class BDownSamplingUNetBlock(torch.nn.Module):
    '''
    DownSamplingUNetBlock may be used for both original and coupled U-Net
    This is the modified version with BoxConvolution instead of MultiResBlock
    Args:
        channels: num inp channels
        scale_space_num: number of downsamplings
        filter_size: normal conv ops of UNet filter size (int)
        num_box_convs: number of box convolutions in the MultiBoxConvs
        feat_root: starting number of channels for first conv->maxpool, each time pooled, number of channels will be multiplied by pool_size
        max_box_sizes: maximum box size (int or tuple)
        pool_size: pooling size (int)
        activation: the activation function
        use_prev_coupled: is not the first block in the coupled UNet?
    '''
    def __init__(self, channels, scale_space_num,
                 filter_size, num_box_convs, num_box_per_channels,
                 feat_root, max_box_sizes, pool_size,
                 activation, use_prev_coupled=False):
        super(BDownSamplingUNetBlock, self).__init__()
        self.input_channels = channels
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.scale_space_num = scale_space_num

        self.num_box_convs = num_box_convs
        self.conv_box_list = torch.nn.ModuleList()
        self.max_box_sizes = max_box_sizes
        self.num_box_per_channels = num_box_per_channels

        self.act_feat_num = feat_root
        self.relu = torch.nn.ReLU()
        self.conv1s = torch.nn.ModuleList()
        self.conv1_1s = torch.nn.ModuleList()
        self.use_prev_coupled = use_prev_coupled
        self.layer_attentions = torch.nn.ModuleList()
        self.custom_pad = lambda inp: layers.pad_2d(inp, 'SAME',
            kind='pool2d',
            k_h=self.ksize_pool[0],
            k_w=self.ksize_pool[1],
            s_h=self.stride_pool[0],
            s_w=self.stride_pool[1])
        self.max_pool = torch.nn.MaxPool2d(
                self.ksize_pool, self.stride_pool)
        self.activation = activation
        self.last_feat_num = channels
        self.act_feat_num = feat_root
        for layer in range(0, scale_space_num):
            self.conv1s.append(layers.DilConv2dBnLrnDrop(
                    [filter_size, filter_size,
                     self.last_feat_num, self.act_feat_num],
                    rate=2 ** layer, padding='SAME',
                    activation=None))

            self.conv_box_list.append(
                MultiBoxConvBlock(
                    self.act_feat_num, self.num_box_convs,
                    self.num_box_per_channels, self.max_box_sizes,
                    self.activation
                ))
            if use_prev_coupled:
                self.conv1_1s.append(layers.Conv2dBnLrnDrop(
                    [1, 1, 2*self.act_feat_num, self.act_feat_num],
                    activation=activation))
            if layer > scale_space_num - 2:
                self.layer_attentions = attention.SAWrapperBlock(
                    self.act_feat_num
                )


            self.last_feat_num = self.act_feat_num
            self.act_feat_num *= pool_size

    def forward(self, unet_inp, prev_dw_h_convs=None):
        '''
        :param prev_dw_h_convs: previous down-sampling tower's outputs
                                (used for coupling connection)
        '''
        dw_h_convs = OrderedDict()
        for layer in range(0, self.scale_space_num):
            x = self.conv1s[layer](unet_inp)
            x = self.conv_box_list[layer](x)
            if self.use_prev_coupled:
                assert(prev_dw_h_convs is not None),\
                    "ERROR: Second Unet block not fed with previous data"
                prev_dw_h_conv = prev_dw_h_convs[layer]
                x = torch.cat([prev_dw_h_conv, x], dim=1)
                x = self.conv1_1s[layer](x)
            if layer > self.scale_space_num - 2:
                dw_h_convs[layer] = self.layer_attentions(x)
            else:
                dw_h_convs[layer] = x

            if layer < self.scale_space_num - 1:
                x = self.custom_pad(x)
                unet_inp = self.max_pool(x)
            else:
                unet_inp = x
        # print("UNet Output: ", [(layer, down_sampled.size()) for layer, down_sampled in  dw_h_convs.items()])
        return dw_h_convs, x


class BUpSamplingUNetBlock(torch.nn.Module):
    '''
    BoxUpSamplingUNetBlock, may be used for both original and coupled U-Net
    '''
    def __init__(self, channels, scale_space_num,
                 filter_size, num_box_convs, num_box_per_channels,
                 feat_root, max_box_sizes, pool_size,
                 activation, last_feat_num, act_feat_num,
                 use_prev_coupled=False):
        super(BUpSamplingUNetBlock, self).__init__()
        self.input_channels = channels
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.scale_space_num = scale_space_num

        self.num_box_convs = num_box_convs
        self.max_box_sizes = max_box_sizes
        self.num_box_per_channels = num_box_per_channels

        self.conv_box_list = torch.nn.ModuleList()
        self.act_feat_num = feat_root
        self.conv1s = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.conv1s = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.conv1_1s = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.deconvs = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.use_prev_coupled = use_prev_coupled
        self.layer_attentions = torch.nn.ModuleList()
        self.channels = channels
        self.conv_box_list = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.activation = activation
        self.act_feat_num = act_feat_num
        self.last_feat_num = last_feat_num
        for layer in range(self.scale_space_num - 2, -1, -1):
            self.deconvs[layer] = layers.Deconv2DBnLrnDrop(
                [filter_size, filter_size,
                 self.act_feat_num, self.last_feat_num],
                activation=None)
            self.conv1s[layer] = layers.Conv2dBnLrnDrop(
                    [filter_size, filter_size,
                     pool_size*self.act_feat_num,
                     self.act_feat_num], activation=None)

            self.conv_box_list[layer] = MultiBoxConvBlock(
                self.act_feat_num, self.num_box_convs,
                self.num_box_per_channels, self.max_box_sizes,
                self.activation)
            if self.use_prev_coupled:
                self.conv1_1s[layer] = layers.Conv2dBnLrnDrop(
                    [1, 1, 2 * self.act_feat_num, self.act_feat_num],
                    activation=activation)

            self.last_feat_num = self.act_feat_num
            self.act_feat_num //= pool_size

    def forward(self, dw_h_convs, down_sampled_out, prev_up_h_convs=None):
        up_dw_h_convs = OrderedDict()
        for layer in range(self.scale_space_num - 2, -1, -1):
            dw_h_conv = dw_h_convs[layer]
            # Need to pad
            # print("Down sampled out size: ", down_sampled_out.size())
            deconv = self.deconvs[layer](
                    down_sampled_out, output_size=dw_h_conv.size()[2:]
                    )
            # print("Target size: ", dw_h_conv.size())
            # print("Deconv out size: ", deconv.size())
            conc = torch.cat([dw_h_conv, deconv], dim=1)
            x = self.conv1s[layer](conc)
            x = self.conv_box_list[layer](x)
            if self.use_prev_coupled:
                assert prev_up_h_convs is not None,\
                    "ERROR: Use coupled but no data provided in \
                    upsampling block"
                prev_up_dw_h_conv = prev_up_h_convs[layer]
                x = torch.cat([prev_up_dw_h_conv, x], dim=1)
                x = self.conv1_1s[layer](x)
            up_dw_h_convs[layer] = x
            down_sampled_out = x
        up_sampled_out = down_sampled_out
        return up_sampled_out, dw_h_convs, up_dw_h_convs


class BUNetBlock(torch.nn.Module):
    """
    UNetBlock according to the model
    :param channels: number of input channels
    :param scale_space_num: number of down-sampling / up-sampling blocks
    :param num_box_convs: number of convolution layers in a multi box convolution block
    :param featRoot: number of features in the first layers
    :param filter_size: convolution kernel size
    :param max_box_sizes: maximum block size
    :param pool_size: pooling size
    :param activation: activation function
    :param use_prev_coupled: is this the second in the coupled block?

    :return:

    """
    def __init__(self, channels, scale_space_num,
                 num_box_convs, num_box_per_channels,
                 feat_root, filter_size, max_box_sizes,
                 pool_size, activation,
                 use_prev_coupled=False):
        super(BUNetBlock, self).__init__()
        self.input_channels = channels
        self.pool_size = pool_size
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.scale_space_num = scale_space_num

        self.num_box_convs = num_box_convs
        self.max_box_sizes = max_box_sizes
        self.num_box_per_channels = num_box_per_channels

        self.channels = channels
        self.act_feat_num = feat_root
        self.activation = activation
        self.use_prev_coupled = use_prev_coupled

        print("Use prev coupled: ", use_prev_coupled)

        self.downsamplingblock = BDownSamplingUNetBlock(
                channels, scale_space_num,
                filter_size, num_box_convs, num_box_per_channels,
                feat_root, max_box_sizes, pool_size,
                activation, self.use_prev_coupled)

        self.act_feat_num = self.downsamplingblock.last_feat_num // pool_size
        self.last_feat_num = self.downsamplingblock.last_feat_num

        self.upsamplingblock = BUpSamplingUNetBlock(
            self.channels, self.scale_space_num,
            filter_size, self.num_box_convs, self.num_box_per_channels,
            self.act_feat_num, self.max_box_sizes, pool_size,
            self.activation, self.last_feat_num, self.act_feat_num,
            self.use_prev_coupled)

    def forward(self, unet_inp, prev_dw_h_convs=None,
                prev_up_h_convs=None,
                binary_mask=None):
        dw_h_convs, x = self.downsamplingblock(unet_inp,
                                               prev_dw_h_convs,
                                               )
        unet_inp = x
        up_sampled_out, dw_h_convs, up_dw_h_convs = self.upsamplingblock(
            dw_h_convs, unet_inp, prev_up_h_convs=prev_up_h_convs)
        return up_sampled_out, dw_h_convs, up_dw_h_convs


class BMSAUNet(torch.nn.Module):
    def __init__(self, channels, n_class, scale_space_num,
                 num_box_convs, num_box_per_channels, feat_root,
                 filter_size, max_box_sizes, pool_size, activation):
        super(BMSAUNet, self).__init__()

        self.num_blocks = 3     # Number of Unet Blocks
        self.blocks = torch.nn.ModuleList()
        self.end_convs = torch.nn.ModuleList()
        for block_id in range(self.num_blocks):
            if block_id == 0:
                use_prev_coupled = False
                num_channels = channels
            else:
                num_channels = n_class
                use_prev_coupled = True

            self.blocks.append(BUNetBlock(
                 num_channels, scale_space_num,
                 num_box_convs, num_box_per_channels, feat_root,
                 filter_size, max_box_sizes, pool_size,
                 activation,
                 use_prev_coupled=use_prev_coupled))
            self.end_convs.append(layers.Conv2dBnLrnDrop(
                [4, 4, feat_root, n_class], activation=None))

    def forward(self, inp):
        inp_scale_map = OrderedDict()
        inp_scale_map[0] = inp
        binary_mask = None
        prev_dw_h_convs = None
        prev_up_h_convs = None
        logits_aux = None
        for block_id in range(self.num_blocks):
            out, prev_dw_h_convs, prev_up_h_convs =\
                self.blocks[block_id](inp, prev_dw_h_convs=prev_dw_h_convs,
                                      prev_up_h_convs=prev_up_h_convs,
                                      binary_mask=binary_mask)
            out = self.end_convs[block_id](out)
            inp = out
            if block_id == self.num_blocks - 2:
                logits_aux = out
        out_map = out
        logits = out_map
        return logits, logits_aux


class BMSAUWrapper(torch.nn.Module):
    def __init__(self, channels=1, n_class=2, model_kwargs={}):
        super(BMSAUWrapper, self).__init__()
        self.n_class = n_class
        self.channels = channels

        ### model hyper-parameters
        self.scale_space_num = model_kwargs.get("scale_space_num", 6)
        self.num_box_convs = model_kwargs.get("num_box_convs", 3)
        self.max_box_sizes = model_kwargs.get("max_box_sizes", 28)
        self.num_box_per_channels = model_kwargs.get("num_box_per_channels", 3)
        self.featRoot = model_kwargs.get("featRoot", 8)
        self.filter_size = model_kwargs.get("filter_size", 3)
        self.pool_size = model_kwargs.get("pool_size", 2)

        self.activation_name = model_kwargs.get("activation_name", "relu")
        if self.activation_name == "relu":
            self.activation = torch.nn.ReLU
        if self.activation_name == "elu":
            self.activation = torch.nn.ELU
        self.model = model_kwargs.get("model", "msau")
        self.num_scales = model_kwargs.get("num_scales", 3)
        self.final_act = model_kwargs.get("final_act", "sigmoid")
        self.msau_net = BMSAUNet(self.channels, self.n_class,
                                self.scale_space_num, self.num_box_convs,
                                self.num_box_per_channels,
                                self.featRoot, self.filter_size,
                                self.max_box_sizes,
                                self.pool_size, self.activation)

        if self.final_act == "softmax":
            self.predictor = torch.nn.Softmax(dim=1)
        elif self.final_act == "sigmoid":
            self.predictor = torch.nn.Sigmoid(dim=1)
        elif self.final_act == "identity":
            self.predictor = torch.nn.Sequential()

    def forward(self, inp):
        logits, aux_logits = self.msau_net(inp)
        return self.predictor(logits), logits, aux_logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        pretrained_dict = torch.load(path)
        self.load_state_dict(pretrained_dict)
