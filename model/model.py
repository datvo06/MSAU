from __future__ import print_function, division
import torch
from .layers import layers
from .layers import attention
from collections import OrderedDict


class MultiConvResidualBlock(torch.nn.Module):
    def __init__(self, res_depth, filter_size, channels,
                 use_sparse_conv, activation):
        super(MultiConvResidualBlock, self).__init__()
        self.conv_res_list = torch.nn.ModuleList()
        self.use_sparse_conv = use_sparse_conv
        self.res_depth = res_depth
        self.activation = None
        if activation is not None:
            self.activation = activation()
        for a_res in range(0, res_depth):
            if a_res < res_depth-1:
                self.conv_res_list.append(
                    layers.Conv2dBnLrnDrop(
                        [filter_size, filter_size, channels, channels],
                        activation=activation,
                        use_sparse_conv=self.use_sparse_conv
                    )
                )
            else:
                self.conv_res_list.append(
                    layers.Conv2dBnLrnDrop(
                        [filter_size, filter_size, channels, channels],
                        activation=None,
                        use_sparse_conv=self.use_sparse_conv
                    )
                )
        self.relu = torch.nn.ReLU()

    def forward(self, x, binary_mask=None):
        orig_x = x
        x = self.relu(x)
        for a_res in range(self.res_depth):
            if self.use_sparse_conv:
                x, binary_mask = self.conv_res_list[a_res](x, binary_mask)
            else:
                x = self.conv_res_list[a_res](x, binary_mask)
        x += orig_x
        if self.activation is not None:
            x = self.activation(x)
        if self.use_sparse_conv:
            return x, binary_mask
        return x


class DownSamplingUNetBlock(torch.nn.Module):
    '''
    DownSamplingUNetBlock may be used for both original and coupled U-Net

    Args:
        use_residual: whether to use the multiple conv blocks with residual connections
        channels: num inp channels
        scale_space_num: number of downsamplings
        res_depth: the residual multi conv blocks depth
        filter_size: size of each filter (int)
        pool_size: pooling size (int)
        activation: the activation function
        use_sparse_conv: whether to use the sparse convolution op
        use_prev_coupled: is not the first block in the coupled UNet?
    '''
    def __init__(self, use_residual, channels, scale_space_num,
                 res_depth, feat_root, filter_size, pool_size,
                 activation, use_sparse_conv=False,
                 use_prev_coupled=False):
        super(DownSamplingUNetBlock, self).__init__()
        self.input_channels = channels
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.use_sparse_conv = False
        self.scale_space_num = scale_space_num
        self.use_residual = use_residual
        self.res_depth = res_depth
        self.conv_res_list = torch.nn.ModuleList()
        self.act_feat_num = feat_root
        self.conv1s = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
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
            if self.use_residual:
                self.conv1s.append(layers.DilConv2dBnLrnDrop(
                    [filter_size, filter_size,
                     self.last_feat_num, self.act_feat_num],
                    rate=2 ** layer, padding='SAME',
                    activation=None))
                self.conv_res_list.append(
                    MultiConvResidualBlock(
                        res_depth, filter_size, self.act_feat_num,
                        self.use_sparse_conv, self.activation))
                if use_prev_coupled:
                    self.conv1_1s.append(layers.Conv2dBnLrnDrop(
                        [1, 1, 2*self.act_feat_num, self.act_feat_num],
                        activation=activation))
                if layer > scale_space_num - 2:
                    self.layer_attentions = attention.SAWrapperBlock(
                        self.act_feat_num
                    )
            else:
                self.conv1s.append(layers.Conv2dBnLrnDrop(
                    [filter_size, filter_size, channels, feat_root],
                    activation=activation))
                self.conv1_1s.append(layers.Conv2dBnLrnDrop(
                    [filter_size, filter_size,
                     self.act_feat_num, self.act_feat_num],
                    activation=activation))

            self.last_feat_num = self.act_feat_num
            self.act_feat_num *= pool_size

    def forward(self, unet_inp, prev_dw_h_convs=None,
                binary_mask=None):
        '''
        :param prev_dw_h_convs: previous down-sampling tower's outputs
                                (used for coupling connection)
        '''
        dw_h_convs = OrderedDict()
        for layer in range(0, self.scale_space_num):
            if self.use_residual:
                x = self.conv1s[layer](unet_inp)
                if self.use_sparse_conv:
                    x, binary_mask = self.conv_res_list[layer](x, binary_mask)
                else:
                    x = self.conv_res_list[layer](x)
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
            else:
                conv1 = self.conv1s[layer](unet_inp)
                dw_h_convs[layer] = self.conv1_1s[layer](conv1)
                x = dw_h_convs[layer]

            if layer < self.scale_space_num - 1:
                x = self.custom_pad(x)
                unet_inp = self.max_pool(x)
            else:
                unet_inp = x
        # print("UNet Output: ", [(layer, down_sampled.size()) for layer, down_sampled in  dw_h_convs.items()])
        return dw_h_convs, x


class UpSamplingUNetBlock(torch.nn.Module):
    '''
    UpSamplingUNetBlock, may be used for both original and coupled U-Net
    '''
    def __init__(self, use_residual, channels, scale_space_num,
                 res_depth, feat_root, filter_size, pool_size,
                 activation, last_feat_num, act_feat_num,
                 use_prev_coupled=False):
        super(UpSamplingUNetBlock, self).__init__()
        self.input_channels = channels
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.scale_space_num = scale_space_num
        self.use_residual = use_residual
        self.res_depth = res_depth
        self.conv_res_list = torch.nn.ModuleList()
        self.act_feat_num = feat_root
        self.conv1s = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.res_depth = res_depth
        self.conv1s = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.conv1_1s = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.deconvs = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.use_prev_coupled = use_prev_coupled
        self.layer_attentions = torch.nn.ModuleList()
        self.channels = channels
        self.conv_res_list = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.activation = activation
        self.act_feat_num = act_feat_num
        self.last_feat_num = last_feat_num
        for layer in range(self.scale_space_num - 2, -1, -1):
            self.deconvs[layer] = layers.Deconv2DBnLrnDrop(
                [filter_size, filter_size,
                 self.act_feat_num, self.last_feat_num],
                activation=None)
            if self.use_residual:
                self.conv1s[layer] = layers.Conv2dBnLrnDrop(
                        [filter_size, filter_size,
                         pool_size*self.act_feat_num,
                         self.act_feat_num], activation=None)
                self.conv_res_list[layer] = MultiConvResidualBlock(
                    res_depth, filter_size, self.act_feat_num, False,
                    self.activation)
                if self.use_prev_coupled:
                    self.conv1_1s[layer] = layers.Conv2dBnLrnDrop(
                        [1, 1, 2 * self.act_feat_num, self.act_feat_num],
                        activation=activation)
            else:
                self.conv1s[layer] = layers.Conv2dBnLrnDrop(
                    [filter_size, filter_size, pool_size * self.act_feat_num,
                     self.act_feat_num], activation=self.activation)
                self.conv1_1s[layer] = layers.Conv2dBnLrnDrop(
                    [filter_size, filter_size, self.act_feat_num,
                     self.act_feat_num], activation=self.activation)
            self.last_feat_num = self.act_feat_num
            self.act_feat_num //= pool_size

    def forward(self, dw_h_convs, down_sampled_out, prev_up_h_convs=None):
        up_dw_h_convs = OrderedDict()
        for layer in range(self.scale_space_num - 2, -1, -1):
            dw_h_conv = dw_h_convs[layer]
            # Need to pad
            # print("Down sampled out size: ", down_sampled_out.size())
            deconv = self.deconvs[layer](down_sampled_out, output_size=dw_h_conv.size()[2:])
            # print("Target size: ", dw_h_conv.size())
            # print("Deconv out size: ", deconv.size())
            '''

            diffY = dw_h_conv.size()[2] - deconv.size()[2]
            diffX = dw_h_conv.size()[3] - deconv.size()[3]
            deconv = torch.nn.functional.pad(
                deconv, (diffX // 2, diffX - diffX//2,
                         diffY // 2, diffY - diffY//2))
            '''

            conc = torch.cat([dw_h_conv, deconv], dim=1)
            if self.use_residual:
                x = self.conv1s[layer](conc)
                x = self.conv_res_list[layer](x)
                if self.use_prev_coupled:
                    assert prev_up_h_convs is not None,\
                        "ERROR: Use coupled but no data provided in \
                        upsampling block"
                    prev_up_dw_h_conv = prev_up_h_convs[layer]
                    x = torch.cat([prev_up_dw_h_conv, x], dim=1)
                    x = self.conv1_1s[layer](x)
                up_dw_h_convs[layer] = x
                down_sampled_out = x
            else:
                conv1 = self.conv1s[layer](conc)
                down_sampled_out = self.conv2s(conv1)
        up_sampled_out = down_sampled_out
        return up_sampled_out, dw_h_convs, up_dw_h_convs


class UNetBlock(torch.nn.Module):
    """
    UNetBlock according to the model
    :param input: input image
    :param useResidual: use residual connection (ResNet)
    :param use_lstm: run a separable LSTM horizontally then
                    vertically across input features
    :param useSPN: use Spatial Propagation Network
    :param channels: number of input channels
    :param scale_space_num: number of down-sampling / up-sampling blocks
    :param res_depth: number of convolution layers in a down-sampling block
    :param featRoot: number of features in the first layers
    :param filter_size: convolution kernel size
    :param pool_size: pooling size
    :param activation: activation function
    :param use_prev_coupled: is this the second in the coupled block?

    :return:

    """
    def __init__(self, use_residual, use_lstm, use_spn, channels,
                 scale_space_num, res_depth, feat_root, filter_size, pool_size,
                 activation, use_sparse_conv=False,
                 use_prev_coupled=False):
        super(UNetBlock, self).__init__()
        self.input_channels = channels
        self.pool_size = pool_size
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.use_sparse_conv = use_sparse_conv
        self.scale_space_num = scale_space_num
        self.use_residual = use_residual
        self.use_spn = use_spn
        self.res_depth = res_depth
        self.channels = channels
        self.act_feat_num = feat_root
        self.activation = activation
        self.use_prev_coupled = use_prev_coupled

        print("Using sparse conv: ", use_sparse_conv)
        print("Use prev coupled: ", use_prev_coupled)

        self.downsamplingblock = DownSamplingUNetBlock(
            use_residual, channels, scale_space_num,
            res_depth, feat_root, filter_size,
            pool_size, activation, use_sparse_conv,
            use_prev_coupled)
        self.act_feat_num = self.downsamplingblock.last_feat_num // pool_size
        self.last_feat_num = self.downsamplingblock.last_feat_num

        self.lstm = None
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = layers.SeparableRNNBlock(self.act_feat_num,
                                                 self.last_feat_num,
                                                 cell_type='LSTM')
        if self.use_spn:
            self.downsample_resnet = layers.DownSampleResNet(
                self.act_feat_num, self.act_feat_num,
                filter_size, res_depth, self.ksize_pool, activation)
            self.cspn = layers.cspn.Affinity_Propagate()
        self.upsamplingblock = UpSamplingUNetBlock(
            self.use_residual, self.channels, self.scale_space_num,
            res_depth, self.act_feat_num, filter_size, self.pool_size,
            self.activation, self.last_feat_num, self.act_feat_num, self.use_prev_coupled)

    def forward(self, unet_inp, prev_dw_h_convs=None,
                prev_up_h_convs=None,
                binary_mask=None):
        dw_h_convs, x = self.downsamplingblock(unet_inp,
                                               prev_dw_h_convs,
                                               binary_mask)
        unet_inp = x
        if self.use_lstm:
            unet_inp = self.lstm(unet_inp)
        if self.use_spn:
            guidance_out = self.downsample_resnet(
                dw_h_convs[self.scale_space_num - 2])
            unet_inp = self.cspn(guidance_out, unet_inp, None)

        up_sampled_out, dw_h_convs, up_dw_h_convs = self.upsamplingblock(
            dw_h_convs, unet_inp, prev_up_h_convs=prev_up_h_convs)
        return up_sampled_out, dw_h_convs, up_dw_h_convs


class MSAUNet(torch.nn.Module):
    def __init__(self, channels, n_class, scale_space_num, res_depth,
                 feat_root, filter_size, pool_size, activation):
        super(MSAUNet, self).__init__()
        use_residual = True
        use_lstm = False
        self.use_spn = False

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
            if self.use_spn and block_id == self.num_blocks - 1:
                enable_spn = True
            else:
                enable_spn = False
            self.blocks.append(UNetBlock(use_residual, use_lstm, enable_spn,
                                         num_channels, scale_space_num,
                                         res_depth,
                                         feat_root, filter_size, pool_size,
                                         activation, use_sparse_conv=False,
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


class MSAUWrapper(torch.nn.Module):
    def __init__(self, channels=1, n_class=2, model_kwargs={}):
        super(MSAUWrapper, self).__init__()
        self.n_class = n_class
        self.channels = channels

        ### model hyper-parameters
        self.scale_space_num = model_kwargs.get("scale_space_num", 6)
        self.res_depth = model_kwargs.get("res_depth", 3)
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

        self.msau_net = MSAUNet(self.channels, self.n_class,
                                self.scale_space_num, self.res_depth,
                                self.featRoot, self.filter_size,
                                self.pool_size, self.activation)

        if self.final_act == "softmax":
            self.predictor = torch.nn.Softmax(dim=1)
        elif self.final_act == "sigmoid":
            self.predictor = torch.nn.Sigmoid(dim=1)
        elif self.final_act == "identity":
            self.predictor = torch.nn.Sequential()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inp):
        logits, aux_logits = self.msau_net(inp)
        return self.predictor(logits), logits, aux_logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        pretrained_dict = torch.load(path)
        self.load_state_dict(pretrained_dict)

    def loss(self, out_grid, out_grid_aux, label_mask):
        '''
        Args:
            :param out_grid: 1xCxHxW
        '''
        # First, gather the point where label_mask != 0
        label_mask_expanded = (label_mask != 0
                               ).unsqueeze(0).repeat(1, self.n_class, 1, 1)
        out_grid = out_grid[label_mask_expanded].view(1, self.n_class, -1)
        out_grid_aux = out_grid_aux[label_mask_expanded].view(1, self.n_class,
                                                              -1)
        label_mask = label_mask[label_mask != 0].view(1, -1)
        return self.criterion(out_grid, label_mask) + self.criterion(
            out_grid_aux, label_mask)
