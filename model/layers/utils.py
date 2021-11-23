import torch.nn.functional as F
import numpy as np


def pad_2d(inp, padding, kind, k_h, k_w, s_h=1, s_w=1, dilation=1):
    if padding == 'VALID':
        return inp
    elif padding == 'SAME' and kind in ('conv2d', 'pool2d'):
        in_height, in_width = inp.size(2), inp.size(3)
        out_height = int(np.ceil(float(in_height) / float(s_h)))
        out_width = int(np.ceil(float(in_width) / float(s_w)))

        pad_along_height = max((out_height - 1) * s_h + k_h - in_height, 0)
        pad_along_width = max((out_width - 1) * s_w + k_w - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        inp = F.pad(inp, (pad_left, pad_right, pad_top, pad_bottom))
        return inp
    elif kind in ('atrous_conv2d',):
        effective_height = k_h + (k_h - 1) * (dilation - 1)
        effective_width = k_w + (k_w - 1) * (dilation - 1)
        return pad_2d(inp, padding,
                      'conv2d', effective_height,
                      effective_width, s_h, s_w, dilation=1)
    else:
        raise NotImplementedError


