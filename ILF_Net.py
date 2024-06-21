# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import torch.nn as nn
from torch import Tensor


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)  # conv_layer定义卷积自适应
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        # 到这里 第一个patch 相同
        out = self.c5(out)
        return out


class QPAttention(nn.Module):
    def __init__(self, in_channels=48):
        super(QPAttention, self).__init__()
        out_channels = in_channels
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = gamma * self.res(x) + beta

        return x + res


class RLFB_QPA(nn.Module):
    def __init__(self, in_nc, nf, nb):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.  default = 8
        """
        super(RLFB_QPA, self).__init__()

        self.nb = nb
        self.in_nc = in_nc

        # rdb backbone

        for i in range(1, nb):
            setattr(
                self, 'RLFB{}'.format(i), nn.Sequential(
                    RLFB(nf)
                )
            )

            setattr(
                self, 'qp_att{}'.format(i),
                QPAttention(nf)
            )

        self.liner_gamma = nn.Linear(1, nf)
        self.liner_beta = nn.Linear(1, nf)

    def forward(self, qp, inputs):
        nb = self.nb
        gamma = self.liner_gamma(qp)
        beta = self.liner_beta(qp)
        out = inputs
        # skip = out
        for i in range(1, nb):
            rlfb = getattr(self, 'RLFB{}'.format(i))

            qp_att = getattr(self, 'qp_att{}'.format(i))

            out = rlfb(out)
            out = qp_att(out, gamma, beta)
        return out


class Generator(nn.Module):
    def __init__(self, in_channel, nf) -> None:
        super(Generator, self).__init__()
        # The first layer of convolutional layer.

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.rlfb_qpa = RLFB_QPA(
            in_nc=1,
            nf=nf,
            nb=7,
        )

        self.add_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, 1, 3, 1, 1),
            nn.ReLU(True)
        )

    # The model should be defined in the Torch.script method.
    def forward(self, qp, x: Tensor) -> Tensor:
        qp = qp.unsqueeze(1)
        out1 = self.in_conv(x)
        out_rlfb = self.rlfb_qpa(qp, out1)
        out = self.add_conv(out1 + out_rlfb)
        out = self.out_conv(out)    # the last output layer is ReLU.

        return out



