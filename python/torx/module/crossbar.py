# Copyright 2019 The PytorX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from saf import SAF
from w2g import w2g


class crossbar_Conv2d(nn.Module):
    def __init__(self,
                 weight,
                 dilation,
                 padding,
                 stride,
                 nbits=7,
                 Gmax=1 / 3e3,
                 Gmin=1 / 3e6,
                 crxb_size=32,
                 enable_SAF=False):
        super(crossbar_Conv2d, self).__init__()
        r"""
        This module consists the function:
        1) weight quantization
        2) weight partition (i.e., map to multiple crossbars).
        3) use two resistive memory cells to represent single weight.
        4) convert weights to conductance.

        Args:
            weight (Tensor): weight tensor
            nbits (int): number of bits (resolution) of single resistive memory cell.
            Gmax (fp): maximum conductance (* in unit of S).
            Gmin (fp): minimum conductance (* in unit of S).
            crxb_size (int): crossbar dimension in (crxb_size, crxb_size)
        """
        # Get the configuration from Conv module
        self.weight_size = weight.size()
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        # crossbar basic setup
        self.nbits = nbits
        self.Gmax = Gmax
        self.Gmin = Gmin
        self.crxb_size = crxb_size
        self.half_lvl = 2**self.nbits - 1  # number of lvls determined by ReRAM cell
        self.delta_g = (self.Gmax - self.Gmin) / self.half_lvl

        # configurations for proper reshape, add padding to fit
        weight_row_length = np.prod(self.weight_size[1:])
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            weight_row_length, self.crxb_size)
        weight_col_length = self.weight_size[0]
        self.crxb_col, self.crxb_col_pads = self.num_pad(
            weight_col_length, self.crxb_size)

        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)

        # compute the crossbar shape then define the weight-2-conductance module
        weight_flatten = weight.view(self.weight_size[0], -1)
        weight_padded = F.pad(weight_flatten,
                              self.w_pad,
                              mode='constant',
                              value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row,
                                         self.crxb_size).transpose(1, 2)

        # additional module to perform the conversion between fp32 weight to conductance
        self.w2g = w2g(self.delta_g,
                       Gmin=self.Gmin,
                       G_SA0=self.Gmax,
                       G_SA1=self.Gmin,
                       weight_shape=weight_crxb.shape)

    def forward(self, input, weight):
        r'''
        Perform the computation between input voltage and weight.

        Args:
            input (fp, voltage): the un-reshaped input voltage tensor
            weight (fp): the un-reshape weight tensor

        '''
        # 1. unfold the input
        input_unfold = F.unfold(input,
                                kernel_size=self.weight_size[3],
                                dilation=self.dilation,
                                padding=self.padding,
                                stride=self.stride)

        # quantize and flatten the weight
        with torch.no_grad():
            self.delta_w = weight.abs().max() / self.half_lvl

        weight_quan = quantize_weight(weight, self.delta_w)
        # self.weight_size[0] is number of output channels
        weight_flatten = weight_quan.view(self.weight_size[0], -1)

        # 2. add paddings
        input_padded = F.pad(input_unfold,
                             self.input_pad,
                             mode='constant',
                             value=0)
        weight_padded = F.pad(weight_flatten,
                              self.w_pad,
                              mode='constant',
                              value=0)

        # 3. reshape both input and weight tensor w.r.t crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row,
                                         self.crxb_size).transpose(1, 2)

        # convert the floating point weight into conductance pair values
        # G_crxb[0] and G_crxb[1] are postive and negative arrays respectively
        G_crxb = self.w2g(weight_crxb)

        # 4. compute matrix multiplication followed by reshapes
        output = torch.matmul(G_crxb[0], input_crxb) - \
            torch.matmul(G_crxb[1], input_crxb)

        return output

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding


class _quantize_weight(torch.autograd.Function):
    r'''
    weight quantization function. For most of the weight quantization work,
    there is no clipping operation applied on the weight. In other word, the
    clippling threshold is the maximum of the weight, which is supposed to be 
    done in calculate the quantization step (delta_w).
    '''

    @staticmethod
    def forward(ctx, input, delta_w):
        ctx.delta_w = delta_w
        output = torch.round(input / delta_w)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.delta_w
        return grad_input, None


quantize_weight = _quantize_weight.apply

