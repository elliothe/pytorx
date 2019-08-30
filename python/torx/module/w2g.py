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

import torch
import torch.nn as nn
import torch.nn.functional as F
from .saf import SAF


class W2G(nn.Module):
    ''' Convert float weight to conductance (Siemens).

    This module convert convert the post-quantization fixed point weight
    (weight_hat) into a pair of conductance values. output[0] is the G_pos
    and output[1] is the G_neg.

    Args:
        delta_g (in Siemens): resolution of memristor cell.
        g_min (in Siemens): the minimum conductance.
        g_sa0 (in Siemens): stuck-at-0 conductance.
        g_sa1 (in Siemens): stuck-at-1 conductance.
        p_sa0 (float): stuck-at-0 rate.
        p_sa1 (float): stuck-at-1 rate.
    '''
    def __init__(self,
                 delta_g,
                 g_min,
                 g_sa0,
                 g_sa1,
                 weight_shape,
                 enable_rand=False,
                 enable_saf=False):
        super(W2G, self).__init__()
        self.delta_g = delta_g
        self.g_min = g_min
        self.g_sa0 = g_sa0
        self.g_sa1 = g_sa1
        self.p_sa0 = 0
        self.p_sa1 = 0
        self.enable_rand = enable_rand
        self.enable_saf = enable_saf
        # define stuck-at-fault for postive and negative arrays separately.
        self.saf_pos = SAF(weight_shape,
                           p_sa0=self.p_sa0,
                           p_sa1=self.p_sa1,
                           g_sa0=self.g_sa0,
                           g_sa1=self.g_sa1)
        self.saf_neg = SAF(weight_shape,
                           p_sa0=self.p_sa0,
                           p_sa1=self.p_sa1,
                           g_sa0=self.g_sa0,
                           g_sa1=self.g_sa1)

    def forward(self, input):
        ''' input is the post-quantization weight tensor'''
        # x_relu() function is must-have, for correct pos/neg split.
        self.g_pos = self.g_min + X_RELU(input) * self.delta_g
        self.g_neg = self.g_min + F.relu(-input) * self.delta_g

        # the following two steps will update the SAF masking if enable_rand is True
        if self.enable_saf:
            output = torch.cat(
                (self.saf_pos(self.g_pos).unsqueeze(0), self.saf_neg(
                    self.g_neg).unsqueeze(0)), 0)
        else:
            output = torch.cat(
                (self.G_pos.unsqueeze(0), self.G_neg.unsqueeze(0)), 0)

        return output

    def error_correction(self):
        ''' error correction to compensate SAF error. '''
        pos_sa0 = self.saf_pos.index_SA0().float().cuda()
        pos_sa1 = self.saf_pos.index_SA1().float().cuda()
        neg_sa0 = self.saf_neg.index_SA0().float().cuda()
        neg_sa1 = self.saf_neg.index_SA1().float().cuda()
        g_pos_diff = (self._pos - self.g_sa0) * pos_sa0 + \
            (self.g_pos - self.g_sa1) * pos_sa1
        g_neg_diff = (self.G_neg - self.g_sa0) * neg_sa0 + \
            (self.g_neg - self.g_sa1) * neg_sa1

        return g_pos_diff, g_neg_diff

    def update_saf(self,
                   enable_saf,
                   p_sa0,
                   p_sa1,
                   new_saf_mask=False,
                   enable_rand=False):
        ''' update configuration for stuck-at-fault'''
        self.p_sa0 = p_sa0
        self.p_sa1 = p_sa1
        self.enable_saf = enable_saf
        # update the SAF_pos and SAF_neg modules
        self.saf_pos.p_SA0.data[0] = self.p_SA0
        self.saf_pos.p_SA1.data[0] = self.p_SA1
        self.saf_neg.p_SA0.data[0] = self.p_SA0
        self.saf_neg.p_SA1.data[0] = self.p_SA1
        # enable the random mask, thus each forward call get a new p_state mask
        self.enable_rand = enable_rand
        self.saf_pos.enable_rand = enable_rand
        self.saf_neg.enable_rand = enable_rand

        if new_saf_mask:
            self.saf_pos.p_state.data.uniform_()
            self.saf_neg.p_state.data.uniform_()


class _newrelu(torch.autograd.Function):
    '''
    This self-define function is used for mapping weight on positive
    and negative array. It will prevent close to zero weights trapped
    within the region that quantized into zero, which will never be
    updated by back-propagation, thus degrades the accuracy. 
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


X_RELU = _newrelu.apply
