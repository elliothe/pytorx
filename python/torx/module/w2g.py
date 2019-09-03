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

from .SAF import SAF


class w2g(nn.Module):
    '''
    perfrom the weight conversion within this function, which convert the 
    post-quantization fixed point weight (weight_hat) into a pair of
    conductance values. output[0] is the G_pos and output[1] is the G_neg
    '''

    def __init__(self, delta_g, Gmin, G_SA0, G_SA1, weight_shape,
                 enable_rand=True, enable_SAF=False):
        super(w2g, self).__init__()
        self.delta_g = delta_g
        self.Gmin = Gmin
        self.G_SA0 = G_SA0
        self.G_SA1 = G_SA1
        self.p_SA0 = 0.1
        self.p_SA1 = 0.1
        self.enable_rand = enable_rand
        self.enable_SAF = enable_SAF
        self.SAF_pos = SAF(weight_shape, p_SA0=self.p_SA0, p_SA1=self.p_SA1,
                           G_SA0=self.G_SA0, G_SA1=self.G_SA1)
        self.SAF_neg = SAF(weight_shape, p_SA0=self.p_SA0, p_SA1=self.p_SA1,
                           G_SA0=self.G_SA0, G_SA1=self.G_SA1)

    def forward(self, input):
        # x_relu() function is Critical
        self.G_pos = self.Gmin + x_relu(input) * self.delta_g
        self.G_neg = self.Gmin + F.relu(-input) * self.delta_g
        # the following two steps will update the SAF masking if enable_rand is True
        if self.enable_SAF:
            output = torch.cat((self.SAF_pos(self.G_pos).unsqueeze(0),
                                self.SAF_neg(self.G_neg).unsqueeze(0)),
                               0)
        else:
            output = torch.cat((self.G_pos.unsqueeze(0),
                                self.G_neg.unsqueeze(0)), 0)

        return output

    def error_compensation(self):
        pos_SA0 = self.SAF_pos.index_SA0().float().cuda()
        pos_SA1 = self.SAF_pos.index_SA1().float().cuda()
        neg_SA0 = self.SAF_neg.index_SA0().float().cuda()
        neg_SA1 = self.SAF_neg.index_SA1().float().cuda()
        G_pos_diff = (self.G_pos-self.G_SA0)*pos_SA0 + \
            (self.G_pos-self.G_SA1)*pos_SA1
        G_neg_diff = (self.G_neg-self.G_SA0)*neg_SA0 + \
            (self.G_neg-self.G_SA1)*neg_SA1

        return G_pos_diff, G_neg_diff

    def update_SAF(self, enable_SAF, p_SA0, p_SA1, new_SAF_mask=False, enable_rand=False):
        self.p_SA0 = p_SA0
        self.p_SA1 = p_SA1
        self.enable_SAF = enable_SAF
        # update the SAF_pos and SAF_neg modules
        self.SAF_pos.p_SA0.data[0] = self.p_SA0
        self.SAF_pos.p_SA1.data[0] = self.p_SA1
        self.SAF_neg.p_SA0.data[0] = self.p_SA0
        self.SAF_neg.p_SA1.data[0] = self.p_SA1
        # enable the random mask, thus each forward call get a new p_state mask
        self.enable_rand = enable_rand
        self.SAF_pos.enable_rand = enable_rand
        self.SAF_neg.enable_rand = enable_rand

        if new_SAF_mask:
            self.SAF_pos.p_state.data.uniform_()
            self.SAF_neg.p_state.data.uniform_()


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
    
x_relu = _newrelu.apply

############################################################
# Testbenchs
############################################################

def test_w2g_module_output_conductance_range():
    '''
    ensure the w2g module has the correct output conductance range
    which is between G_min and G_max.
    '''

    return