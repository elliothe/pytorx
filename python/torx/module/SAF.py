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


class SAF(nn.Module):
    '''Stuck-At-Fault Module'''

    def __init__(self, G_shape, p_SA0=0.1, p_SA1=0.1, G_SA0=1e6, G_SA1=1e3):
        super(SAF, self).__init__()
        r'''
        This module performs the Stuck-At-Fault (SAF) non-ideal effect injection.

        Args:
            G_shape (tensor.size): crossbar array size.
            p_SA0 (FP): Stuck-at-Fault rate at 0 (range from 0 to 1).
            p_SA1 (FP): Stuck-at-Fault rate at 1 (range from 0 to 1).
            G_SA0 (FP): Stuck-at-Fault conductance at 0 (in unit of S).
            G_SA1 (FP): Stuck-at-Fault conductance at 1 (in unit of S).
        '''

        # stuck at 0 leads to high conductance
        self.p_sa0 = nn.Parameter(torch.Tensor([p_SA0]),
                                  requires_grad=False)  # probability of SA0
        self.g_sa0 = G_SA0
        # stuck at 1 leads to low conductance
        self.p_sa1 = nn.Parameter(torch.Tensor([p_SA1]),
                                  requires_grad=False)  # probability of SA1
        self.g_sa1 = G_SA1

        assert (
            self.p_sa0 + self.p_sa1
        ) <= 1, 'The sum of SA0 and SA1 ratio should be less than 1 !!'

        # TODO: maybe change the SAF profile to uint8 format to avoid calculating the SAF
        # defect state on-the-fly, for simulation speedup. However the current setup has
        # higher configurability to simulate the real-time SAF state if there is run-time change.

        # initialize a random mask
        self.p_state = nn.Parameter(torch.Tensor(G_shape), requires_grad=False)
        self.update_saf_profile()  # init the SAF distribution profile

    def forward(self, input):
        '''
        The forward function alter the elements that indexed by p_state to the defected conductance,
        and mask the gradient of those defect cells owing to the auto-differentiation.
        '''
        output = INJECT_SAF(input, self.p_state, self.p_sa0, self.p_sa1,
                            self.g_sa0, self.g_sa1)
        return output

    def index_sa0(self):
        '''return the index of SA0'''
        return self.p_state.le(self.p_sa0)

    def index_sa1(self):
        '''return the index of SA1'''
        return self.p_state.gt(1 - self.p_sa1)

    def update_saf_profile(self, dist='uniform'):
        '''generate brand-new stuck-at-fault profile w.r.t various distribution'''
        if dist == 'uniform':
            self.p_state.data.uniform_()

        #TODO(zhezhi): add more stuck-at-fault distribution.

class _SAF(torch.autograd.Function):
    r'''
    This autograd function performs the gradient mask for the weight
    element with Stuck-at-Fault defects, where those weights will not
    be updated during backprop through gradient masking.

    Args:
        input (Tensor): weight tensor in FP32
        p_state (Tensor): probability tensor for indicating the SAF state
            w.r.t the preset SA0/1 rate (i.e., p_SA0 and p_SA1).
        p_SA0 (FP): Stuck-at-Fault rate at 0 (range from 0 to 1).
        p_SA1 (FP): Stuck-at-Fault rate at 1 (range from 0 to 1).
        G_SA0 (FP): Stuck-at-Fault conductance at 0 (in unit of S).
        G_SA1 (FP): Stuck-at-Fault conductance at 1 (in unit of S).
    '''

    @staticmethod
    def forward(ctx, input, p_state, p_sa0, p_sa1, g_sa0, g_sa1):
        ctx.save_for_backward(p_state, p_sa0, p_sa1)
        output = input.clone()
        # fixed the conductance based on profile.
        output[p_state.le(p_sa0)] = g_sa0  
        output[p_state.gt(1 - p_sa1)] = g_sa1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        p_state, p_sa0, p_sa1 = ctx.saved_tensors
        grad_input = grad_output.clone()
        # mask the gradient of defect cells
        grad_input[p_state.le(p_sa0) + p_state.gt(1 - p_sa1)] = 0
        return grad_input, None, None, None, None, None


INJECT_SAF = _SAF.apply
