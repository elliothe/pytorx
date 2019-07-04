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
import torch.functional as F
import torch.nn as nn

class SAF(nn.Module):
    r'''
    This module performs the Stuck-At-Fault (SAF) non-ideal effect injection.
    In this module, it i
    '''
    def __init__(self, G_shape, p_SA0=0.1, p_SA1=0.1, G_SA0=1e6, G_SA1=1e3):
        super(SAF, self).__init__()
        # stuck at 0 leads to high conductance 
        self.p_SA0 = nn.Parameter(torch.Tensor([p_SA0]), requires_grad=False) # probability of SA0
        self.G_SA0 = G_SA0
        # stuck at 1 leads to low conductance
        self.p_SA1 = nn.Parameter(torch.Tensor([p_SA1]), requires_grad=False) # probability of SA1
        self.G_SA1 = G_SA1
        
        assert (self.p_SA0+self.p_SA1)<=1, 'The sum of probability of SA0 and SA1 is greater than 1 !!'
    
        # initialize a random mask
        self.p_state = nn.Parameter(torch.Tensor(G_shape), requires_grad = False)
        self.update_SAF_dist() # init the SAF distribution
         
    def forward(self, input):
        output = Inject_SAF(input, self.p_state, self.p_SA0, self.p_SA1, self.G_SA0, self.G_SA1)
        return output
    
    def index_SA0(self):
        return self.p_state.le(self.p_SA0)
        
    def index_SA1(self):
        return self.p_state.gt(1-self.p_SA1)

    def update_SAF_dist(self, dist):
        self.p_state.data.uniform_() #update the SAF distribution.
        return



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
    def forward(ctx, input, p_state, p_SA0, p_SA1, G_SA0, G_SA1):
        # p_state is the mask
        ctx.save_for_backward(p_state, p_SA0, p_SA1)
        output = input.clone()
        output[p_state.le(p_SA0)] = G_SA0
        output[p_state.gt(1-p_SA1)] = G_SA1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        p_state, p_SA0, p_SA1 = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[p_state.le(p_SA0) + p_state.gt(1-p_SA1)] = 0 #mask the gradient of defect cells
        return grad_input, None, None, None, None, None

Inject_SAF = _SAF.apply
