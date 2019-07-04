import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .SAF import *


class crossbar(nn.Module):
    
    def __init__(self, weight, dilation, padding, stride, 
                 nbits=7, Gmax=1/3e3, Gmin=1/3e6, crxb_size=32):
        super(crossbar,self).__init__()
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
        self.half_lvl = 2**self.nbits - 1 # number of lvls determined by ReRAM cell
        self.delta_g = (self.Gmax - self.Gmin)/self.half_lvl
        
        # configurations for proper reshape, add padding to fit 
        weight_row_length = np.prod(self.weight_size[1:])
        self.crxb_row, self.crxb_row_pads = self.num_pad(weight_row_length,
                                                         self.crxb_size)
        weight_col_length = self.weight_size[0]
        self.crxb_col, self.crxb_col_pads = self.num_pad(weight_col_length,
                                                         self.crxb_size)
        
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)
              
        # compute the crossbar shape then define the weight-2-conductance module
        weight_flatten = weight.view(self.weight_size[0], -1)
        weight_padded = F.pad(weight_flatten, self.w_pad, mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1,2)
        
        # additional module to perform the conversion between fp32 weight to conductance
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax, 
                    G_SA1=self.Gmin, weight_shape=weight_crxb.shape)       
        
    def forward(self, input, weight):
        r'''
        Perform the computation between input voltage and weight.
        
        Args:
            input (fp, voltage): the un-reshaped input voltage tensor
            weight (fp): the un-reshape weight tensor
        
        '''           
        # 1. unfold the input
        input_unfold = F.unfold(input, kernel_size = self.weight_size[3],
                                dilation=self.dilation, padding=self.padding, 
                                stride=self.stride)
        
        # quantize and flatten the weight
        with torch.no_grad():
            self.delta_w = weight.abs().max()/self.half_lvl
        
        weight_quan = quantize_weight(weight, self.delta_w)
        weight_flatten = weight_quan.view(self.weight_size[0], -1) # self.weight_size[0] is number of output channels
        
        # 2. add paddings
        input_padded = F.pad(input_unfold, self.input_pad, mode='constant', value=0)
        weight_padded = F.pad(weight_flatten, self.w_pad, mode='constant', value=0)
        
        # 3. reshape both input and weight tensor w.r.t crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1,2)
        
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight_crxb) # G_crxb[0] and G_crxb[1] are postive and negative arrays respectively
        
        # 4. compute matrix multiplication followed by reshapes
        output = torch.matmul(G_crxb[0], input_crxb) - \
                      torch.matmul(G_crxb[1], input_crxb)
                
        return output
        
    def num_pad(self, source, target):
        crxb_index = math.ceil(source/target)
        num_padding = crxb_index * target - source    
        return crxb_index, num_padding


class _quantize_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delta_w):
        ctx.delta_w = delta_w
        output = torch.round(input/delta_w)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.delta_w
        return grad_input, None
    
# weight quantization function
quantize_weight = _quantize_weight.apply 


class w2g(nn.Module):
    '''
    perfrom the weight conversion within this function, which convert the 
    fixed point weight (weight_hat) into a pair of conductance values.
    
    output[0] is the G_pos and output[1] is the G_neg
    '''
    def __init__(self, delta_g, Gmin, G_SA0, G_SA1, weight_shape,
                 enable_rand=False, enable_SAF=False):
        super(w2g,self).__init__()
        self.delta_g = delta_g
        self.Gmin = Gmin
        self.G_SA0 = G_SA0
        self.G_SA1 = G_SA1
        self.p_SA0 = 0
        self.p_SA1 = 0
        self.enable_rand = enable_rand
        self.enable_SAF = enable_SAF
        self.SAF_pos = SAF(weight_shape, p_SA0=self.p_SA0, p_SA1=self.p_SA1, 
                        G_SA0=self.G_SA0, G_SA1=self.G_SA1, enable_rand=self.enable_rand)
        self.SAF_neg = SAF(weight_shape, p_SA0=self.p_SA0, p_SA1=self.p_SA1, 
                        G_SA0=self.G_SA0, G_SA1=self.G_SA1, enable_rand=self.enable_rand)
    
    def forward(self, input):
        self.G_pos = self.Gmin + x_relu(input)*self.delta_g
        self.G_neg = self.Gmin + F.relu(-input)*self.delta_g
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
        G_pos_diff = (self.G_pos-self.G_SA0)*pos_SA0 + (self.G_pos-self.G_SA1)*pos_SA1
        G_neg_diff = (self.G_neg-self.G_SA0)*neg_SA0 + (self.G_neg-self.G_SA1)*neg_SA1
        
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


