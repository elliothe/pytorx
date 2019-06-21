import torch
import torch.functional as F
import torch.nn as nn
from dac import *

class _SAF(torch.autograd.Function):
    '''
    This autograd function performs the gradient mask for the weight
    element with Stuck-at-Fault defects, where those weights will not
    be updated during backprop through gradient masking.
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
        grad_input[p_state.le(p_SA0)+p_state.gt(1-p_SA1)] = 0 #mask the gradient of defect cells
        return grad_input, None, None, None, None, None


Inject_SAF = _SAF.apply



class SAF(nn.Module):
    
    def __init__(self, G_shape, p_SA0=0.1, p_SA1=0.1, G_SA0=1e6, G_SA1=1e3, enable_rand=False):
        super(SAF, self).__init__()
        # stuck at 0 leads to high conductance 
        self.p_SA0 = nn.Parameter(torch.Tensor([p_SA0]), requires_grad=False) # probability of SA0
        self.G_SA0 = G_SA0
        # stuck at 1 leads to low conductance
        self.p_SA1 = nn.Parameter(torch.Tensor([p_SA1]), requires_grad=False) # probability of SA1
        self.G_SA1 = G_SA1
        
        assert (self.p_SA0+self.p_SA1)<=1, 'The sum of probability of SA0 and SA1 is greater than 1 !!'
    
        # initialize a random mask
        self.p_state = nn.Parameter(torch.Tensor(G_shape).uniform_(), requires_grad = False)
        self.enable_rand = enable_rand
         
    def forward(self, input):
        if self.enable_rand:
            self.p_state.data.uniform_() # Generate random mask scheme for each batch
        
        output = Inject_SAF(input, self.p_state, self.p_SA0, self.p_SA1, self.G_SA0, self.G_SA1)
        
        return output
    
    def index_SA0(self):
        return self.p_state.le(self.p_SA0)
        
    def index_SA1(self):
        return self.p_state.gt(1-self.p_SA1)


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
        '''
        Note that, in this 
        '''
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