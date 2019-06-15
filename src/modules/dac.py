import torch
import torch.nn as nn
import torch.nn.functional as F

class _quantize_dac(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, delta_x):
        # ctx is a context object that can be used to stash information for backward computation
        ctx.delta_x = delta_x
        output = torch.round(input/ctx.delta_x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.delta_x
        return grad_input, None

# quantization function of DAC module
quantize_dac = _quantize_dac.apply    


class dac_module(nn.Module):
    
    def __init__(self, nbits, Vdd=3.3, Vss=0):
        super(dac_module,self).__init__()
        r"""
        This Digital-Analog Converter (DAC) module includes two functions:
        1) quantize the floating-point input (FP32) to fixed-point integer;
        2) the fixed-point integer will be converted into voltages.
        After two functions performed, it will feed the voltage output to crossbar array, and reserve all the 
        other information (e.g., voltages). Note that, as discussed in our paper, the offset voltage where
        Vref = Vdd/2 is emitted due to the computation equivalency.

        Args:
            nbits (int): number of bits (resolution) of DAC.
            Vdd (fp): Vdd voltage (* in unit of volt).
            Vss (fp): Vss voltage (* in unit of volt).
        """
        self.nbits = nbits
        self.Vdd = Vdd
        self.Vss = Vss
        
        # generate DAC configuration
        self.full_lvls = 2**self.nbits - 1 # symmetric representation
        self.half_lvls = (self.full_lvls-1)/2 # number of lvls (>=0 or <=0)
        
        # input quantization
        self.threshold = nn.Parameter(torch.Tensor([1]), requires_grad = False) # quantization threshold, need to re-init
        self.delta_x = self.threshold.item()/self.half_lvls # quantization resolution, need to re-init 
        self.delta_v = (self.Vdd - self.Vss)/(self.full_lvls - 1) # DAC resolution voltage 
        self.counter = 0
        self.acc = 0 # accumulator 
        
        self.training = False # flag to determine the operation mode
    
    def forward(self, input):
        
        # step 1: quantize the floating-point input (FP32) to fixed-point integer.
        input_clip = F.hardtanh(input, 
                                min_val = -self.threshold.item(),
                                max_val = self.threshold.item()) # clip input 
        input_quan = quantize_dac(input_clip, self.delta_x)
        
        # step 2: convert to voltage, here the offset (reference voltage) is emitted
        output_voltage = input_quan * self.delta_v
        
        return output_voltage
    
    def update_quantizer(self, input):
        
        # quantizer threshold 
        with torch.no_grad():
            if self.training:
                self.counter += 1
                self.threshold.data = input.abs().max()
                self.delta_x = self.threshold.item()/self.half_lvls
                self.acc += self.threshold.data
            else:
                # In evaluation mode, fixed the 
                self.threshold.data[0] = self.acc/self.counter
                
        return 