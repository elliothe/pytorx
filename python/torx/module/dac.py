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
from torch.nn import functional as F


class QuantizeDac(torch.autograd.Function):
    r'''customized quantization function of DAC for scaled gradient.
    Here the gradient is scaled to be consistent with the forward path'''

    @staticmethod
    def forward(ctx, input, delta_x):
        # ctx is a context object that can be used to stash information for backward computation
        ctx.delta_x = delta_x
        output = torch.round(input / ctx.delta_x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.delta_x
        return grad_input, None


# quantization function of DAC module
QUANTIZE_DAC = QuantizeDac.apply


class DAC(nn.Module):
    '''Digital-Analog Converter (DAC) module'''

    def __init__(self, nbits=8, Vdd=3.3, Vss=0, quan_method='dynamic'):
        super(DAC, self).__init__()
        r"""
        This Digital-Analog Converter (DAC) module includes two functions:
        1) quantize the floating-point input (FP32) to fixed-point integer;
        2) the fixed-point integer will be converted into voltages.
        After two functions performed, it will feed the voltage output to crossbar array, and 
        reserve all the other information (e.g., voltages). Note that, as discussed in our paper,
        the offset voltage where Vref = Vdd/2 is emitted due to the computation equivalency.

        Args:
            nbits (int): number of bits (resolution) of DAC.
            Vdd (fp): Vdd voltage (* in unit of volt).
            Vss (fp): Vss voltage (* in unit of volt).
            quan_method (str): quantization method, currently use the 'dynamic' method
                for ensure the correctness of the code
        """
        self.nbits = nbits
        self.vdd = Vdd
        self.vss = Vss
        self.quan_method = quan_method

        # DAC configuration
        self.full_lvls = 2**self.nbits - 1  # symmetric representation
        self.half_lvls = (self.full_lvls - 1) / 2  # number of lvls

        # input quantization
        # quantization threshold, which will be reinitialized
        self.threshold = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        # quantization resolution, which will be reinitialized
        self.delta_x = self.threshold.item() / self.half_lvls
        self.delta_v = (self.vdd - self.vss) / \
            (self.full_lvls - 1)  # DAC resolution voltage
        self.counter = 0
        self.acc = 0  # accumulator

        self.training = True  # flag to determine the operation mode

    def forward(self, input):
        r'''
        This function performs the conversion. Note that, output tensor (voltage) is in the
        same shape as the input tensor (FP32). The input reshape operation is completed by
        other module.
        '''

        # step 1: quantize the floating-point input (FP32) to fixed-point integer.
        # update the threshold before clipping
        # TODO: change the threshold tuning into KL_div calibration method
        self.update_threshold(input)
        input_clip = F.hardtanh(input,
                                min_val=-self.threshold.item(),
                                max_val=self.threshold.item())  # clip input

        self.delta_x = self.threshold.item() / self.half_lvls

        input_quan = QUANTIZE_DAC(input_clip, self.delta_x)

        # step 2: convert to voltage, here the offset (reference) voltage
        #  is emitted for simplicity.
        output_voltage = input_quan * self.delta_v

        return output_voltage

    def update_threshold(self, input):
        r'''update the clipping threshold, which is equivalent to
        update the DAC quantizer'''

        if 'dynamic' in self.quan_method:
            self.threshold.data = input.abs().max()
        else:
            # quantizer threshold
            with torch.no_grad():
                if self.training:
                    self.counter += 1
                    self.threshold.data = input.abs().max()
                    self.acc += self.threshold.data
                else:
                    # In evaluation mode, fixed the threshold
                    self.threshold.data[0] = self.acc / self.counter

