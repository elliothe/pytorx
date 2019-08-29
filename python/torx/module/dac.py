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


class DAC(nn.Module):

    def __init__(self, nbits=8, Vdd=3.3, Vss=0, quan_method='dynamic'):
        super(DAC, self).__init__()
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
            quan_method (str): describe the quantization method, default method is
                               mainly used for functionality test.
        """
        self.nbits = nbits
        self.Vdd = Vdd
        self.Vss = Vss
        self.quan_method = quan_method

        # generate DAC configuration
        self.full_lvls = 2**self.nbits - 1  # symmetric representation
        self.half_lvls = (self.full_lvls-1)/2  # number of lvls (>=0 or <=0)

        # input quantization
        # quantization threshold, need to re-init
        self.threshold = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        # quantization resolution, need to re-init
        self.delta_x = self.threshold.item() / self.half_lvls
        self.delta_v = (self.Vdd - self.Vss) / (self.full_lvls - 1)  # DAC resolution voltage
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
        input_clip = F.hardtanh(input, min_val=-self.threshold.item(),
                                max_val=self.threshold.item())  # clip input

        self.delta_x = self.threshold.item()/self.half_lvls

        input_quan = quantize_dac(input_clip, self.delta_x)

        # step 2: convert to voltage, here the offset (reference) voltage term is emitted
        output_voltage = input_quan * self.delta_v

        return output_voltage

    def update_threshold(self, input):
        # for testing use the run-time maximum
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
                    self.threshold.data[0] = self.acc/self.counter

        return


############################################################
# Testbenchs
############################################################
# doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()


def test_threshold_update():
    '''
    check the threshold is updated by the input
    '''
    dac_test = DAC()
    pre_th = dac_test.threshold.item()  # init threshold
    test_input = torch.rand(10)  # test input
    dac_test.update_threshold(test_input)
    post_th = dac_test.threshold.item()
    # ensure threshold is update by the call of update threshold
    assert post_th != pre_th

    return


def test_output_voltage_range():
    '''
    ensure the output voltage of DAC is between the range of 
    Vdd and Vss.
    '''
    dac_test = DAC()
    test_input = torch.rand(10)
    assert dac_test(test_input).max() < dac_test.Vdd
    assert dac_test(test_input).min() > dac_test.Vss

    return
