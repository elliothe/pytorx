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
# pylint: disable=import-self, invalid-name, unused-argument
'''
Pytorx module testcase
================================
functionality tests
'''
from __future__ import print_function
import torch
from torx.module import *

# generic run functions for pytorch and pytorx


#######################################################################
# Digital Analog Converter (DAC)
# -------
def test_threshold_update():
    '''
    check the threshold is updated by the input.
    '''
    dac_test = DAC()
    pre_th = dac_test.threshold.item()  # init threshold
    test_input = torch.rand(10)  # test input
    dac_test.update_threshold(test_input)
    post_th = dac_test.threshold.item()
    # ensure threshold is update by the call of update threshold
    assert post_th != pre_th


def test_output_voltage_range():
    '''
    ensure the output voltage of DAC is between the range of
    Vdd and Vss.
    '''
    dac_test = DAC()
    test_input = torch.rand(10)
    dac_test.update_threshold(test_input)
    assert dac_test(test_input).max() < dac_test.vdd
    assert dac_test(test_input).min() > dac_test.vss


#######################################################################
# Stuck-at-Fault (SAF)
# -------
def test_saf_update_profile():
    ''' update SAF profile. '''
    g_shape = torch.Size([16, 3, 3, 3])
    saf_module = SAF(g_shape)
    pre_index_sa0 = saf_module.index_sa0()
    saf_module.update_saf_profile(dist='uniform')
    post_index_sa0 = saf_module.index_sa0()
    # print((pre_index_SA0-post_index_SA0).sum())
    assert (pre_index_sa0 -
            post_index_sa0).sum().item() != 0, 'SAF profile is not updated!'
    # print(saf_module.index_SA0())


def test_sa0_sa1_overlap():
    '''
    ensure there is no SAF state overlap between SA0 and SA1
    '''
    G_shape = torch.Size([3, 1, 3, 3])
    saf_module = SAF(G_shape)
    index_SA0 = saf_module.index_sa0()
    index_SA1 = saf_module.index_sa1()
    assert (index_SA0 * index_SA1
            ).sum().item() == 0, 'exist element is 1 for both SA0/1 index!'


