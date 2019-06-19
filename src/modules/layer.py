import torch
import torch.nn as nn


class crxb_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, crxb_size=64,
                 quantize=8, enable_ec_SAF=False):
        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                           stride, padding, dilation, groups, bias)
        
        assert self.groups==1, "currently not support grouped convolution for custom conv"
        
        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        self.enable_ec_SAF = enable_ec_SAF
        
        self.nchout_index = nn.Parameter(torch.arange(self.out_channels), requires_grad = False)
        weight_flatten = self.weight.view(self.out_channels, -1)
        self.crxb_row, self.crxb_row_pads = self.num_pad(weight_flatten.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(weight_flatten.shape[0], self.crxb_size)
        self.h_out = None
        self.w_out = None
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)
        weight_padded = F.pad(weight_flatten, self.w_pad, mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1,2)
        
        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl = 2**8
        self.h_lvl = (self.n_lvl-2)/2                        
        # ReRAM cells
        self.Gmax = 1/3000 # max conductance
        self.Gmin = 1/3e6 # min conductance
        self.delta_g = (self.Gmax-self.Gmin)/(2**7) # conductance step
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax, 
                    G_SA1=self.Gmin, weight_shape=weight_crxb.shape)
        
        # DAC
        self.Vdd = 3.3 # unit: volt
        self.delta_v = self.Vdd/(self.n_lvl-1)
        self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.counter = nn.Parameter(torch.Tensor(1), requires_grad = False)
        
#         self.max_i_LSB = ((self.Vdd/2)*self.Gmax*self.crxb_size)/self.h_lvl
        
    def num_pad(self, source, target):
        crxb_index = math.ceil(source/target)
        num_padding = crxb_index * target - source    
        return crxb_index, num_padding
    
    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max()/self.h_lvl
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max()/self.h_lvl
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data/self.counter.data
        
        input_clip = F.hardtanh(input, min_val=-self.h_lvl*self.delta_x.item(),
                                    max_val=self.h_lvl*self.delta_x.item())    
        input_quan = quantize_input(input_clip, self.delta_x)*self.delta_v # convert to voltage
        
        weight_quan = quantize_weight(self.weight, self.delta_w)

        # 2. Perform the computation between input voltage and weight conductance     
        if self.h_out is None and self.w_out is None:    
            self.h_out = int((input.shape[2]-self.kernel_size[0]+2*self.padding[0])/self.stride[0] + 1)
            self.w_out = int((input.shape[3]-self.kernel_size[0]+2*self.padding[0])/self.stride[0] + 1)    
            
        # 2.1 flatten and unfold the weight and input
        input_unfold = F.unfold(input_quan, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding, 
                                stride=self.stride)
        weight_flatten = weight_quan.view(self.out_channels, -1)
        
        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten, self.w_pad, mode='constant', value=0)
        input_padded = F.pad(input_unfold, self.input_pad, mode='constant', value=0)
        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1,2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight_crxb)
        
        # 2.4. compute matrix multiplication followed by reshapes

        if ir_drop:
            from IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col, self.crxb_row, self.crxb_size,
                                                        input.shape[0],
                                                        input_padded.shape[2])

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - \
                      torch.matmul(G_crxb[1], input_crxb)
        

        # perform ADC operation (i.e., current to digital conversion)
        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max()/(self.h_lvl)
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data/self.counter.data
            self.delta_y = self.delta_w*self.delta_x*self.delta_i/(self.delta_v*self.delta_g)
#         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl*self.delta_i.item(),
                                max_val=self.h_lvl*self.delta_i.item())  
        09/10 = adc(output_clip, self.delta_i, self.delta_y)
        
        if self.w2g.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y/self.delta_i
                output_adc += (torch.matmul(G_pos_diff, input_crxb) \
                           - torch.matmul(G_neg_diff, input_crxb))*ec_scale
        
        output_sum = torch.sum(output_adc, dim=2)
        output = output_sum.view(output_sum.shape[0],
                                 output_sum.shape[1]*output_sum.shape[2],
                                 self.h_out,
                                 self.w_out).index_select(dim=1,index=self.nchout_index)
        
        if self.bias is not None:
            output += self.bias.unsqueeze(1).unsqueeze(1)
        
        return output
    
    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0



class crxb_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, crxb_size=64,
                quantize=8, enable_ec_SAF=False):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)
        
        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        self.enable_ec_SAF = enable_ec_SAF
        
        self.out_index = nn.Parameter(torch.arange(out_features), requires_grad = False)
        self.crxb_row, self.crxb_row_pads = self.num_pad(self.weight.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(self.weight.shape[0], self.crxb_size)
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, self.crxb_row_pads)
        weight_padded = F.pad(self.weight, self.w_pad, mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1,2)
        
        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl = 2**8
        self.h_lvl = (self.n_lvl-2)/2                        
        # ReRAM cells
        self.Gmax = 1/3000 # max conductance
        self.Gmin = 1/3e6 # min conductance
        self.delta_g = (self.Gmax-self.Gmin)/(2**7) # conductance step
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax, 
                    G_SA1=self.Gmin, weight_shape=weight_crxb.shape)
        
        # DAC
        self.Vdd = 3.3 # unit: volt
        self.delta_v = self.Vdd/(self.n_lvl-1)
        self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.counter = nn.Parameter(torch.Tensor(1), requires_grad = False)
        
#         self.max_i_LSB = ((self.Vdd/2)*self.Gmax*self.crxb_size)/self.h_lvl                
        
    def num_pad(self, source, target):
        crxb_index = math.ceil(source/target)
        num_padding = crxb_index * target - source    
        return crxb_index, num_padding
    
    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max()/self.h_lvl
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max()/self.h_lvl
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data/self.counter.data

        input_clip = F.hardtanh(input, min_val=-self.h_lvl*self.delta_x.item(),
                                    max_val=self.h_lvl*self.delta_x.item())    
        input_quan = quantize_input(input_clip, self.delta_x)*self.delta_v # convert to voltage
        
        weight_quan = quantize_weight(self.weight, self.delta_w)

        # 2. Perform the computation between input voltage and weight conductance   
        # 2.1. skip the input unfold and weight flatten for fully-connected layers
        # 2.2. add padding
        weight_padded = F.pad(weight_quan, self.w_pad, mode='constant', value=0)
        input_padded = F.pad(input_quan, self.input_pad, mode='constant', value=0)
        # 2.3. reshape
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row, self.crxb_size, 1)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1,2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight_crxb)
        
        # 2.4. compute matrix multiplication

        if ir_drop:
            from IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col,
                                                        self.crxb_row,
                                                        self.crxb_size,
                                                        input.shape[0],
                                                        1)

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            output_crxb = torch.matmul(G_crxb[0], input_crxb) \
                    - torch.matmul(G_crxb[1], input_crxb)

        # perform ADC operation (i.e., current to digital conversion)
        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max()/(self.h_lvl)
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data/self.counter.data
            self.delta_y = self.delta_w*self.delta_x*self.delta_i/(self.delta_v*self.delta_g)
#         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl*self.delta_i.item(),
                                max_val=self.h_lvl*self.delta_i.item())  
        output_adc = adc(output_clip, self.delta_i, self.delta_y)
        
        if self.w2g.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y/self.delta_i
                output_adc += (torch.matmul(G_pos_diff, input_crxb) \
                           - torch.matmul(G_neg_diff, input_crxb))*ec_scale
                
        output_sum = torch.sum(output_adc, dim=2).squeeze(dim=3)
        output = output_sum.view(input.shape[0],
                                 output_sum.shape[1]*output_sum.shape[2]).index_select(dim=1, index=self.out_index)

        
        if self.bias is not None:
            output += self.bias
            
        return output
    
    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0