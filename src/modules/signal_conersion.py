# This script includes the functions that realize signal conversions, 
# which includes DAC()

# Written by: Zhezhi (Elliot) He


# This function is for the input quantization and weight quantziation
class _quan_input(torch.autograd.Function):
    '''
    T
    '''
    @staticmethod
    def forward(ctx, input, delta_x):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.delta_x = delta_x
        output = torch.round(input/delta_x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.delta_x, None
        return grad_input


class _quan_weight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, delta_w):
        ctx.delta_w = delta_w
        output = torch.round(input/delta_w)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.delta_w, None
        return grad_input


# This function is for the output quantization (i.e., ADC)
class _adc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, delta_i, delta_y):
        ctx.delta_i = delta_i
        ctx.delta_y = delta_y
        output = torch.round(input/ctx.delta_i)*ctx.delta_y
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.delta_y*grad_output.clone()/ctx.delta_i, None, None
        return grad_input


class _newrelu(torch.autograd.Function):

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


quantize_input = _quan_input.apply
quantize_weight = _quan_weight.apply
adc = _adc.apply
x_relu = _newrelu.apply