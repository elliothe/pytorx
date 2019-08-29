import torch


class _adc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, delta_i, delta_y):
        ctx.delta_i = delta_i
        ctx.delta_y = delta_y
        output = torch.round(input / ctx.delta_i) * ctx.delta_y
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.delta_y * grad_output.clone() / ctx.delta_i, None, None
        return grad_input
