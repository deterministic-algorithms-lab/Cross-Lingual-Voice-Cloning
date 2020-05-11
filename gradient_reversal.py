import torch

class reverse_grad(torch.autograd.Function) :
    
    @staticmethod
    def forward(ctx, x) :
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output) :
        return - (grad_output.clamp(-0.5,0.5))

rg = reverse_grad.apply

def grad_reverse(x) :
    return rg(x)