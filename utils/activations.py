import torch
from torch import nn
import torch.nn.functional as F

class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class HardSwish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.

class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)

class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()



# def forward(ctx, input): nhận các tensor inputs, và trả về tensor output.
#   Biến ctx để lưu lại các tensor cần thiết trong quá trình backward (chain rule).
# def backward(ctx, grad_output): grad_output chứa đạo hàm của loss đến tensor ở node đấy,
#   ctx lấy các giá trị lưu ở hàm forward để tính đạo hàm ngược qua node đó.
class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)

class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))