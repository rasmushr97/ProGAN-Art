import torch
import torch.nn as nn
from math import sqrt

class Print(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x)
        return x

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input):
        return input * torch.rsqrt(
            (input**2).mean(dim=1, keepdim=True) + self.epsilon
        )

class MinibatchStddev(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MinibatchStddev, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        out_std = torch.sqrt(x.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(x.size(0), 1, 4, 4)
        return torch.cat([x, mean_std], 1)

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.normal_(0, 1)
        equal_lr(self)
        self.bias.data.fill_(0.)

class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.normal_(0, 1)
        equal_lr(self)
        self.bias.data.fill_(0.)

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight.data.normal_()
        self.bias.data.zero_()
        equal_lr(self)


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0, pixel_norm=True, lrelu=True):
        super().__init__()

        modules = []
        modules.append(Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding))
        if lrelu:
            modules.append(nn.LeakyReLU(0.2))
        if pixel_norm: 
            modules.append(PixelNorm())

        self.block = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.block(x)
