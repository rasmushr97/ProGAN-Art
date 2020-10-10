import torch
import torch.nn as nn
import torch.nn.functional as F
import layers as ll
import torch.optim as optim

def upscale(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

def downscale(x):
    return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

class Discriminator(nn.Module):
    def __init__(self, features, channels, lr, betas, eps, device='cpu'):
        super().__init__()
        self.step = 0
        self.alpha = 1.0
        self.features = features
        self.channels = channels
        self.device = device
        self.lr, self.betas, self.eps = lr, betas, eps

        in_features = features * 2 ** 5
        self.from_rgb = ll.Conv2d(channels, in_features, 1, stride=1, padding=0)
        self.prev_from_rgb = None

        self.blocks = nn.ModuleList()
        self.final_block = nn.Sequential(
            ll.ConvBlock(in_features, in_features, 3, stride=1, padding=1),
            ll.ConvBlock(in_features, in_features, 3, stride=1, padding=1),
            ll.MinibatchStddev(),
            ll.ConvBlock(in_features+1, in_features, 3, stride=1, padding=1),
            ll.ConvBlock(in_features, in_features, 4, stride=1, padding=0),
            ll.Conv2d(in_features, 1, 1, stride=1, padding=0),
            nn.Flatten(),
        )

        self.optim = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)
        self.to(self.device)

    def forward(self, x):
        out = self.from_rgb(x)
        
        for index, block in enumerate(self.blocks):
            out = block(out)
            out = downscale(out)

            if index == 0 and 0 <= self.alpha < 1.0:
                skip_rgb = downscale(x)
                skip_rgb = self.prev_from_rgb(skip_rgb)
                out = (1 - self.alpha) * skip_rgb + self.alpha * out
        
        out = self.final_block(out)

        return out

    def add_scale(self):
        self.step += 1

        in_features = self.features * 2 ** (8 - max(3, self.step + 1))
        out_features = self.features * 2 ** (8 - max(3, self.step)) 

        self.prev_from_rgb = self.from_rgb
        self.from_rgb = ll.Conv2d(self.channels, in_features, 1, stride=1, padding=0)
        
        new_block = nn.Sequential(
            ll.ConvBlock(in_features, in_features, 3, stride=1, padding=1),
            ll.ConvBlock(in_features, out_features, 3, stride=1, padding=1),
        )
        self.blocks.insert(0, new_block)

        self.optim = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)

        self.to(self.device)

    def set_alpha(self, alpha):
        self.alpha = alpha



class Generator(nn.Module):

    def __init__(self, zdim, features, channels, lr, betas, eps, device='cpu'):
        super().__init__()
        self.step = 0
        self.alpha = 1.0
        self.device = device
        self.features = features
        self.channels = channels
        self.zdim = zdim
        self.lr, self.betas, self.eps = lr, betas, eps

        in_features = features * 2 ** (8 - max(3, self.step))
        out_features = features * 2 ** (8 - max(3, self.step + 1))        

        self.to_rgb = ll.Conv2d(out_features, self.channels, 1, stride=1, padding=0)
        self.prev_to_rgb = None
        self.blocks = nn.ModuleList()

        self.input_block = nn.Sequential(
            ll.ConvBlock(self.zdim, in_features, 4, stride=1, padding=3),
            ll.ConvBlock(in_features, in_features, 3, stride=1, padding=1),
            ll.ConvBlock(in_features, out_features, 3, stride=1, padding=1),
        )

        self.optim = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)
        self.to(self.device)

    def forward(self, x):
        out = self.input_block(x)

        for index, block in enumerate(self.blocks[:-1]):
            out = upscale(out)
            out = block(out)

        if self.prev_to_rgb is not None:
            skip_rgb = self.prev_to_rgb(out)
            skip_rgb = upscale(skip_rgb)

            out = upscale(out)
            out = self.blocks[-1](out)
            out = self.to_rgb(out)

            out = (1-self.alpha) * skip_rgb + self.alpha * out
        else:
            out = self.to_rgb(out)

        #out = torch.tanh(out)

        return out

    def add_scale(self):
        self.step += 1

        in_features = self.features * 2 ** (8 - max(3, self.step))
        out_features = self.features * 2 ** (8 - max(3, self.step + 1)) 

        self.prev_to_rgb = self.to_rgb
        self.to_rgb = ll.Conv2d(out_features, self.channels, 1, stride=1, padding=0)

        new_block = nn.Sequential(
            ll.ConvBlock(in_features, in_features, 3, stride=1, padding=1),
            ll.ConvBlock(in_features, out_features, 3, stride=1, padding=1),
        )
        
        self.blocks.append(new_block)

        self.optim = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)

        self.to(self.device)

    def set_alpha(self, alpha):
        self.alpha = alpha