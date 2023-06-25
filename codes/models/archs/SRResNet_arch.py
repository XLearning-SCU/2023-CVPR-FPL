import math
import functools
import torch.nn as nn

from models.archs import arch_util

class _Residual_Block(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf)
        )
        
    def forward(self, x):
        return self.block(x) + x

class SRResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super().__init__()

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_nc, nf, 9, 1, 4),
            nn.PReLU()
        )

        self.residual = arch_util.make_layer(
            functools.partial(_Residual_Block, nf=nf), nb
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf)
        )

        assert upscale==4, 'only x4 upscale is implemented'
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv_output = nn.Conv2d(nf, out_nc, 9, 1, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out1 = self.conv_input(x)
        out = self.residual(out1)
        out = self.conv_mid(out) + out1
        out = self.upscale(out)
        out = self.conv_output(out)
        return out