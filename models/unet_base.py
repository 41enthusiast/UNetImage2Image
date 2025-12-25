from torch import nn

from torchsummary import summary
import torch
import torch.nn.functional as F

from .model_utils import *
from torchvision import models


class ConvBlk(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.25)
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_blk(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlk(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = ConvBlk(in_channels, out_channels, in_channels // 2)
        # else:
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels//2),
            nn.LeakyReLU(0.25)
        )
        self.conv = ConvBlk(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)#concat for no info loss from add
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        block_scales = [64,128,256,512,1024]

        self.in_conv = nn.Conv2d(in_channels, block_scales[0], kernel_size = 3, padding=1, bias=False)

        self.downs = nn.ModuleList()
        for i,j in zip(block_scales[:-1],block_scales[1:]):
            self.downs.append(Down(i, j))
        
        self.ups = nn.ModuleList()
        block_scales = block_scales[::-1]
        for i,j in zip(block_scales[:-1],block_scales[1:]):
            self.ups.append(Up(i, j))

        self.outconv = nn.Conv2d(block_scales[-1], out_channels, kernel_size = 3, padding = 1, bias = False)

    def forward(self, x):
        outs = []
        num_blocks = len(self.downs)
        outs.append(self.in_conv(x))
        # outs.append(self.down1(outs[-1]))
        # outs.append(self.down2(outs[-1]))
        # outs.append(self.down3(outs[-1]))
        # outs.append(self.down4(outs[-1]))
        for down_lyr in self.downs:
            outs.append(down_lyr(outs[-1]))
        
        out =outs.pop()
        for up_lyr in self.ups:
            down_out = outs.pop()
            out = up_lyr(out, down_out)
        
        out = self.outconv(out)
        return out

    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_gpu()
    net = UNet(3,1).to(device)
    print(summary(net, input_size = (3, 256, 256)))
    