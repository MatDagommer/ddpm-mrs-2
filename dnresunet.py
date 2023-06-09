## Data visualization and plotting
import matplotlib.pyplot as plt

## neural networks
import torch
import torch.nn as nn

class DnResUNet(nn.Module):

    def __init__(self, name, n_channels, n_out, device, dilation):
        super(DnResUNet, self).__init__()
        self.name = name
        self.n_channels = n_channels
        self.n_out = n_out

        self.inputL = ConvBlock(n_channels, 32, dilation).to(device)
        self.down1 = Down(32, 64, dilation).to(device)
        self.down2 = Down(64, 128, dilation).to(device)
        self.down3 = Down(128, 256, dilation).to(device)
        self.down4 = Down(256, 512, dilation).to(device)
        self.down5 = Down(512, 1024, dilation).to(device)
        self.conv = ConvBlock(1024, 1024, dilation).to(device)
        self.up1 = Up(1024, 512, dilation).to(device)
        self.up2 = Up(512, 256, dilation).to(device)
        self.up3 = Up(256, 128, dilation).to(device)
        self.up4 = Up(128, 64, dilation).to(device)
        self.up5 = Up(64, 32, dilation).to(device)


        self.outputL = OutConv(32, n_out, dilation).to(device)
        
    def forward(self, x):

        x1 = self.inputL(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.down5(x5)
        b = self.conv(b)
        x = self.up1(b, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        x = self.outputL(x)
        
        return x
    

class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        KERNEL_SIZE = 3 
        P = (KERNEL_SIZE + 2*(dilation-1) - 1)//2
        self.convblock = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=P, padding_mode = 'replicate', dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
      x = self.convblock(x)
      return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        KERNEL_SIZE = 3
        P = (KERNEL_SIZE + 2*(dilation-1) - 1)//2
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=KERNEL_SIZE, padding=P, padding_mode = 'replicate', dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=KERNEL_SIZE, padding=P, padding_mode = 'replicate', dilation=dilation)
        )
        self.block2 = nn.Sequential(
          ConvBlock(in_channels, out_channels, dilation)
        )

    def forward(self, x):
        return self.block1(x) + self.block2(x)

class Up(nn.Module):
  def __init__(self, in_channels, out_channels, dilation):
     super().__init__()
     self.up = nn.Sequential(
         nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
         nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=dilation)
     )
     self.conv = ResBlock(out_channels * 2, out_channels, dilation)

  def forward(self, x1, x2):
      x1 = self.up(x1)
      x = torch.cat([x1, x2], dim=1)
      return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            ResBlock(in_channels, out_channels, dilation)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(OutConv, self).__init__()
        self.conv_tanh = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv_tanh(x)
    

