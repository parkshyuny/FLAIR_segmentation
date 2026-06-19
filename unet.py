import torch

from torch import nn

class UNet(nn.Module):
    """
    U-Net model for image segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1, hidden_dim=64):
        super().__init__()

        self.down_pool = nn.MaxPool2d((2, 2), 2)
        self.down_convolution_1 = EncoderBlock(in_channels, hidden_dim)
        self.down_convolution_2 = EncoderBlock(hidden_dim, hidden_dim * 2)
        self.down_convolution_3 = EncoderBlock(hidden_dim * 2, hidden_dim * 4)
        self.down_convolution_4 = EncoderBlock(hidden_dim * 4, hidden_dim * 8)

        self.bottle_neck = EncoderBlock(hidden_dim * 8, hidden_dim * 16)

        self.up_convolution_1 = DecoderBlock(hidden_dim * 16, hidden_dim * 8)
        self.up_convolution_2 = DecoderBlock(hidden_dim * 8, hidden_dim * 4)
        self.up_convolution_3 = DecoderBlock(hidden_dim * 4, hidden_dim * 2)
        self.up_convolution_4 = DecoderBlock(hidden_dim * 2, hidden_dim)

        self.out = nn.Conv2d(hidden_dim, out_channels, (1, 1))

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.down_convolution_2(self.down_pool(down_1))
        down_3 = self.down_convolution_3(self.down_pool(down_2))
        down_4 = self.down_convolution_4(self.down_pool(down_3))
    
        bottle_neck = self.bottle_neck(self.down_pool(down_4))

        up_1 = self.up_convolution_1(bottle_neck, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        out = torch.sigmoid(out)
        
        return out
        
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.down_convolution(x)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up_convolution = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(True),
        )

    def forward(self, x, out_down):
        out_up = self.up_convolution(x)
        x = torch.cat([out_up, out_down], 1)
        return self.conv(x)


# from torchinfo import summary

# model = UNet()
# summary(model, input_size=(1, 3, 256, 256))