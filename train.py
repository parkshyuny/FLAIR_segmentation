from torchinfo import summary
from unet import UNet

model = UNet()
print(summary(model, input_size=(1, 3, 256, 256)))