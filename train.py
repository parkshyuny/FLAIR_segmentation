import argparse
import torch
import torch.optim as optim

from utils.data_loading import PetDataset
from utils.loss import DiceLoss
from pathlib import Path
from torch.utils.data import DataLoader
from torchinfo import summary
from unet import UNet

torch.manual_seed(111)

def main(args):
    image_dir = Path("dataset/images")
    mask_dir = Path("dataset/masks")
    batch_size = args.batch_size
    image_size = args.image_size

    ds = PetDataset(image_dir, mask_dir, image_size)
    dataloader = DataLoader(ds, batch_size, shuffle=True)

    model = UNet()
    summary(model, input_size=(1, 3, 256, 256))

    optimizer = optim.Adam(model.parameters(), ls=args.lr)
    L = DiceLoss()

    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()

            x, y_true = data

            y_pred = model(x)
            loss = L(y_pred, y_true)
            loss.backward()
            optimizer.step()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of pet images."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="dataset/images",
        help="Directory for images",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="dataset/masks",
        help="Directory for target masks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target image size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.001)"
    )

    main(parser.parse_args())