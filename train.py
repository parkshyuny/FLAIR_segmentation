import argparse
import torch
import os
import numpy as np

from tqdm import tqdm
from utils.loss import *
from utils.visualize import sample_images
from unet import UNet
from dataset import FLAIRDataset
from torch.utils.data import DataLoader
from torch import optim

torch.manual_seed(111)

def main(args):
    make_dirs(args)

    batch_size = args.batch_size
    images_dir = args.images_dir
    lr = args.lr

    loader_train, loader_valid = data_loaders(images_dir, batch_size)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet()

    loss_train = []
    loss_valid = []

    optimizer = optim.Adam(unet.parameters(), lr=lr)

    epochs = 1
    for epoch in range(epochs):
        for mode in ["train", "valid"]:
            if mode == "train":
                unet.train()
            else:
                unet.eval()
            
            for step, (X, y_true) in enumerate(tqdm(loaders[mode], total=len(loaders[mode]))):
                y_pred = unet(X)

                loss = bce_dice_score(y_true, y_pred)

                if mode == "train":
                    loss_train.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    loss_valid.append(loss.item())
                
                if step % 10 == 0 and mode == "valid":
                    print(f"Epoch {epoch} : Step {step + 1} : Validation loss {np.mean(loss_valid)}")
                    sample_images(
                        y_pred.detach().numpy(), 
                        y_true.detach().numpy(),
                        step
                    )
                    loss_valid = []

def data_loaders(images_dir, batch_size):
    dataset_train = FLAIRDataset(images_dir, "train")
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    dataset_valid = FLAIRDataset(images_dir, "validation")
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=True,
    )
    
    return loader_train, loader_valid

def make_dirs(args):
    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(args.weights, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of FLAIR images."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./images_dir",
        help="Folder with images and masks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target input image size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs",
        help="Folder to save logs"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./weights",
        help="Folder to save weights"
    )

    main(parser.parse_args())