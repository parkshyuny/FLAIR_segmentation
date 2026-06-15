import argparse
import numpy as np
import torch
import torch.optim as optim
import os

from dataset import FLAIRDataset
from utils.loss import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger
from unet import UNet

torch.manual_seed(111)

def main(args):
    make_dirs(args)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet()

    dsc_loss = DiceLoss()
    loss_train = []
    loss_valid = []

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    step = 0

    logger = Logger(args.logs)
    
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for mode in ["train", "valid"]:
            if mode == "train":
                unet.train()
            else:
                unet.eval()
            
            for i, data in enumerate(loaders[mode]):
                optimizer.zero_grad()

                X, y_true = data
                y_pred = unet(X)

                loss = dsc_loss(y_true, y_pred)
                if mode == "train":
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    step += 1
                else:
                    loss_valid.append(loss.item())
            
                if (i + 1) % 10 == 0:
                    if mode == "train":
                        logger.scalar_summary(loss_train, step, "train_")
                        loss_train = []
                    else:
                        logger.scalar_summary(loss_valid, step, prefix="val_")
                        loss_valid = []

def data_loaders(args):
    dataset_train = FLAIRDataset(args.images_dir, "train")
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    dataset_valid = FLAIRDataset(args.images_dir, "validation")
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
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
        default="./dataset",
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
        default=100,
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