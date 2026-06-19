import argparse
import torch
import os

from dataset import FLAIRDataset
from torch.utils.data import DataLoader

torch.manual_seed(111)

def main(args):
    make_dirs(args)

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