import torch
from voc_dataset.voc_segmentation import VocSegmentationUNet
from unet.unet_model import UNet
from io_utils import *
from logger import Logger, LogDuration
import argparse
import sys
import os
import tqdm


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1, metavar="epochs",
                        help="Number of epochs")
    parser.add_argument("-bt", "--batch-train", type=int, default=1, metavar="batch_train",
                        help="Size of batch for training")
    parser.add_argument("-bv", "--batch-valid", type=int, default=1, metavar="batch_val",
                        help="Size of batch for validation")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.1, metavar="lr",
                        help="Learning rate")
    parser.add_argument("-model", "--pretrained-model", type=str, default=0.1, metavar="model_path",
                        help="Absolute path to pretrained model")
    parser.add_argument("-tdata", "--train-dataset-index", type=str, metavar="dataset_index_train",
                        help="Absolute path to train dataset index file")
    parser.add_argument("-vdata", "--valid-dataset-index", type=str, metavar="dataset_index_val",
                        help="Absolute path to valid dataset index file")

    args = parser.parse_args()
    return args


def train_unet(model, epoch_cnt, train_loader):

    for epoch in range(epoch_cnt):
        model.train()

        epoch_loss = 0
        batches_per_epoch = 10

        with tqdm(total=batches_per_epoch, desc=f'Epoch {epoch + 1}/{epoch_cnt}', unit='img') as pbar:

            for batch in train_loader:
                print("do")
    return


def main():
    logger = Logger("training")

    args = parse_cmd_args()
    model = UNet(3, 2)
    train_dataset = VocSegmentationUNet(args.dataset_index_train, ["person"])
    val_dataset = VocSegmentationUNet(args.dataset_index_val, ["person"])

    train_unet(model)

    try:
        train_unet(model)
    except KeyboardInterrupt:
        save_model(model, "interrupted_train")
        logger("Training interrupted, model saved", caller="training script")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    return


if __name__ == "__main__":
    main()
