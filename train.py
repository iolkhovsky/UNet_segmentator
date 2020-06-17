import torch
from voc_dataset.voc_segmentation import VocSegmentationUNet, make_dataloaders
from voc_dataset.voc_index import VocIndex
from unet.unet_model import UNet
from unet.loss import compute_loss
from io_utils import *
from logger import Logger, LogDuration
import argparse
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import time


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1, metavar="epochs",
                        help="Number of epochs")
    parser.add_argument("-bt", "--batch-train", type=int, default=1, metavar="batch_train",
                        help="Size of batch for training")
    parser.add_argument("-bv", "--batch-valid", type=int, default=1, metavar="batch_val",
                        help="Size of batch for validation")
    parser.add_argument("--learning-rate", type=float, default=0.1, metavar="lr",
                        help="Learning rate")
    parser.add_argument("-model", "--pretrained-model", type=str, metavar="model_path",
                        help="Absolute path to pretrained model")
    parser.add_argument("-data", "--dataset-index", type=str, metavar="dataset_index",
                        help="Absolute path to dataset index file")
    parser.add_argument("-as", "--autosave-period", type=int, metavar="asave_period",
                        help="Period for model's autosave")
    parser.add_argument("--validation_share", type=float, default=0.1, metavar="val_share",
                        help="Period for model's autosave")
    parser.add_argument("--validation_period", type=int, default=10, metavar="val_period",
                        help="Period for model's autosave")
    args = parser.parse_args()
    return args


def train_unet(model, train_dataloader, val_dataloader, lr=1e-3, epoch_cnt=1, valid_period=10, logger=print):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)
    tboard_writer = SummaryWriter()
    prev_tstamp = time()
    iteration_duration = 0
    global_step = 0

    for epoch in range(epoch_cnt):
        epoch_train_loss = 0
        batches_per_epoch = len(train_dataloader)
        with tqdm(total=batches_per_epoch, desc=f'Epoch {epoch + 1}/{epoch_cnt}', unit='img') as pbar:
            for idx, batch in enumerate(train_dataloader):
                model.train()

                input_images = batch["input"]
                target_outputs = batch["target"]
                weights = batch["weight"]

                prediction = model.forward(input_images)
                train_loss = compute_loss(prediction, target_outputs, weights, model.out_classes)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.item()

                if idx + 1 % valid_period == 0:
                    model.eval()
                    val_batch = next(iter(val_dataloader))
                    val_images = val_batch["input"]
                    val_target = val_batch["target"]
                    val_weights = val_batch["weight"]
                    val_pred = model.forward(val_images)
                    val_loss = compute_loss(val_pred, val_target, val_weights, model.out_classes)
                    logger("Step", global_step, "Validation", epoch, "batch", idx, "Loss", val_loss.item())
                    tboard_writer.add_scalar("Loss/Val", val_loss.item(), global_step)

                iteration_duration = time() - prev_tstamp
                prev_tstamp = time()

                logger("Step", global_step, "Epoch", epoch, "batch", idx, "Loss", train_loss.item(), "duration",
                       iteration_duration)

                tboard_writer.add_scalar("Loss/Train", train_loss.item(), global_step)
                global_step += 1

        tboard_writer.add_scalar("Loss/Epoch", epoch_train_loss, global_step)
    return


def main():
    train_logger = Logger(path="training_log.txt", hint="training", print_to_console=False)

    args = parse_cmd_args()
    model = UNet(3, 2)
    dataset = VocSegmentationUNet(args.dataset_index, ["person"])
    train_dataloader, val_dataloader = make_dataloaders(dataset, args.batch_train, args.batch_valid,
                                                        args.validation_share, True)

    try:
        train_unet(model, train_dataloader, val_dataloader, args.learning_rate, args.epochs, args.validation_period,
                   logger=train_logger)
    except KeyboardInterrupt:
        save_model(model, "interrupted_train")
        train_logger("Training interrupted, model saved", caller="training script")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    return


if __name__ == "__main__":
    main()
