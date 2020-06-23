import torch
from voc_dataset.voc_segmentation import VocSegmentationUNet, make_dataloaders
from voc_dataset.voc_index import VocIndex
from unet.utils import visualize_prediction_target, get_accuracy
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
import torchvision


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch-train", type=int, default=1,
                        help="Size of batch for training")
    parser.add_argument("--batch-valid", type=int, default=1,
                        help="Size of batch for validation")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("-model", "--pretrained-model", type=str,
                        help="Absolute path (or relative to \"models_checkpoints\" folder) to pretrained model")
    parser.add_argument("-data", "--dataset-index", type=str,
                        help="Absolute path to dataset index file")
    parser.add_argument("--autosave-period", type=int, default=10,
                        help="Period for model's autosave in batches")
    parser.add_argument("--validation_share", type=float, default=0.1,
                        help="Share of data used in validation")
    parser.add_argument("--validation_period", type=int, default=10,
                        help="Period for model's validation in batches")
    parser.add_argument("--use_gpu", type=int, default=0,
                        help="Use GPU (CUDA) or not")
    args = parser.parse_args()
    return args


def train_unet(model, train_dataloader, val_dataloader, lr=1e-3, epoch_cnt=1, valid_period=10, asave_period=200,
               use_cuda=True, logger=print):

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    tboard_writer = SummaryWriter()
    prev_tstamp = time()
    global_step = 0

    for epoch in range(epoch_cnt):
        batches_per_epoch = len(train_dataloader)
        with tqdm(total=batches_per_epoch, desc=f'Epoch {epoch + 1}/{epoch_cnt}', unit='batch') as pbar:
            for idx, batch in enumerate(train_dataloader):
                model.train()

                input_images = batch["input"]
                target_outputs = batch["target"]
                weights = batch["weight"]

                prediction = model.forward(input_images)
                train_loss = compute_loss(prediction, target_outputs, weights, model.out_classes)
                train_accuracy = get_accuracy(prediction, target_outputs)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                if (idx + 1) % valid_period == 0:
                    model.eval()
                    val_batch = next(iter(val_dataloader))
                    val_images = val_batch["input"]
                    val_target = val_batch["target"]
                    val_weights = val_batch["weight"]

                    val_pred = model.forward(val_images)
                    val_loss = compute_loss(val_pred, val_target, val_weights, model.out_classes)
                    val_accuracy = get_accuracy(val_pred, val_target)
                    logger("Step", global_step, "Validation", epoch, "batch", idx, "Loss", val_loss.item())
                    tboard_writer.add_scalar("Loss/Val", val_loss.item(), global_step)
                    tboard_writer.add_scalar("Accuracy/Val", val_accuracy, global_step)

                    src_imgs, pred_imgs, target_imgs = visualize_prediction_target(val_images, val_pred, val_target,
                                                                                   to_tensors=True)
                    img_grid_pred = torchvision.utils.make_grid(pred_imgs)
                    img_grid_tgt = torchvision.utils.make_grid(target_imgs)
                    img_grid_src = torchvision.utils.make_grid(src_imgs)
                    tboard_writer.add_image('Valid/Predicted', img_tensor=img_grid_pred,
                                            global_step=global_step, dataformats='CHW')
                    tboard_writer.add_image('Valid/Target', img_tensor=img_grid_tgt,
                                            global_step=global_step, dataformats='CHW')
                    tboard_writer.add_image('Valid/Image', img_tensor=img_grid_src,
                                            global_step=global_step, dataformats='CHW')
                if (idx + 1) % asave_period == 0:
                    save_model(model, use_cuda=use_cuda, hint="e"+str(epoch)+"b"+str(idx))

                iteration_duration = time() - prev_tstamp
                prev_tstamp = time()

                logger("Step", global_step, "Epoch", epoch, "batch", idx, "Loss", train_loss.item(), "duration",
                       iteration_duration)

                tboard_writer.add_scalar("Loss/Train", train_loss.item(), global_step)
                tboard_writer.add_scalar("Accuracy/Train", train_accuracy, global_step)
                global_step += 1
                pbar.update(len(input_images))
    return


def main():
    train_logger = Logger(path="log.txt", hint="training", print_to_console=False)

    args = parse_cmd_args()
    model = UNet(3, 2)
    dataset = VocSegmentationUNet(args.dataset_index, ["person"])
    train_dataloader, val_dataloader = make_dataloaders(dataset, args.batch_train, args.batch_valid,
                                                        args.validation_share, True)
    use_cuda = torch.cuda.is_available() and int(args.use_gpu)

    try:
        train_unet(model, train_dataloader, val_dataloader, args.learning_rate, args.epochs, args.validation_period,
                   args.autosave_period, use_cuda=use_cuda, logger=train_logger)
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
