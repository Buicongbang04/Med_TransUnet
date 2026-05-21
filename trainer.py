import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def _get_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _dice_binary(pred, target, eps=1e-5):
    pred = (pred == 1).float()
    target = (target == 1).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union.item() == 0:
        return torch.tensor(1.0, device=pred.device)

    return (2.0 * intersection + eps) / (union + eps)


@torch.no_grad()
def validate(model, valloader, ce_loss, dice_loss, num_classes):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for sampled_batch in valloader:
        image_batch = sampled_batch["image"]
        label_batch = sampled_batch["label"]

        if len(image_batch.shape) == 3:
            image_batch = image_batch.unsqueeze(1)

        if len(image_batch.shape) == 2:
            image_batch = image_batch.unsqueeze(0).unsqueeze(0)

        image_batch = image_batch.float().cuda()
        label_batch = label_batch.long().cuda()

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch)
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        batch_dice = []
        for b in range(preds.shape[0]):
            batch_dice.append(_dice_binary(preds[b], label_batch[b]))

        total_loss += loss.item()
        total_dice += torch.stack(batch_dice).mean().item()
        n += 1

    model.train()

    return total_loss / max(n, 1), total_dice / max(n, 1)


def trainer_synapse(args, model, snapshot_path):
    """
    Trainer compatible with original TransUNet style, but adapted for current project:

    - epoch-based training
    - train + validation loop
    - CE + Dice loss
    - validation Dice monitoring
    - save best_model.pth
    - save periodic checkpoints
    - optional early stopping to reduce overfitting
    """

    os.makedirs(snapshot_path, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train.log") if hasattr(args, "output_dir") else os.path.join(snapshot_path, "train.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    if args.dataset == "LiTS":
        from datasets.dataset_lits import LiTS_dataset, RandomGenerator

        train_dataset = LiTS_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose([
                RandomGenerator(output_size=[args.img_size, args.img_size])
            ]),
        )

        val_dataset = LiTS_dataset(
            base_dir=args.root_path.replace("train_npz", "val_npz"),
            list_dir=args.list_dir,
            split="val",
            transform=None,
        )

    else:
        from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

        train_dataset = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose([
                RandomGenerator(output_size=[args.img_size, args.img_size])
            ]),
        )

        val_dataset = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="val",
            transform=None,
        )

    logging.info("The length of train set is: {}".format(len(train_dataset)))
    logging.info("The length of val set is: {}".format(len(val_dataset)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)

    val_interval = getattr(args, "val_interval", 1)
    save_interval = getattr(args, "save_interval", 10)
    early_stop_patience = getattr(args, "early_stop_patience", 20)
    min_delta = getattr(args, "min_delta", 1e-4)

    best_dice = 0.0
    best_epoch = -1
    epochs_without_improvement = 0

    logging.info("{} iterations per epoch. {} max iterations.".format(len(trainloader), max_iterations))
    logging.info("Validation interval: {} epoch(s)".format(val_interval))
    logging.info("Checkpoint interval: {} epoch(s)".format(save_interval))
    logging.info("Early stopping patience: {} epoch(s)".format(early_stop_patience))

    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice_loss = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda().long()

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_dice_loss += loss_dice.item()

            writer.add_scalar("train/lr", lr_, iter_num)
            writer.add_scalar("train/total_loss", loss.item(), iter_num)
            writer.add_scalar("train/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("train/loss_dice", loss_dice.item(), iter_num)

            if iter_num % 20 == 0:
                logging.info(
                    "epoch %d iteration %d : loss: %.5f, loss_ce: %.5f, loss_dice: %.5f"
                    % (epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item())
                )

                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)

                pred_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

                writer.add_image("train/Image", image, iter_num)
                writer.add_image("train/Prediction", pred_vis[0, ...] * 50, iter_num)
                writer.add_image("train/GroundTruth", label_batch[0, ...].unsqueeze(0) * 50, iter_num)

        avg_train_loss = epoch_loss / max(len(trainloader), 1)
        avg_train_ce = epoch_ce / max(len(trainloader), 1)
        avg_train_dice_loss = epoch_dice_loss / max(len(trainloader), 1)

        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch_num)
        writer.add_scalar("epoch/train_ce", avg_train_ce, epoch_num)
        writer.add_scalar("epoch/train_dice_loss", avg_train_dice_loss, epoch_num)

        logging.info(
            "Epoch %d finished. train_loss: %.5f, train_ce: %.5f, train_dice_loss: %.5f"
            % (epoch_num, avg_train_loss, avg_train_ce, avg_train_dice_loss)
        )

        if (epoch_num + 1) % val_interval == 0:
            val_loss, val_dice = validate(model, valloader, ce_loss, dice_loss, num_classes)

            writer.add_scalar("val/loss", val_loss, epoch_num)
            writer.add_scalar("val/dice_tumor", val_dice, epoch_num)

            logging.info(
                "Validation epoch %d : val_loss: %.5f, val_dice_tumor: %.5f"
                % (epoch_num, val_loss, val_dice)
            )

            if val_dice > best_dice + min_delta:
                best_dice = val_dice
                best_epoch = epoch_num
                epochs_without_improvement = 0

                best_path = os.path.join(snapshot_path, "best_model.pth")
                torch.save(_get_state_dict(model), best_path)

                logging.info(
                    "New best model saved to %s | epoch: %d | val_dice_tumor: %.5f"
                    % (best_path, epoch_num, best_dice)
                )
            else:
                epochs_without_improvement += 1
                logging.info(
                    "No improvement for %d epoch(s). Best epoch: %d | best val_dice_tumor: %.5f"
                    % (epochs_without_improvement, best_epoch, best_dice)
                )

            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                logging.info(
                    "Early stopping triggered at epoch %d. Best epoch: %d | best val_dice_tumor: %.5f"
                    % (epoch_num, best_epoch, best_dice)
                )
                break

        if (epoch_num + 1) % save_interval == 0:
            save_path = os.path.join(snapshot_path, "epoch_%d.pth" % epoch_num)
            torch.save(_get_state_dict(model), save_path)
            logging.info("Periodic checkpoint saved to {}".format(save_path))

        if epoch_num == max_epoch - 1:
            last_path = os.path.join(snapshot_path, "epoch_%d.pth" % epoch_num)
            torch.save(_get_state_dict(model), last_path)
            logging.info("Last checkpoint saved to {}".format(last_path))

    writer.close()

    logging.info("Training finished. Best epoch: %d | best val_dice_tumor: %.5f" % (best_epoch, best_dice))

    return "Training Finished!"