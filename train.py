"""
File: train.py
"""
import torch
import argparse
import wandb
from data import augment, generate_masks
from tqdm import tqdm
from models import *
from pathlib import Path
from datasets import load_dataset
import os

os.environ["LD_LIBRARY_PATH"] = ""

ds = load_dataset("keremberke/license-plate-object-detection", name="full")
print("Loaded dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(args):
    if args.model == "baseline":
        model = BaselineModel()
    elif args.model == "fullconv":
        model = FullConvolutionModel()
    else:
        raise ValueError("Model not supported")

    wandb.init(
        project="cs231n-final-project",
        config={
            "architecture": repr(model),
            **vars(args),
        },
    )

    # make the save path
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # move to device
    model = nn.DataParallel(model)
    model = model.to(device)

    def loss_fn(preds, target, reduction="mean", l=vars(args).get("l", 1.0)):
        loss = 0.0
        eps = 1e-6

        # normal loss on the background (0th channel)
        loss -= torch.sum(target[:, 0, :] * torch.log(preds[:, 0, :] + eps))

        # exaggerated loss on the license plate (1st channel)
        loss += torch.sum(
            target[:, 1, :] * torch.pow(l * torch.log(preds[:, 1, :] + eps), 2)
        )

        if reduction == "mean":
            loss /= target.numel()
        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for ep_idx in range(args.epochs):
        avg_train_loss = 0.0
        N_batches = 0
        for batch_idx in range(0, len(ds["train"]), args.batch_size):
            batch = ds["train"][batch_idx : batch_idx + args.batch_size]
            augmented_batch = augment(batch)
            masks = generate_masks(augmented_batch)

            # convert to torch tensors
            images = torch.stack([img for img, _ in augmented_batch])
            masks = torch.stack(masks)

            # move to device
            images = images.to(device)
            masks = masks.to(device)

            # forward pass
            preds = model(images)

            # compute loss
            loss = loss_fn(preds, masks)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            wandb.log({"loss": loss.item(), "epoch": ep_idx, "batch_idx": batch_idx})
            avg_train_loss += loss.item()
            N_batches += 1

        # step
        scheduler.step()
        avg_train_loss /= N_batches

        # save model
        if ep_idx % args.save_every == 0:
            torch.save(
                model.state_dict(), f"{args.save_path}/model-{args.model}-{ep_idx}.pt"
            )

            # evaluation
            loss = 0.0
            acc = 0.0
            acc_on_plate = 0.0
            plate_pixels = 0
            N_pixels = 0
            CE_denom = 0
            for test_bidx in tqdm(range(0, len(ds["test"]), args.batch_size)):
                batch = ds["test"][test_bidx : test_bidx + args.batch_size]
                augmented_batch = augment(batch)
                masks = generate_masks(augmented_batch)

                # convert to torch tensors
                images = torch.stack([img for img, _ in augmented_batch])
                masks = torch.stack(masks)

                # move to device
                images = images.to(device)
                masks = masks.to(device)

                with torch.no_grad():
                    # forward pass
                    preds = model(images)

                    # compute loss
                    loss += loss_fn(preds, masks)

                    # compute accuracy
                    batch_accuracy = (preds.round() == masks).sum().item()
                    acc += batch_accuracy
                    acc_on_plate += (
                        (preds.round()[:, 1, :] * masks[:, 1, :]).sum().item()
                    )

                    N_pixels += masks.numel()
                    CE_denom += 1
                    plate_pixels += masks[:, 1, :].sum().item()

            loss /= CE_denom
            acc /= N_pixels
            acc_on_plate /= plate_pixels
            wandb.log(
                {
                    "test_loss": loss.item(),
                    "epoch": ep_idx,
                    "batch_idx": batch_idx,
                    "accuracy": acc,
                    "accuracy_on_plate": acc_on_plate,
                }
            )

            print(
                f"[epoch {ep_idx}] train loss: {avg_train_loss:.4f} test loss: {loss.item():.4f} test accuracy: {acc:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to train for"
    )
    parser.add_argument("--batch-size", type=int, default=30, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--model", type=str, default="baseline", help="Model to train")
    parser.add_argument(
        "-l",
        "--lambda",
        type=float,
        default=2.0,
        help="Lambda for loss (how much to upweight the license plate)",
    )
    parser.add_argument(
        "--save-path", type=str, default="out", help="Path to save model"
    )
    parser.add_argument(
        "--save-every", type=int, default=1, help="Save model every n epochs"
    )
    args = parser.parse_args()

    train(args)
