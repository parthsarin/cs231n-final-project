"""
File: train.py
"""
import torch
import torch.nn as nn
import argparse
import wandb
from datasets import load_dataset
from data import augment, generate_masks
from tqdm import tqdm
from models import *
import os

os.environ["LD_LIBRARY_PATH"] = ""
ds = load_dataset("keremberke/license-plate-object-detection", name="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "model": args.model,
            "architecture": repr(model),
        },
    )

    model = nn.DataParallel(model)
    model = model.to(device)

    # loss_fn = nn.CrossEntropyLoss()
    def loss_fn(preds, target, reduction="mean"):
        loss = 0.0

        # normal loss on the background (0th channel)
        loss -= torch.sum(target[:, 0, :] * torch.log(preds[:, 0, :]))

        # exaggerated loss on the license plate (1st channel)
        loss += torch.sum(
            target[:, 1, :] * torch.pow(-10 * torch.log(preds[:, 1, :]), 2)
        )

        if reduction == "mean":
            loss /= target.numel()
        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

        # save model
        if ep_idx % args.save_every == 0:
            torch.save(
                model.state_dict(), f"{args.save_path}/model-{args.model}-{ep_idx}.pt"
            )

            # evaluation
            loss = 0.0
            acc = 0.0
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
                    N_pixels += masks.numel()
                    CE_denom += 1

            loss /= CE_denom
            acc /= N_pixels
            wandb.log(
                {
                    "test_loss": loss.item(),
                    "epoch": ep_idx,
                    "batch_idx": batch_idx,
                    "accuracy": acc,
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
        "--save-path", type=str, default="out", help="Path to save model"
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save model every n epochs"
    )
    args = parser.parse_args()

    train(args)
