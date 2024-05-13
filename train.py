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

ds = load_dataset("keremberke/license-plate-object-detection", name="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # assuming image size is 640x640
        # predict probs for each pixel
        self.m = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(369664, 1024),
            nn.ReLU(),
            nn.Linear(1024, 640 * 640),
            nn.Sigmoid(),
            nn.Unflatten(1, (640, 640)),
        )

        torch.nn.init.xavier_uniform_(self.m[0].weight)

    def forward(self, x):
        return self.m(x)


class FullConvolutionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # assuming image size is 640x640
        # predict probs for each pixel
        self.m = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 512, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(1024, 2048, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2048, 2**12, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2**12, 2**13, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8192, 320 * 320),
            nn.ReLU(),
            nn.Linear(320 * 320, 640 * 640),
            nn.Sigmoid(),
            nn.Unflatten(1, (640, 640)),
        )

        torch.nn.init.xavier_uniform_(self.m[0].weight)

    def forward(self, x):
        return self.m(x)


def train(args):
    if args.model == "baseline":
        model = BaselineModel()
    elif args.model == "fullconv":
        model = FullConvolutionModel()
    else:
        raise ValueError("Model not supported")

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    test_loss = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep_idx in range(args.epochs):
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

        # save model
        if ep_idx % args.save_every == 0:
            torch.save(
                model.state_dict(), f"{args.save_path}/model-{args.model}-{ep_idx}.pt"
            )

            # evaluation
            train_loss = loss.item()
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
                    loss += test_loss(preds, masks)

                    # compute accuracy
                    batch_accuracy = (preds.round() == masks).sum().item()
                    acc += batch_accuracy
                    N_pixels += masks.numel()
                    CE_denom += masks.sum().item()

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
                f"[epoch {ep_idx}] train loss: {train_loss:.4f} test loss: {loss.item():.4f} test accuracy: {acc:.4f}"
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

    wandb.init(
        project="cs231n-final-project",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "model": args.model,
        },
    )

    train(args)
