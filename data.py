"""
File: data.py
"""
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors

IMG_SIZE = (640, 640)


def generate_mask(example):
    """
    Returns the 0/1 mask for where the license plate is with 1s on the license plate
    """
    v2
    pass


def augment(batch):
    """
    Augments the batch
    """
    boxes = [
        tv_tensors.BoundingBoxes(
            example["objects"]["bbox"],
            format="XYWH",
            canvas_size=(example["height"], example["width"]),
        )
        for example in batch
    ]

    images = [example["image"] for example in batch]

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=IMG_SIZE, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return [transforms(image, box) for image, box in zip(images, boxes)]
