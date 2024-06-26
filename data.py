"""
File: data.py
"""
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors

IMG_SIZE = (320, 320)


def generate_masks(augmented_batch):
    """
    Returns the 0/1 mask for where the license plate is with 1s on the license plate
    """
    masks = []
    for img, box in augmented_batch:
        mask = torch.zeros((2, *IMG_SIZE))
        mask[0, :, :] = 1
        for bbox in box:
            x, y, w, h = bbox
            a, b, c, d = x, y, x + w, y + h
            a, b, c, d = int(a), int(b), int(c), int(d)
            mask[0, b:d, a:c] = 0
            mask[1, b:d, a:c] = 1
        masks.append(mask)
    return masks


def augment(batch, test=False):
    """
    Augments the batch
    """
    boxes = [
        tv_tensors.BoundingBoxes(
            obj["bbox"],
            format="XYWH",
            canvas_size=(h, w),
        )
        for obj, h, w in zip(batch["objects"], batch["height"], batch["width"])
    ]

    images = batch["image"]
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop(size=IMG_SIZE, antialias=True),
    ]
    if not test:
        transforms += [
            v2.RandomPhotometricDistort(p=0.3),
            v2.RandomHorizontalFlip(p=0.3),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        ]
    transforms += [
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transforms = v2.Compose(transforms)
    return [transforms(image, box) for image, box in zip(images, boxes)]
