import os
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, Compose, InterpolationMode


class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        super().__init__()

        self.root = root
        self.transform = transform
        self.extensions = extensions

        self.samples = []
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(extensions):
                    self.samples.append((os.path.join(class_dir, file_name), class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError(f"Could not read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def transforms_testing(img_wh=224,
                       resize_wh=256,
                       interpolation=InterpolationMode.BILINEAR):
    return Compose([
        v2.ToPILImage(),
        v2.Resize((resize_wh, resize_wh), interpolation=interpolation, antialias=True),
        v2.CenterCrop((img_wh, img_wh)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

def transforms_training(img_wh=224,
                        interpolation=InterpolationMode.BILINEAR,
                        horizontal_flip_prob=0.5):
    return Compose([
        v2.ToPILImage(),
        v2.RandomResizedCrop((img_wh, img_wh), interpolation=interpolation, antialias=True),
        v2.RandomHorizontalFlip(horizontal_flip_prob) if horizontal_flip_prob > 0 else v2.Identity(),
        v2.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

