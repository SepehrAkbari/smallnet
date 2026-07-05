'''
Dataset loaders for SPL experiments.
'''

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from src.dataset import CamVidDataset


VOC_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def load_camvid_class_names(class_dict_path):
    import csv

    names = []
    with open(class_dict_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names.append(row[0])
    return names


def imagenet_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def camvid_split_paths(data_root, split):
    data_root = Path(data_root)
    return data_root / split, data_root / f"{split}_labels"


def camvid_split_available(data_root, split):
    images_dir, masks_dir = camvid_split_paths(data_root, split)
    return images_dir.is_dir() and masks_dir.is_dir()


def make_camvid_loader(
    data_root,
    split,
    batch_size=4,
    num_workers=2,
    image_size=(352, 480),
    class_dict_path=None,
    shuffle=False,
):
    data_root = Path(data_root)
    images_dir, masks_dir = camvid_split_paths(data_root, split)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"CamVid image split is missing: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"CamVid label split is missing: {masks_dir}")
    class_dict_path = Path(class_dict_path) if class_dict_path else data_root / "class_dict.csv"
    dataset = CamVidDataset(
        images_dir,
        masks_dir,
        class_dict_path,
        transform=imagenet_transform(image_size),
        image_size=image_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class VOCSegmentationTensorDataset(Dataset):
    def __init__(self, root, year="2012", image_set="val", download=False, image_size=(520, 520)):
        from torchvision.datasets import VOCSegmentation

        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.image_size = tuple(image_size)
        self.image_transform = imagenet_transform(self.image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = self.image_transform(image)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long).reshape(self.image_size)
        return image, mask


def make_voc_loader(root, split="val", batch_size=4, num_workers=2, image_size=(520, 520), download=False):
    dataset = VOCSegmentationTensorDataset(
        root=root,
        image_set=split,
        download=download,
        image_size=image_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
