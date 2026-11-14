from pathlib import Path
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR, SEED

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def make_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, val_fraction=0.15):
    """Create a stratified validation split from train; keep test untouched."""
    data_dir = Path(data_dir)
    train_aug = datasets.ImageFolder(data_dir / 'train', transform=train_tf)
    train_eval = datasets.ImageFolder(data_dir / 'train', transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / 'test', transform=eval_tf)

    if train_aug.classes != test_ds.classes:
        raise ValueError('Train and test class folders must match exactly.')
    if len(train_aug) < 4:
        raise ValueError('At least four training images are required for a train/validation split.')

    indices = np.arange(len(train_aug))
    labels = np.asarray(train_aug.targets)
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=SEED, stratify=labels
    )
    generator = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(Subset(train_aug, train_idx), batch_size=batch_size,
                          shuffle=True, generator=generator, num_workers=0)
    val_dl = DataLoader(Subset(train_eval, val_idx), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl, train_aug.classes
