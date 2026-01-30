
import numpy as np
import random

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from config_parameters import SEED



random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_dataset(
    image_dir: str,
    transform: transforms.Compose,
) -> ImageFolder:
    """
    Generate an ImageFolder dataset from a directory and apply transforms.
    """
    dataset = ImageFolder(
        root=image_dir,
        transform=transform,
    )
    return dataset


def generate_train_val_test_datasets(
    dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple:
    """
    Split a dataset into train, validation, and test subsets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    return train_ds, val_ds, test_ds
