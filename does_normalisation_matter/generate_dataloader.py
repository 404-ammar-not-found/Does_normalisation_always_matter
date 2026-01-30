import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from config_parameters import SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_dataloader(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Create DataLoader objects for training, validation, and testing datasets.
    """
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader, test_dataloader
