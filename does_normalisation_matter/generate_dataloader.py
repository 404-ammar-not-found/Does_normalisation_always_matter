import torch
from torch.utils.data import DataLoader


def generate_dataloader(train_ds, val_ds, test_ds, BATCH_SIZE=32, num_workers=4):
    train_dataloader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


    test_dataloader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, validation_dataloader, test_dataloader
