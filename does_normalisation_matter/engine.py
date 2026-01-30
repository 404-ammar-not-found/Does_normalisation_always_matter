import numpy as np
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from config_parameters import SEED


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(
    model,
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    device: str,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    verbose_ct: int,
    num_epochs: int,
):
    """
    Train a classification model with early stopping and validation tracking.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.to(device)


    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None


    results = {
        "train_loss": [],
        "val_loss": [],
    }


    for epoch in tqdm(range(1, num_epochs + 1)):

        model.train()
        train_loss = 0.0

        for x, y in train_dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dataloader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in validation_dataloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(validation_dataloader.dataset)

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        if verbose_ct and epoch % verbose_ct == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state_dict = {
                k: v.clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose_ct:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    return model, best_state_dict, results
