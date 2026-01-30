import pickle
from pathlib import Path
import numpy as np
import random


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchvision import models, transforms

from config_parameters import *
from dataset_filteration import filter_small_classes
from engine import train_model
from generate_dataloader import generate_dataloader
from generate_dataset import (
    generate_dataset,
    generate_train_val_test_datasets,
)


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_transforms(colour_normalised: bool) -> transforms.Compose:
    """
    Return image transform pipeline with or without colour normalisation.
    """
    transform_list = [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    if colour_normalised:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(transform_list)


def build_model(num_classes: int) -> nn.Module:
    """
    Create a frozen ResNet18 with a trainable classification head.
    """
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, num_classes),
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluate model on a dataloader and return loss, predictions, and labels.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x)
            loss = criterion(preds, y)

            total_loss += loss.item() * x.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    total_loss /= len(dataloader.dataset)

    return (
        total_loss,
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


def run_experiment(
    experiment_name: str,
    transform: transforms.Compose,
    output_dir: Path,
) -> dict:
    """
    Run a full training + evaluation pipeline for a given transform.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        image_dir=IMAGE_DIR,
        transform=transform,
    )

    filtered_dataset = filter_small_classes(
        dataset,
        min_samples=500,
    )

    class_names = filtered_dataset.classes

    train_ds, val_ds, test_ds = generate_train_val_test_datasets(
        filtered_dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=42,
    )

    train_loader, val_loader, test_loader = generate_dataloader(
        train_ds,
        val_ds,
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = build_model(num_classes=len(class_names))

    model, best_state_dict, results = train_model(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        verbose_ct=VERBOSE_CT,
        num_epochs=NUM_EPOCHS,
    )

    plt.figure()
    plt.plot(results["val_loss"], label=experiment_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(output_dir / "val_loss.png")
    plt.close()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    criterion = nn.CrossEntropyLoss()
    test_loss, preds, labels = evaluate_model(
        model,
        test_loader,
        criterion,
    )

    pred_labels = preds.argmax(dim=1)
    cm = confusion_matrix(
        labels.numpy(),
        pred_labels.numpy(),
        labels=range(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(ax=ax, xticks_rotation=90)
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    return {
        "val_loss": results["val_loss"],
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def compare_experiments(
    non_norm_results: dict,
    norm_results: dict,
    output_dir: Path,
) -> None:
    """
    Create comparison plots for validation loss and confusion matrices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(non_norm_results["val_loss"], label="Non-normalised")
    plt.plot(norm_results["val_loss"], label="Colour-normalised")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(output_dir / "val_loss_comparison.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(32, 14))

    ConfusionMatrixDisplay(
        non_norm_results["confusion_matrix"],
        display_labels=non_norm_results["class_names"],
    ).plot(ax=axes[0], xticks_rotation=90)
    axes[0].set_title("Non-normalised")

    ConfusionMatrixDisplay(
        norm_results["confusion_matrix"],
        display_labels=norm_results["class_names"],
    ).plot(ax=axes[1], xticks_rotation=90)
    axes[1].set_title("Colour-normalised")

    plt.savefig(output_dir / "confusion_matrix_comparison.png")
    plt.close()


if __name__ == "__main__":
    non_normalised_results = run_experiment(
        experiment_name="Non-normalised",
        transform=get_transforms(colour_normalised=False),
        output_dir=Path("../outputs/non_normalised"),
    )

    colour_normalised_results = run_experiment(
        experiment_name="Colour-normalised",
        transform=get_transforms(colour_normalised=True),
        output_dir=Path("../outputs/colour_normalised"),
    )

    compare_experiments(
        non_norm_results=non_normalised_results,
        norm_results=colour_normalised_results,
        output_dir=Path("../outputs/comparison"),
    )
