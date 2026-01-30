from collections import Counter
from torchvision.datasets import ImageFolder


def filter_small_classes(dataset: ImageFolder, 
                         min_samples: int
                         ) -> ImageFolder:
    # Count samples per class index
    counts = Counter(dataset.targets)

    # Keep only classes with enough samples
    keep_class_indices = {
        cls_idx for cls_idx, c in counts.items() if c >= min_samples
    }

    # Filter samples
    new_samples = [
        (path, cls)
        for path, cls in dataset.samples
        if cls in keep_class_indices
    ]

    # Rebuild class mappings
    old_idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    new_classes = sorted({old_idx_to_class[i] for i in keep_class_indices})
    new_class_to_idx = {cls: i for i, cls in enumerate(new_classes)}

    # Remap samples + targets
    remapped_samples = [
        (path, new_class_to_idx[old_idx_to_class[cls]])
        for path, cls in new_samples
    ]

    dataset.samples = remapped_samples
    dataset.targets = [cls for _, cls in remapped_samples]
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx

    return dataset
