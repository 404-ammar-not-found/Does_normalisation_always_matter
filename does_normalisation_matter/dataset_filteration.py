from collections import Counter

from torchvision.datasets import ImageFolder


def filter_small_classes(
    dataset: ImageFolder,
    min_samples: int,
) -> ImageFolder:
    """
    Remove classes from an ImageFolder dataset that have fewer than a
    specified minimum number of samples.

    The dataset is modified in-place by filtering samples, rebuilding
    class indices, and remapping targets to ensure consistency.
    """
    # Count samples per class index
    counts = Counter(dataset.targets)

    # Identify class indices that meet the minimum sample requirement
    keep_class_indices = {
        class_idx
        for class_idx, count in counts.items()
        if count >= min_samples
    }

    # Filter samples belonging to retained classes
    filtered_samples = [
        (path, class_idx)
        for path, class_idx in dataset.samples
        if class_idx in keep_class_indices
    ]

    # Rebuild class name mappings
    old_idx_to_class = {
        idx: class_name
        for class_name, idx in dataset.class_to_idx.items()
    }

    new_classes = sorted(
        {old_idx_to_class[idx] for idx in keep_class_indices}
    )

    new_class_to_idx = {
        class_name: new_idx
        for new_idx, class_name in enumerate(new_classes)
    }

    # Remap samples and targets to new class indices
    remapped_samples = [
        (path, new_class_to_idx[old_idx_to_class[class_idx]])
        for path, class_idx in filtered_samples
    ]

    dataset.samples = remapped_samples
    dataset.targets = [class_idx for _, class_idx in remapped_samples]
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx

    return dataset
