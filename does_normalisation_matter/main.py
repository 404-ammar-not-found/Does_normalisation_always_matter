from generate_dataset import generate_dataset, generate_train_val_test_datasets
from dataset_filteration import filter_small_classes
from generate_dataloader import generate_dataloader

dataset = generate_dataset()
print("Before:", (dataset.classes), "classes")

filtered_dataset = filter_small_classes(dataset, min_samples=50)
print("After:", (filtered_dataset.classes), "classes")

train_ds, val_ds, test_ds = generate_train_val_test_datasets(filtered_dataset,
                                                             train_ratio=0.7,
                                                             val_ratio=0.15,
                                                             test_ratio=0.15,
                                                             random_seed=42)


train_dataloader, validation_dataloader, test_dataloader = generate_dataloader(train_ds, val_ds, test_ds)

