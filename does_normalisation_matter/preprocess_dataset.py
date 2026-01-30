"""
Preprocess raw image dataset by resizing and center-cropping images
to a fixed resolution suitable for model training.
"""

import os
from PIL import Image


SRC = "../data/dataset"
DST = "../data/dataset_224"


def preprocess_dataset(src_dir: str, dst_dir: str) -> None:
    """
    Resize and center-crop all images from the source directory
    and save the processed images to the destination directory.
    """
    for cls in os.listdir(src_dir):
        os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)

        for img_name in os.listdir(os.path.join(src_dir, cls)):
            path = os.path.join(src_dir, cls, img_name)

            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((256, 256))
                img = img.crop((16, 16, 240, 240))
                img.save(
                    os.path.join(dst_dir, cls, img_name),
                    quality=90,
                )
            except Exception:
                pass


if __name__ == "__main__":
    preprocess_dataset(SRC, DST)
