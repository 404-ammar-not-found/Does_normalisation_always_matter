# preprocess.py
from PIL import Image
import os

SRC = "../data/dataset"
DST = "../data/dataset_224"

for cls in os.listdir(SRC):
    os.makedirs(os.path.join(DST, cls), exist_ok=True)
    for img_name in os.listdir(os.path.join(SRC, cls)):
        path = os.path.join(SRC, cls, img_name)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((256, 256))
            img = img.crop((16, 16, 240, 240))  # center crop 224
            img.save(os.path.join(DST, cls, img_name), quality=90)
        except:
            pass