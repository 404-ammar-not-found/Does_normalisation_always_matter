from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image, ImageFile

import zipfile
import os

transform = transforms.Compose([
    transforms.Resize(255), 
    transforms.CenterCrop(224),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_dataset(image_dir :str = "data/dataset", 
                     transform: transforms.Compose = transform) -> ImageFolder:
    
    dataset = ImageFolder(root=image_dir, 
                          transform=transform)
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset()
    print(f"Dataset size: {len(dataset)} images")