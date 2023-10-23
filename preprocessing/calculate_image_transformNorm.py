import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import numpy as np
import torch
from PIL import Image

IMAGE_DIR = "../data/"

def open_gray_image(path):
    with open(path, 'rb') as f:
        with Image.open(f).convert('L') as img:
            return img


transform = transforms.Compose([
    transforms.Resize(64,),
    transforms.ToTensor(),
])


data = datasets.ImageFolder(
    IMAGE_DIR, loader=open_gray_image, transform=transform)

mean = torch.zeros(2)
std = torch.zeros(2)
count = [0, 0, 0]

for image, cls in data:
    if cls == 'images':
        mean[0] += image.mean()
        std[0] += image.std()
        count[0] += 1
    

mean /= count
std /= count
print(mean) # 0.9489
print(std) # 0.1565
