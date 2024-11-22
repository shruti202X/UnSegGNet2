import os
import torch
from PIL import Image
os.environ['TORCH'] = torch.__version__

# print(torch.__version__) # 2.4.1+cu118

folder1 = "C:\\Users\\sb398\\Desktop\\project2\\UnSeGArmaNet\\datasets\\CVC-ClinicDB\\Original"
folder2 = "C:\\Users\\sb398\\Desktop\\project2\\UnSeGArmaNet\\datasets\\CVC-ClinicDB\\Ground Truth"

images = sorted(os.listdir(folder1))
masks = sorted(os.listdir(folder2))

images = [folder1 + '\\' + x for x in images]
masks = [folder2 + '\\' + x for x in masks]

images = images[1:-1]
masks = masks[1:-1]

# print(len(masks), len(images)) # 610 610

Image.open(masks[5])