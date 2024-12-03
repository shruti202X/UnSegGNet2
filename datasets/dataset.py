import os
import deeplake
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio

class Dataset:
    def __init__(self, dataset):
        if dataset not in ["CUB", "ECSSD", "DUTS", "Clinical"]:
            raise ValueError(f'Dataset: {dataset} is not supported')
        self.dataset = dataset
        if dataset == "CUB":
            self.images, self.masks = load_cub()
        elif dataset == "ECSSD":
            ds = deeplake.load("hub://activeloop/ecssd")
            self.images = ds["images"]
            self.masks = ds["masks"]
        elif dataset == "DUTS":
            self.images, self.masks = load_duts()
        elif dataset == "Clinical":
            folder1 = "datasets/CVC-ClinicDB/Original"
            folder2 = "datasets/CVC-ClinicDB/Ground Truth"

            images = sorted(os.listdir(folder1))
            masks = sorted(os.listdir(folder2))

            images = [folder1 + '/' + x for x in images]
            masks = [folder2 + '/' + x for x in masks]

            self.images = images[1:-1]
            self.masks = masks[1:-1]
        self.loader = len(self.images)

    def load_samples(self):
        for imagep, true_maskp in zip(self.images, self.masks):
            try:
                if self.dataset == "CUB":
                    img = np.asarray(Image.open(imagep))
                    seg = np.asarray(Image.open(true_maskp).convert('L'))
                    true_mask = np.where(seg >= 200,1,0)
                elif self.dataset == "ECSSD":
                    img = np.asarray(imagep)
                    seg = np.asarray(true_maskp)
                    true_mask = np.where(seg == True, 1, 0)
                elif self.dataset == "DUTS":
                    img = np.asarray(Image.open(imagep))
                    seg = np.asarray(Image.open(true_maskp).convert('L'))
                    true_mask = np.where(seg == 255,1,0).astype(np.uint8)
                elif self.dataset == "Clinical":
                    img = imageio.imread(imagep)
                    seg = np.asarray(Image.open(true_maskp).convert('L'))
                    img, seg = cts(img, seg)
                    true_mask = np.where(seg == 255,1,0).astype(np.uint8)
                yield img, true_mask
            except Exception as e:
                print(e)
            finally:
                self.loader -= 1

           
def load_cub():
    cp = os.path.join(os.getcwd(), 'datasets')
    
    fold = f'{cp}/segmentations'
    file_paths = []
    for root, _, files in os.walk(fold):
        for file in files:
            file_paths.append(os.path.join(root,file))

    fold2 = f'{cp}/CUB_200_2011/images'
    fp2 = []
    for root, _, files in os.walk(fold2):
        for file in files:
            fp2.append(os.path.join(root,file))

    fp2  = sorted(fp2)
    file_paths = sorted(file_paths)

    with open(f'{cp}/CUB_200_2011/train_test_split.txt') as f:
        count = {}
        pretest = set()
        for line in f:
            x = line.split()[1]
            if x in count:
                count[x]+=1
            else:
                count[x] = 1
            if x == "0":
                pretest.add(line.split()[0])

    with open(f'{cp}/CUB_200_2011/images.txt') as u:
        test = []
        for line in u:
            x,y  = line.split()[0],line.split()[1]
            if x in pretest:
                test.append(y)

    masks = sorted([f'{cp}/segmentations/' + x[:len(x)-3] + 'png' for x in test])
    test = sorted([f'{cp}/CUB_200_2011/images/' + x for x in test])

    return test, masks


def load_duts():
    cp = os.path.join(os.getcwd(), 'datasets')

    fold = os.path.join(cp, 'DUTS-TE/DUTS-TE-Image')
    file_paths = []
    for root, _, files in os.walk(fold):
        for file in files:
            file_paths.append(os.path.join(root,file))

    fold2 = os.path.join(cp, 'DUTS-TE/DUTS-TE-Mask')
    fp2 = []
    for root, _, files in os.walk(fold2):
        for file in files:
            fp2.append(os.path.join(root,file))

    masks = sorted(fp2)
    test = sorted(file_paths)

    return test, masks

def cts(image, segmap,wnd = [20,20]):
    """
    Crop the image and segmentation map to the boundary of the segmentation.

    :param image: A numpy array representing the color image.
    :param segmap: A numpy array representing the binary segmentation map.
    :return: Cropped image and segmentation map.
    """
    [ht,wdt] = segmap.shape
    # Find the indices where segmap is 1
    rows, cols = np.where(segmap == 255)
    # Find min and max coordinates
    min_row, max_row = max(min(rows)-wnd[0],0), min(max(rows)+wnd[0],ht)
    min_col, max_col = max(min(cols)-wnd[1],0), min(max(cols)+wnd[1],wdt)

    # Crop the image and segmap
    cropped_image = image[min_row:max_row+1, min_col:max_col+1]
    cropped_segmap = segmap[min_row:max_row+1, min_col:max_col+1]

    return cropped_image, cropped_segmap