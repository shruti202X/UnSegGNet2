import gc
import os
from datetime import datetime
from matplotlib import pyplot as plt
import torch
from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--bs", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--dataset", type=str, default='ECSSD')
parser.add_argument("--threshold", type=float, default=0.2)
parser.add_argument("--conv_type", type=str, default='GAT')
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--debug_samples", type=int, default=-1)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    print(device)

    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type)

    torch.cuda.empty_cache()
    gc.collect()

    ds = Dataset(args.dataset)

    filename = "_".join([args.conv_type, args.dataset, args.activation, args.loss_type, args.process, str(args.threshold), str(args.epochs)])
    img_folder = "../Img_" + filename
    if os.path.exists(img_folder):
        now = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
        img_folder += now
        filename += now
    log_file = f"log_{filename}.log"
    os.mkdir(img_folder)

    print("No. IoU mIoU")
    with open(log_file, "a") as f:
        f.write("No. IoU mIoU\n")

    total_iou = 0
    total_samples = 0
    while ds.loader > 0:
        for img, mask in ds.load_samples():
            try: 
                iou, segmentation, segmentation_over_image = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
                print(f"{total_samples} {iou:.2f} {(total_iou/total_samples):.2f}")
                if args.debug:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(segmentation_over_image)
                    ax[1].imshow(mask)
                    ax[2].imshow(segmentation)
                    ax[0].axis("off")
                    ax[1].axis("off")
                    ax[2].axis("off")
                    plt.savefig(f"{img_folder}/img_{total_samples}.png", bbox_inches="tight", dpi=300)
                    plt.close(fig)
                with open(log_file, "a") as f:
                    f.write(f"{total_samples} {iou:.2f} {(total_iou/total_samples):.2f}\n")
                if args.debug and args.debug_samples!=-1 and total_samples>=args.debug_samples:
                    print(f'Final mIoU: {(total_iou / total_samples):.4f}')
                    torch.cuda.empty_cache()
                    gc.collect()
                    exit()
            except Exception as e:
                print(e)
                continue
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f'Final mIoU: {(total_iou / total_samples):.4f}')
    torch.cuda.empty_cache()
    gc.collect()