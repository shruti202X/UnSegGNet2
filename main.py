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
parser.add_argument("--dataset", type=str, default='Clinical')
parser.add_argument("--threshold", type=float, default=0.2)
parser.add_argument("--conv_type", type=str, default='GAT')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--debug_samples", type=int, default=-1)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    print(device)

    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type, args.batch_size)

    torch.cuda.empty_cache()
    gc.collect()

    ds = Dataset(args.dataset)

    filename = "_".join([args.conv_type, args.dataset, args.activation, args.loss_type, args.process, str(args.threshold), str(args.epochs), str(args.batch_size)])
    img_folder = "../Img_" + filename
    if os.path.exists(img_folder):
        now = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
        img_folder += now
        filename += now
    log_file = f"logs/log_{filename}.store"
    os.mkdir(img_folder)

    total_iou = 0
    total_samples = 0
    start = 0
    imgs = []
    masks = []
    while ds.loader > 0:
        for img, mask in ds.load_samples():
            start += 1
            imgs.append(img)
            masks.append(mask)
            # print(f"Processing {total_samples + start}th sample")
            if start == args.batch_size:
                try:
                    ious, best_segmentations, segmentation_over_images = seg.segment(imgs, masks)
                    sum_ious = sum(ious)
                    total_iou += sum_ious
                    total_samples += args.batch_size

                    for i in range(args.batch_size):
                        if args.debug and ious[i] <= 0.5:
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            axes[0].imshow(segmentation_over_images[i])
                            axes[0].axis("off")
                            axes[1].imshow(best_segmentations[i])
                            axes[1].axis("off")
                            axes[2].imshow(mask)
                            axes[2].axis("off")
                            plt.savefig(f'{img_folder}/failed_{total_samples}.png', bbox_inches='tight', dpi=300)
                            plt.close(fig)
      
                    iou = sum_ious / len(ious)
                    print(f"IoU for Batch {total_samples/args.batch_size} : {iou:.2f}, {ious}, mIoU so far: {(total_iou/total_samples):.2f}")
                    with open(log_file, "a") as f:
                        f.write(f"IoU for Batch {total_samples/args.batch_size} : {iou:.2f}, {ious}, mIoU so far: {(total_iou/total_samples):.2f}\n")
                        
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    if args.debug and args.debug_samples != -1 and total_samples > args.debug_samples:
                        exit()

                except Exception as e:
                    print(e)
                    continue

                start = 0
                imgs = []
                masks = []
            
            gc.collect()
            torch.cuda.empty_cache()

    if total_samples > 0:
        print(f'Final mIoU: {(total_iou / total_samples):.4f}')
    else:
        print("No valid samples processed, mIoU cannot be computed.")


    gc.collect()
    torch.cuda.empty_cache()
