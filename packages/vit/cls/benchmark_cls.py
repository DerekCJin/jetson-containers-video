# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
#
# model: /data/models/efficientvit/cls/b1-r288.pt
# run in /opt/efficientvit: python3 benchmark_cls.py --model b1-r288

import argparse
import math
import os
import datetime
import socket
import numpy as np
import cv2
from PIL import Image

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_cls_model

from typing import Dict


def load_image(data_path: str, mode="rgb"):
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/data/datasets/tiny-imagenet-200/test")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=50)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str)
    #parser.add_argument("--model", type=str, default="b1")
    parser.add_argument("--weight_url", type=str, default=None)
    #parser.add_argument("--weight_url", type=str, default="/data/models/efficientvit/cls/b1_r288.pt")
    parser.add_argument("--image_path", type=str, default="assets/fig/cat.jpg") #get 1 image in tiny_imagenet/test folder
    parser.add_argument("--output_path", type=str, default="/data/benchmarks/benchmark_efficientvit_cls.txt")
    parser.add_argument('-s', '--save', type=str, default='/data/benchmarks/benchmark_efficientvit_cls.txt', help='txt file to save benchmarking results to')

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    

    #get 3 images from imagenet/test
    #if not args.images:
    args.images = [
        "/data/datasets/tiny-imagenet-200/test/images/test_0.JPEG",
        "/data/datasets/tiny-imagenet-200/test/images/test_5000.JPEG",
        "/data/datasets/tiny-imagenet-200/test/images/test_9999.JPEG",
    ]

    print(args)

    model = create_cls_model(args.model, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    predicted_labels = {"image":"label"}        
    for img_path in args.images:
        # load image
        image = load_image(img_path)
        #oh, ow, _ = image.shape
        #image = cv2.resize(image, dsize=(ow, oh))
        image = transform(image)
        #output = model(images)
        image = image.unsqueeze(0)
        output = model(image)
        l = torch.argmax(output)
        #l = torch.max(output.data,1)
        predicted_labels[img_path] = l
        print(f"{img_path}:{l}")

    s=str(predicted_labels)
    print(f"********{s}") 
    print(f"args.save = {args.save}")
    if args.save:

        if not os.path.isfile(args.save):  # txt header
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, api, model, weight_url\n\n")
        else:
            os.remove(args.save)
            with open(args.save, 'w') as file:
                file.write(f"timestamp, hostname, api, model, weight_url\n\n")
        with open(args.save, 'a') as file:
            file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ")
            file.write(f"efficientvit-python, {args.model}, {args.weight_url}\n\n")
            #file.write(f"{str(predicted_labels)}\n\n")  
            for k,v in predicted_labels.items():
                file.write(f"{k}\t{v}\n\n")

if __name__ == "__main__":
    main()

