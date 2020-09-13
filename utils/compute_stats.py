import argparse
import os

import numpy as np
import torchvision.transforms as transforms

from datasets.smallimagenet import SmallImagenet
from datasets.tinyimagenet import TinyImageNet


def compute_mean_std(dataset):
    count = 0
    r_mean, r_var = np.zeros(3), np.zeros(3)
    for img, y in dataset:
        img = img.view(3, -1)
        r_mean += img.mean(axis=-1).numpy()
        r_var += img.var(axis=-1).numpy()
        count += 1
    return r_mean / count, np.sqrt(r_var / count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes mean and std. of dataset')
    parser.add_argument('--dataset', required=True, type=str, choices=["SmallImageNet", "TinyImageNet"],
                        help='Dataset to use')
    parser.add_argument('--resolution', type=int, help="Resolution for SmallImageNet (default: 32)")
    parser.add_argument('--data-dir', type=str, default="data", help="Path to load dataset from (default: data)")
    args = parser.parse_args()

    dir_name = f"{args.dataset}_{args.resolution}x{args.resolution}" if args.dataset == "SmallImageNet" else "tiny-imagenet-200"
    root = os.path.join(args.data_dir, dir_name)
    dataset = SmallImagenet(root=root, size=args.resolution, train=True, transform=transforms.ToTensor()) \
        if args.dataset == "SmallImageNet" else TinyImageNet(root=root, train=True, transform=transforms.ToTensor())
    mean, std = compute_mean_std(dataset)
    print(f"Mean: {mean}, Std: {std}")
