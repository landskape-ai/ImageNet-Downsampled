import argparse
import os
import sys
import zipfile

import wget


def bar_progress(current, total, width=80):
    progress_message = f"Downloading: {int(current / total * 100)}% [{current * 1e-6:.2f} / {total * 1e-6:.2f}] MB"
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_smallimagenet(resolution, down_dir):
    os.makedirs(down_dir, exist_ok=True)
    file_name = "Imagenet{}_{}.zip"
    base_url = "http://www.image-net.org/data/downsample/{}"
    for t in ["train", "val"]:
        print(f"\nDownloading Small-ImageNet-{resolution} {t} set...")
        f_name = file_name.format(resolution, t)
        url = base_url.format(f_name)
        wget.download(url, down_dir, bar=bar_progress)


def extract_file(file, out_dir, delete=False):
    print(f"\nExtracting {file} ...")
    with zipfile.ZipFile(file, 'r') as f:
        f.extractall(out_dir)
    if delete:
        os.remove(file)


def download_tinyimagenet(down_dir):
    os.makedirs(down_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    print(f"Downloading Tiny-ImageNet...")
    wget.download(url, down_dir, bar=bar_progress)


def load_smallimagenet_zip(resolution, down_dir):
    filename = "{}/Imagenet{}_{}.zip"
    train_filename = filename.format(down_dir, resolution, "train")
    val_filename = filename.format(down_dir, resolution, "val")
    if not (os.path.isfile(train_filename) or os.path.isfile(val_filename)):
        download_smallimagenet(resolution, down_dir)
    return train_filename, val_filename


def load_tinyimagenet_zip(down_dir):
    filename = os.path.join(down_dir, "tiny-imagenet-200.zip")
    if not os.path.isfile(filename):
        download_tinyimagenet(down_dir)
    return filename,


def prepare(dataset, resolution, down_dir, out_dir):
    assert dataset in ["SmallImageNet", "TinyImageNet"]
    files = load_smallimagenet_zip(resolution, down_dir) if dataset == "SmallImageNet" else load_tinyimagenet_zip(
        down_dir)
    out = f"{dataset}_{resolution}x{resolution}" if dataset == "SmallImageNet" else ''
    out_dir = os.path.join(out_dir, out)
    for file in files:
        extract_file(file, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ImageNet Downloader')
    parser.add_argument('--dataset', required=True, type=str, choices=["SmallImageNet", "TinyImageNet"],
                        help='Dataset to prepare')
    parser.add_argument('--resolution', type=int, help="Resolution for SmallImageNet (default: 32)")
    parser.add_argument('--data-dir', type=str, help="Path to extract dataset (default: data)")
    parser.add_argument('--download-dir', type=str, help="Path to download dataset (default: data/compressed")
    args = parser.parse_args()
    prepare(args.dataset, args.resolution, args.download_dir, args.data_dir)
