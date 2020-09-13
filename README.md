# Scripts to train ResNets on Downsampled Variants of the ImageNet dataset

### Small-ImageNet
[A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets](https://arxiv.org/abs/1707.08819)

Download and extract dataset:
 `python utils/prepare_dataset.py --dataset SmallImageNet --resolution 32 --data-dir data --download-dir data/compressed`

Supported resolutions: 8, 16, 32, 64 (must be >=32 for ImageNet ResNets)

Training:

CIFAR ResNets: `python train.py data --dataset SmallImageNet --size 32 --classes 1000 --depth 20 --ngpu 1 --epochs 200 -b 128 --lr 0.1 --momentum 0.9 --wd 5e-4 --prefix test --project Imagenet`

ImageNet ResNets: `python train.py data --dataset SmallImageNet --size 32 --classes 1000 --depth 18 --ngpu 1 --epochs 100 -b 256 --lr 0.1 --momentum 0.9 --wd 1e-4 --prefix test --project Imagenet`

`classes` can be changed to select a subset of the dataset. `size` is the resolution of the dataset.

### Tiny-ImageNet
[Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com)

Download and extract dataset:
 `python utils/prepare_dataset.py --dataset TinyImageNet --data-dir data --download-dir data/compressed`

Training:

CIFAR ResNets: `python train.py data --dataset TinyImageNet --depth 20 --ngpu 1 --epochs 200 -b 128 --lr 0.1 --momentum 0.9 --wd 5e-4 --prefix test --project Imagenet`

ImageNet ResNets: `python train.py data --dataset TinyImageNet --depth 18 --ngpu 1 --epochs 100 -b 256 --lr 0.1 --momentum 0.9 --wd 1e-4 --prefix test --project Imagenet`

### Utils
Run `python utils/compute_stats.py --dataset SmallImageNet --resolution 8 --data-dir data` to compute mean and std. of the dataset. `dataset_stats.json` contains stats for 1000 classes.