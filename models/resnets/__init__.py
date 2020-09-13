from . import cifar, imagenet


def ResidualNet(depth, num_classes=1000):
    assert depth in [18, 20, 32, 34, 44, 50, 56, 101, 110, 1202]

    if depth == 18:
        model = imagenet.ResNet(imagenet.BasicBlock, [2, 2, 2, 2], num_classes, depth)

    elif depth == 34:
        model = imagenet.ResNet(imagenet.BasicBlock, [3, 4, 6, 3], num_classes, depth)

    elif depth == 50:
        model = imagenet.ResNet(imagenet.Bottleneck, [3, 4, 6, 3], num_classes, depth)

    elif depth == 101:
        model = imagenet.ResNet(imagenet.Bottleneck, [3, 4, 23, 3], num_classes, depth)

    elif depth == 20:
        model = cifar.resnet20(num_classes)

    elif depth == 32:
        model = cifar.resnet32(num_classes)

    elif depth == 44:
        model = cifar.resnet44(num_classes)

    elif depth == 56:
        model = cifar.resnet56(num_classes)

    elif depth == 110:
        model = cifar.resnet110(num_classes)

    elif depth == 1202:
        model = cifar.resnet1202(num_classes)

    return model
