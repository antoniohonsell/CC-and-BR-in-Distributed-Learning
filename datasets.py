import torch
from torchvision import datasets, transforms
from typing import Tuple
from torch.utils.data import DataLoader
from torch import nn

def get_mnist_datasets(data_dir: str = "./data") -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Downloads MNIST and returns (train_set, test_set) with standard transforms.
    Output shape: 1x28x28, normalized with dataset stats.
    """
    mean, std = (0.1307,), (0.3081,)

    train_transform = transforms.Compose([
        #transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.MNIST(root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    return train_set, test_set



CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def make_plain_transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

def _reset_bn(m: nn.Module):
    if isinstance(m, nn.BatchNorm2d):
        m.running_mean.zero_()
        m.running_var.fill_(1)
        m.num_batches_tracked.zero_()

@torch.no_grad()
def recompute_bn_stats(model: nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       num_batches: int = 200) -> None:
    """
    Re-estimate BatchNorm running stats by streaming a few batches
    """
    model.apply(_reset_bn)
    was_training = model.training
    model.train()  # BN uses batch stats
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        seen += 1
        if seen >= num_batches:
            break
    model.train(was_training)


def get_cifar10_datasets(data_dir: str = "./data") -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Downloads CIFAR-10 and returns (train_set, test_set) with standard transforms.
    """
    # CIFAR-10 channel-wise mean/std in [0,1]
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
    
     
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_set, test_set