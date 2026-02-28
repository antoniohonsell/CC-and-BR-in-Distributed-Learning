import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


### CNN for the MNIST dataset

class LightNet2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.fc1 = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (N, 6, 14, 14)
        x = torch.flatten(x, 1)               # (N, 1176)
        x = self.fc1(x)                       # logits (N, 10)
        return x
    

### ---- Model: ResNet-18 adapted to CIFAR-10 ----
def make_resnet18_cifar10() -> nn.Module:
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    # CIFAR-10: 32x32 -> use 3x3 stride=1 and remove maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m