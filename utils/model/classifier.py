''' ref:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(BasicBlock, self).__init__()
        planes = planes * wide
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(Bottleneck, self).__init__()
        mid_planes = planes * wide
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_dims, out_dims, wide=1):
        super(ResNet, self).__init__()
        self.wide = wide
        self.in_planes = 64
        # self.conv1 = nn.Conv2d(in_dims, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dims, 64, kernel_size=7, stride=2, padding=3, bias=False) # for 224x224 input
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for 224x224 input
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, out_dims)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wide))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out) # for 224x224 input
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def feature_extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(in_dims, out_dims):
    return ResNet(BasicBlock, [2,2,2,2], in_dims, out_dims, 1)


class MLP4(torch.nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.fc1 = torch.nn.Linear(in_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.fc4 = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP3(torch.nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.fc1 = torch.nn.Linear(in_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP2(torch.nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP1(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.fc1(x)





__classifier_zoo__ = {
        "resnet18": resnet18,
        "mlp4": MLP4,
        "mlp3": MLP3,
        "mlp2": MLP2,
        "mlp1": MLP1,
}


def get_classifier(name: str, **kwargs):
    return __classifier_zoo__[name](**kwargs)
