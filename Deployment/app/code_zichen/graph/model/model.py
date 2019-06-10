from torch.nn import init

from .base_model import *


def init_fc_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def model_init(model_self):
    model_self.apply(init_fc_weights)
    for key in model_self.state_dict():
        if key.split('.')[-1] == "weight":
            if "conv" in key:
                init.kaiming_normal_(model_self.state_dict()[key], mode='fan_out')
            if "bn" in key:
                if "SpatialGate" in key:
                    model_self.state_dict()[key][...] = 0
                else:
                    model_self.state_dict()[key][...] = 1
        elif key.split(".")[-1] == 'bias':
            model_self.state_dict()[key][...] = 0


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class CbamResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2]):
        self.inplanes = 64
        super(CbamResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=[7, 7], stride=7)

        self.fc = nn.Sequential(Flatten(),
                                nn.Dropout2d(p=0.2),
                                nn.Linear(512*4, 1))

        self.sigmoid = nn.Sigmoid()

        # init weight
        model_init(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)

        x = self.maxpool_1(x)
        # print(x.shape)

        x = self.layer3(x)
        # print(x.shape)

        x = self.maxpool_2(x)
        # print(x.shape)

        x = self.fc(x)
        # print(x.shape)

        x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    import torch
    import numpy as np

    images = torch.FloatTensor(np.ones([2, 1, 224, 224]))
    model = CbamResNet()
    outputs = model(images)
    print(outputs.shape)
    print(outputs)
