import torch
import torch.nn as nn
import numpy as np

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block = block, layers = [3, 4, 6, 3], image_channels = 3, num_classes =None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x ,lam = 0 ,args = None,mix = False):
        layer = [self.layer1,self.layer2,self.layer3,self.layer4]
        mixup_layer = ['conv1','layer1','layer2',"layer3",'layer4','fc']
        if mix:
            k = mixup_layer.index(args.feature_layer) + 1
        else : k = -1
        # print(f"x shape 1: {x.shape}")
        if k == 0:
            bs = x.size(0)/2
            bs = int(bs)
            x1 = x[:bs]
            x2 = x[bs:]
            x = lam * x1 + (1 - lam) * x2
            # print(f"K = 0 x1 shape:{x1.shape},x2 shape :{x2.shape},x shape:{x.shape}")
            x = _noise(x, add_noise_level=args.add_noise, mult_noise_level=args.mult_noise)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(len(layer)):
            if k == (i+1):
                bs = x.size(0)/2
                bs = int(bs)
                # print(f"k = {k} x shape:{x.shape}")
                x1 = x[:bs]
                x2 = x[bs:]
                x = lam * x1 + (1 - lam) * x2
                # print(f"K = {k} x1 shape:{x1.shape},x2 shape :{x2.shape},x shape:{x.shape}")
                x = _noise(x, add_noise_level=args.add_noise, mult_noise_level=args.mult_noise)
            layer_ = layer[i]
            x = layer_(x)

        if k == 5:
            bs = x.size(0)/2
            bs = int(bs)
            x1 = x[:bs]
            x2 = x[bs:]
            x = lam * x1 + (1 - lam) * x2
            # print(f"k = 5  x1 shape:{x1.shape},x2 shape :{x2.shape},x shape:{x.shape}")
            x = _noise(x, add_noise_level=args.add_noise, mult_noise_level=args.mult_noise)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if k == 6:
            bs = x.size(0)/2
            bs = int(bs)
            x1 = x[:bs]
            x2 = x[bs:]
            x = lam * x1 + (1 - lam) * x2
            # print(f"x1 shape:{x1.shape},x2 shape :{x2.shape},x shape:{x.shape}")
            x = _noise(x, add_noise_level=args.add_noise, mult_noise_level=args.mult_noise)
        
        # print(f"x shape 2: {x.shape}")
        
        return x        

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparsity_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.empty(x.shape, dtype=torch.float32, device='cuda').normal_()
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (2 * torch.empty(x.shape, dtype=torch.float32, device='cuda').uniform_() - 1) + 1
    return mult_noise * x + add_noise