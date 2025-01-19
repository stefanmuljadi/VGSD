import torch
from torch import nn

from .config import resnext_101_32_path
from .resnext_101_32x4d_ import resnext_101_32x4d
import torch._utils
import pdb 

class ResNeXt101(nn.Module):
    def __init__(self, pretained=True):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d
        if pretained:
            # pass 
            print('AAAAAA !!!!!!', '/content/VGSD/networks/resnext_modify/resnext_101_32x4d.pth')
            try:
                print("a")
                msg = net.load_state_dict(torch.load('/content/VGSD/networks/resnext_modify/resnext_101_32x4d.pth', weights_only=True))
            except FileNotFoundError:
                print("e")
                print()
                msg = net.load_state_dict(torch.load('../' + resnext_101_32_path))
            print(msg)
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

