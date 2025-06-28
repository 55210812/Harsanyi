"""
This code is adapted from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

    vgg in pytorch
    [1] Karen Simonyan, Andrew Zisserman
        Very Deep Convolutional Networks for Large-Scale Image Recognition.
        https://arxiv.org/abs/1409.1556v6
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ["vgg11tiny",
           "vgg13tiny",
           "vgg16tiny",
           "vgg19tiny"]

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=100, init_weights=True, use_dropout=True):
        super().__init__()
        self.features = features
        self.use_dropout = use_dropout

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

    def get_feature(self, x, layer):
        if layer == "layer1":
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            return out
        elif layer == "a_second_relu":
            out = self.features[0:5](x)
            return out
        elif layer == "b_second_relu":
            out = self.features[0:4](x)
            return out
        else:
            raise NotImplementedError

    def forward_feature(self, feature, layer):
        """
        Calculate the output of the DNN given intermediate layer feature
        :param feature:
        :param layer:
        :return:
        """
        if layer == "layer1":
            out = self.layer2(feature)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        elif layer == "a_second_relu":
            out = self.features[5:](feature)
            out = out.view(out.size()[0], -1)
            out = self.classifier(out)
            return out
        elif layer == "b_second_relu":
            out = self.features[4:](feature)
            out = out.view(out.size()[0], -1)
            out = self.classifier(out)
            return out
        else:
            raise NotImplementedError


    def set_store_activation_rate(self):
        from utils.tools import AverageMeter

        def get_hook(name):
            def store_act_rate(m, i, o):
                self.activation_rate[name].update((o > 0).float().mean().item())
                print(o.shape)
            return store_act_rate

        if len(self.features) == 29:  # VGG-11
            self.activation_rate = {
                "conv_11": AverageMeter(), "conv_21": AverageMeter(), "conv_32": AverageMeter(),
                "conv_42": AverageMeter(), "conv_52": AverageMeter()
            }
            self.features[2].register_forward_hook(get_hook("conv_11"))
            self.features[6].register_forward_hook(get_hook("conv_21"))
            self.features[13].register_forward_hook(get_hook("conv_32"))
            self.features[20].register_forward_hook(get_hook("conv_42"))
            self.features[27].register_forward_hook(get_hook("conv_52"))
        elif len(self.features) == 35:  # VGG-13
            self.activation_rate = {
                "conv_12": AverageMeter(), "conv_22": AverageMeter(), "conv_32": AverageMeter(),
                "conv_42": AverageMeter(), "conv_52": AverageMeter()
            }
            self.features[5].register_forward_hook(get_hook("conv_12"))
            self.features[12].register_forward_hook(get_hook("conv_22"))
            self.features[19].register_forward_hook(get_hook("conv_32"))
            self.features[26].register_forward_hook(get_hook("conv_42"))
            self.features[33].register_forward_hook(get_hook("conv_52"))
        elif len(self.features) == 44:  # VGG-16
            self.activation_rate = {
                "conv_12": AverageMeter(), "conv_22": AverageMeter(), "conv_33": AverageMeter(),
                "conv_43": AverageMeter(), "conv_53": AverageMeter()
            }
            self.features[5].register_forward_hook(get_hook("conv_12"))
            self.features[12].register_forward_hook(get_hook("conv_22"))
            self.features[22].register_forward_hook(get_hook("conv_33"))
            self.features[33].register_forward_hook(get_hook("conv_43"))
            self.features[42].register_forward_hook(get_hook("conv_53"))
        elif len(self.features) == 53:  # VGG-19
            self.activation_rate = {
                "conv_12": AverageMeter(), "conv_22": AverageMeter(), "conv_34": AverageMeter(),
                "conv_44": AverageMeter(), "conv_54": AverageMeter()
            }
            self.features[5].register_forward_hook(get_hook("conv_12"))
            self.features[12].register_forward_hook(get_hook("conv_22"))
            self.features[25].register_forward_hook(get_hook("conv_34"))
            self.features[38].register_forward_hook(get_hook("conv_44"))
            self.features[51].register_forward_hook(get_hook("conv_54"))
        else:
            raise NotImplementedError


def make_layers(cfg, input_channel=3, batch_norm=False):
    layers = []

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)




def vgg11tiny(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['A'], input_channel, batch_norm=False), num_classes)

def vgg13tiny(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['B'], input_channel, batch_norm=False), num_classes)

def vgg16tiny(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['D'], input_channel, batch_norm=False), num_classes)

def vgg19tiny(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['E'], input_channel, batch_norm=False), num_classes)

