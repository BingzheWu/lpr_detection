import torch
import torch.nn as nn
import functools
import torch.nn.functional as F


def conv_block(in_c, out_c, k_size = 3, strides = 1, padding = 1, name='conv_blcok', 
    alpha = 0., bias = False, batch_norm = True):
    out = nn.Sequential()
    out.add_module(name+'_conv', nn.Conv2d(in_c, out_c, k_size, strides, padding, bias = bias))
    if batch_norm:
        out.add_module(name+'_norm', nn.BatchNorm2d(out_c))
    if alpha is not None:
        out.add_module(name+'_activation', nn.ReLU())
    return out

class res_block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(res_block, self).__init__()
        self.expansion = 1
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv_block(inplanes, planes, k_size = 3, strides = 1, padding = 1)
        self.conv2 = conv_block(planes, planes, k_size = 3, strides = 1, alpha = None, padding = 1)
        self.downsample = downsample
        self.relu = nn.ReLU()
        if stride !=  1 or self.inplanes != planes * self.expansion:
            self.downsample = conv_block(self.inplanes, planes* self.expantion, k_size = 1, strides = stride, padding = 0, alpha = None, batch_norm=True )
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
