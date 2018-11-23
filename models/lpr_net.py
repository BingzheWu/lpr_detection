import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from models.ops import *
class lpr_model(nn.Module):
    def __init__(self, planes):
        super(lpr_model, self).__init__()
        self.conv1 = conv_block(3, planes, k_size = 3,  strides = 1, padding = 1)
        self.conv2 = conv_block(planes, planes*2, k_size =3, strides = 1, padding = 1)
        self.pooling = nn.MaxPool2d(2,2)
        planes = planes * 2
        self.conv3 = conv_block(planes, planes*2, k_size = 3, strides = 1, padding = 1)
        planes = planes * 2
        self.res_block1 = res_block(planes, planes)
        print(planes)
        self.stacked_2_res_blocks_1 = nn.Sequential(conv_block(planes, planes*2),
                                                res_block(planes*2, planes*2),
                                                res_block(planes*2, planes*2),
                                                nn.MaxPool2d(2,2))
        planes = planes * 2
        self.stacked_2_res_blocks_2 = nn.Sequential(conv_block(planes, planes),
                                                res_block(planes, planes),
                                                res_block(planes, planes),
                                                nn.MaxPool2d(2,2))
        self.conv4 = conv_block(planes, planes*2)
        planes = planes * 2
        self.stacked_4_res_blocks_1 = []
        for i in range(4):
            self.stacked_4_res_blocks_1.append(res_block(planes, planes))
        self.stacked_4_res_blocks_1 = nn.Sequential(*self.stacked_4_res_blocks_1)
        self.xprobs = conv_block(planes, 2, alpha = None) 
        self.xbbox = conv_block(planes, 6, alpha = None)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        out = self.conv3(out)
        out = self.res_block1(out)
        out = self.pooling(out)
        out = self.stacked_2_res_blocks_1(out)
        out = self.stacked_2_res_blocks_2(out)
        out = self.conv4(out)
        out = self.stacked_4_res_blocks_1(out)
        probs = self.xprobs(out)
        probs = F.softmax(probs)
        xbox = self.xbbox(out)
        o = torch.cat((probs, xbox), 1)
        o = o.permute((0,2,3,1))
        return o
def test_lpr_model():
    net = lpr_model(16)
    x = torch.zeros((1,3,208,208))
    o = net.forward(x)
    print(o.size())
if __name__ == '__main__':
    test_lpr_model()