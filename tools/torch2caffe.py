import sys
sys.path.append('/home/bingzhe/nn_tools')
sys.path.append('.')
import pytorch_to_caffe
import torch
from models.lpr_net import lpr_model
def main():
    name = 'lpr'
    model = lpr_model(16)
    model.eval()
    state_dict = torch.load('experiment/lpr/lpr_resnet/checkpoint.pth.tar')['state_dict']
    model.load_state_dict(state_dict)
    input = torch.ones([1,3,428,428])
    pytorch_to_caffe.trans_net(model, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
main()