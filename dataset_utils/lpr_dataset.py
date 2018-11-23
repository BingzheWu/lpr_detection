import torch
import torch.utils.data as data
import os
import cv2
import numpy as np
from os.path import splitext
import sys
sys.path.append('./')
from detect_utils.utils import image_files_from_folder
import torchvision
class lpr_dataset(data.Dataset):

    def __init__(self, lpr_root, transform = None, image_sets = ['cars_train_augment'], dataset = 'lpr'):
        super(lpr_dataset, self).__init__()
        self.lpr_root = lpr_root
        self.image_sets = image_sets
        if transform is None:
            self.transform = self.trans()
        self.name = dataset
        self.prefix_list = []
        for image_set in image_sets:
            self.dir = os.path.join(lpr_root, image_set)
            Files = image_files_from_folder(self.dir)
            for file in Files:
                prefix = splitext(file)[0]
                self.prefix_list.append(prefix)
    def __getitem__(self, index):
        prefix = self.prefix_list[index]
        image = cv2.imread(prefix+'.jpg')
        target = np.load(prefix+'.npy')
        if self.transform is not None:
            image = image[..., (2,1,0)]
            image = self.transform(image)
        return image, target
    def __len__(self):
        return len(self.prefix_list)
    def trans(self):
        torch_trans = torchvision.transforms
        transform = torch_trans.Compose([torch_trans.ToTensor()])
        return transform
def test_dataset():
    lpr = lpr_dataset('/home/bingzhe/projects/lpr/training_data/cars_images/')
    lpr[0]

if __name__ == '__main__':
    test_dataset()
