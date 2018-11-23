import os
from os.path import isfile, isdir, basename, splitext
import sys
sys.path.append('./')
from detect_utils.label_utils import readShapes
from detect_utils.utils import image_files_from_folder, show, label_files_from_folder
from detect_utils.sampler import augment_sample, labels2output_map
import cv2
def augmentor():
    image_dir = '/home/bingzhe/projects/lpr/training_data/cars_images/cars_train/'
    dst_dir = '/home/bingzhe/projects/lpr/training_data/cars_images/cars_train_augment/'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    Files = image_files_from_folder(image_dir)
    for file in Files:
        #imgfile = splitext(file)[0]+'.jpg'
        prefix = splitext(file)[0]
        save_prefix = prefix.split('/')[-1]
        img_save_path = os.path.join(dst_dir, save_prefix)
        labelfile = prefix+'.txt'
        if isfile(labelfile):
            L = readShapes(labelfile)
            I = cv2.imread(file)
            XX, llp, pts = augment_sample(I, L[0].pts, 208, save_path_prefix = img_save_path)


if __name__ == '__main__':
    for i in range(100):
        augmentor()
        print("One epoch sucess")

