"""
classes for camtraps dataset
Harish Anand, Zhiang Chen, Jnaneshwar Das
Dec 9, 2018
jdas5@asu.edu
"""

import os
import sys
import numpy as np
import skimage.draw
import pickle
import argparse
import matplotlib.pyplot as plt

from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Dataset config
############################################################
class CamtrapsConfig(Config):
    NAME = "camtraps"
    GPU_COUNT = 1 # cannot create model when setting gpu count as 2
    
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + crater
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # IMAGE_CHANNEL = 1 # wrong, the input will be automatically converted to 3 channels (if greyscale, rgb will be repeated)
    
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    
    
    MAX_GT_INSTANCES = 150
    
    DETECTION_MAX_INSTANCES = 200
    
    TRAIN_ROIS_PER_IMAGE = 500
    

############################################################
#  Dataset
############################################################

class CamtrapsDataset(utils.Dataset):
    def load_camtraps(self, datadir, subset):
        self.add_class("camtraps", 1, "animals")
        print(datadir)
        print(subset)        
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(datadir, subset)
        
        files = os.listdir(dataset_dir)
        
        image_id = 0
        for file in files:
            if '.JPG' in file:
                image_path = os.path.join(dataset_dir, file)
                assert os.path.isfile(image_path)
                
                annotation_path = os.path.join(dataset_dir, file.split('.')[0]+'.npy')
                assert os.path.isfile(annotation_path)
                print(file) 
                #image = skimage.io.imread(image_path)
                height, width = 1536, 2048
                
                self.add_image(
                    "camtraps",
                    image_id=image_id,
                    path=image_path,
                    width=width, 
                    height=height,
                    annotation_path=annotation_path)
                
                image_id += 1
                
    def load_mask(self, image_id):
        print(image_id)
        info = self.image_info[image_id]
        print(info)
        if info["source"] != "camtraps":
            return super(self.__class__, self).load_mask(image_id)
        
        mask = np.load(info["annotation_path"])
        mask = np.swapaxes(mask, 0, 1)
        mask = np.swapaxes(mask, 1, 2)    
        if mask.shape[2] == 1:
            mask = mask[:,:,0]
            print(mask.shape)
    
        if len(mask.shape) == 2:
            h,w = mask.shape
            mask_ = mask.reshape((h,w,1)).astype(np.bool)
            return mask_, np.zeros(1).astype('int32')
        
        else:
            h,w,c = mask.shape
            print(h,w,c)
            mask_ = np.zeros(mask.shape, dtype='uint8')
            mask_ = np.logical_or(mask, mask_)
            classes = np.ones([mask.shape[-1]], dtype=np.int32)        
            print(mask_.shape)
            print(classes.shape)
            return mask_, classes
        

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "animals":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    def display_mask(self, image_id):
        masks, ids = self.load_mask(image_id)
        mask = masks.max(2)
        plt.imshow(mask)
        plt.show()
    

############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = CamtrapsConfig()
    config.display()
    dataset = CamtrapsDataset()
    dataset.load_camtraps('/models/datasets/camtraps', 'train')
    m, cls = dataset.load_mask(0)
    print(m[0,:,:].max())
    print(cls)
    print(dataset.image_reference(0))
    
    
