"""
classes for rocks dataset
Zhiang Chen
Feb 5, 2019
zch@asu.edu
"""

import os
import sys
import numpy as np
import skimage.draw
import pickle
import argparse
import matplotlib.pyplot as plt
import skimage.io
import random

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
class RocksConfig(Config):
    IMAGE_CHANNEL_COUNT = 4 # RGB-D four channels
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 105.0]) # four channels as well
    
    NAME = "rocks"
    GPU_COUNT = 1 # cannot create model when setting gpu count as 2
    
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + crater
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    
    
    MAX_GT_INSTANCES = 512
    
    DETECTION_MAX_INSTANCES = 512
    
    TRAIN_ROIS_PER_IMAGE = 500
    ROI_POSITIVE_RATIO = 0.5
    
    USE_MINI_MASK = False
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.5,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 0.5,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    

############################################################
#  Dataset
############################################################

class RocksDataset(utils.Dataset):
    def load_rocks(self, datadir, subset):
        self.add_class("rocks", 1, "rocks")
        
        assert subset in ["train", "val", "infer"]
        dataset_dir = os.path.join(datadir, subset)
        files = [x for x in os.listdir(dataset_dir) if x.endswith(".jpg")]
        
        if subset == "infer":
            fake_annotation = np.zeros((400,400))
            np.save(dataset_dir+"/fake_annotation.npy", fake_annotation)
            

        image_id = 0
        for file in files:
            if '.jpg' in file:
                image_path = os.path.join(dataset_dir, file)
                assert os.path.isfile(image_path)
                
                if subset in ["train","val"]:
                    annotation_path = os.path.join(dataset_dir, file.split('.')[0]+'.npy')
                    
                else:
                    annotation_path = os.path.join(dataset_dir, "fake_annotation.npy")
                    
                assert os.path.isfile(annotation_path)

                #image = skimage.io.imread(image_path)
                height, width = 400, 400

                self.add_image(
                    "rocks",
                    image_id=image_id,
                    path=image_path,
                    width=width, 
                    height=height,
                    annotation_path=annotation_path)

                image_id += 1

    def load_image(self, image_id):
        """load a fake RGBD image"""
        image = skimage.io.imread(self.image_info[image_id]['path'])
        y,x,i = image.shape
        c = random.randint(0,2)
        D = image[:,:,c].reshape((y,x,1))
        image = np.concatenate((image, D), axis=2)
        return image
        
    
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "rocks":
            return super(self.__class__, self).load_mask(image_id)
        
        mask = np.load(info["annotation_path"])
        
        if len(mask.shape) == 2:
            h,w = mask.shape
            mask_ = mask.reshape((h,w,1)).astype(np.bool)
            return mask_, np.zeros(1).astype('int32')
        
        else:
            h,w,c = mask.shape
            mask_ = np.zeros(mask.shape, dtype='uint8')
            mask_ = np.logical_or(mask, mask_)
            classes = np.ones([mask.shape[-1]], dtype=np.int32)        
            return mask_, classes
        

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "rocks":
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
    config = RocksConfig()
    config.display()
    dataset = RocksDataset()
    dataset.load_rocks('../../dataset/rocks', 'train')
    m, cls = dataset.load_mask(0)
    print(m[0,:,:].max())
    print(cls)
    print(dataset.image_reference(0))
    
    
