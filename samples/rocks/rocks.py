"""
classes for lunar rocks dataset
Zhiang Chen
Oct 24, 2018
zch@asu.edu
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
class RocksConfig(Config):
    NAME = "rocks"
    GPU_COUNT = 1 # cannot create model when setting gpu count as 2
    
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + crater
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # IMAGE_CHANNEL = 1 # wrong, the input will be automatically converted to 3 channels (if greyscale, rgb will be repeated)
    
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    
    
    MAX_GT_INSTANCES = 100
    
    DETECTION_MAX_INSTANCES = 200
    
    TRAIN_ROIS_PER_IMAGE = 500
    

############################################################
#  Dataset
############################################################

class RocksDataset(utils.Dataset):
    def load_rocks(self, datadir, subset):
        self.add_class("rocks", 1, "rocks")
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(datadir, subset)
        
        annotation_path = os.path.join(dataset_dir, 'annotations.pickle')
        assert os.path.isfile(annotation_path)
        with open(annotation_path, "rb") as f:
            annotations = pickle.load(f, encoding='latin1')
        del(f)
        
        
        files = os.listdir(dataset_dir)
        files.remove('annotations.pickle')
        image_id = 0
        for file in files:
            if 'label' not in file:
                image_path = os.path.join(dataset_dir, file)
                assert os.path.isfile(image_path)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                mask = annotations[file.split('.')[0]+'.xml']
                
                mask = np.swapaxes(mask, 0, 1)
                mask = np.swapaxes(mask, 1, 2)
                
                self.add_image(
                    "rocks",
                    image_id=image_id,
                    path=image_path,
                    width=width, 
                    height=height,
                    annotation_path=annotation_path,
                    annotation = mask)
                
                image_id += 1
                
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "rocks":
            return super(self.__class__, self).load_mask(image_id)
        
        mask = info["annotation"]
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

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
    dataset.load_rocks('../../dataset/rocks_mask', 'train')
    m, cls = dataset.load_mask(0)
    print(m[0,:,:].max())
    print(cls)
    print(dataset.image_reference(0))
    
    
