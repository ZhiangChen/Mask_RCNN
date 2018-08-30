**This repo is forked from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).**

### 1. docker image
This implemetation requires Tensorflow python3. The official tensorflow docker image only supports tensorflow python2. So [a docker image](https://github.com/ZhiangChen/docker/blob/master/tf3/Dockerfile) for this implementation is built. The container also installed and setup COCOAPI for python3, which resides in ROOT. You can simply make links `ln -s` to the folder you want. 

### 2. setup.sh for the docker container
Unfortunately, when building a image from the Dockerfile, there are some issues of updating the latest version of pip3, which is required to setup the Mask_RCNN repo. So I added a [setup.sh](https://github.com/ZhiangChen/Mask_RCNN/blob/master/setup.sh) to update pip3 and setup the repo (so you don't need to run `python3 setup.py install`). Note this setup script will build a link to coco dataset. You can ignore this in case you don't use coco dataset.

### 3. jupyter notebook
`
./jupyter.sh
`
