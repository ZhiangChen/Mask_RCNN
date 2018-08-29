*This repo is forked from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN). *
### 1. docker image
This implemetation requires Tensorflow python3. The official tensorflow docker image only supports tensorflow python2. So [a docker image](https://github.com/ZhiangChen/docker/blob/master/tf3/Dockerfile) for this implementation is built. The container also installed and setup COCOAPI for python3, which resides in ROOT. You can simply `ln -s` to the folder you want. 


