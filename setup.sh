rm coco_dataset
pip3 install --upgrade pip
hash -d pip3
python3 setup.py install
ln -s /coco/PythonAPI/pycocotools/ pycocotools
ln -s ../datasets/coco_dataset/ coco_dataset
echo "python3 samples/coco/coco.py train --dataset=coco_dataset --model=coco"

