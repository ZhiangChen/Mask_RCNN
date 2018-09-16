rm dataset
pip3 install --upgrade pip
hash -d pip3
python3 setup.py install
ln -s /coco/PythonAPI/pycocotools/ pycocotools
ln -s ../datasets/ dataset
echo "python3 samples/coco/coco.py train --dataset=dataset/coco_dataset --model=coco"

