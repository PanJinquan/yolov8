#!/usr/bin/env bash

# Instance segmentation
model="cfg/models/v8/yolov8-seg.yaml"
weights="data/model/pretrained/yolov8n-seg.pt"
#data="cfg/datasets/coco-data-seg.yaml"
data="cfg/datasets/coco-aije-seg.yaml"
cfg="cfg/segment-hyp.yaml"
python train.py --device 4 --model $model --weights $weights --data $data --batch 8 --cfg $cfg


# object detection
#model="cfg/models/v8/yolov8s.yaml"
#weights="data/model/pretrained/yolov8s.pt"
#data="cfg/datasets/coco-data-seg.yaml"
#cfg="cfg/detect-hyp.yaml"
#output="output/detect"
#python train.py --model $model --weights $weights --data $data --cfg $cfg
