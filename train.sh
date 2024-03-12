#!/usr/bin/env bash
# DDP多卡训练时，Argument命令行参数会失效
# Instance segmentation
#model="cfg/models/v8/yolov8-seg.yaml"
#weights="data/model/pretrained/yolov8n-seg.pt"
model="cfg/models/v8/yolov8m-seg.yaml"
weights="data/model/pretrained/yolov8m-seg.pt"
#data="cfg/datasets/coco-data-seg.yaml"
#data="cfg/datasets/coco-aije-seg.yaml"
data="cfg/datasets/labelme-data-seg.yaml"
cfg="cfg/segment-hyp.yaml"
output="output/indoor"
python train.py --model $model --weights $weights --data $data --batch 64 --cfg $cfg  --output $output --device 0,1,2,3

# object detection
#model="cfg/models/v8/yolov8s.yaml"
#weights="data/model/pretrained/yolov8s.pt"
#data="cfg/datasets/coco-data-seg.yaml"
#cfg="cfg/detect-hyp.yaml"
#output="output/detect"
#python train.py --model $model --weights $weights --data $data --cfg $cfg
