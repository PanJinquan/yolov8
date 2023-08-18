# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-15 13:48:21
    @Brief  :
"""

# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-15 11:45:16
    @Brief  : docs:https://github.com/ultralytics/ultralytics
              1. https://docs.ultralytics.com/tasks/detect/
              2. https://docs.ultralytics.com/tasks/segment
              3. https://docs.ultralytics.com/tasks/pose/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import numpy as np
import cv2
from easydict import EasyDict
from ultralytics import YOLO
from ultralytics import utils
from pybaseutils import image_utils, file_utils, color_utils, yaml_utils
from ultralytics.models.yolo.detect import DetectionTrainer


class Trainer(object):
    def __init__(self, opt):
        self.opt = EasyDict(opt.__dict__)
        self.model = YOLO(opt.model).load(opt.weights)  # build from YAML and transfer weights
        self.names = self.model.names
        print("ROOT                  :{}".format(utils.ROOT))
        print("DEFAULT_CFG_PATH      :{}".format(utils.DEFAULT_CFG_PATH))
        print("NUM_THREADS           :{}".format(utils.NUM_THREADS))
        print("DEFAULT_CFG_DICT      :{}".format(utils.DEFAULT_CFG_DICT))
        print("weights               :{}".format(opt.weights))
        print("model                 :{}".format(opt.model))
        print("hype                  :{}".format(opt.hype))
        print("data                  :{}".format(opt.data))
        print("model num class       :{}".format(len(self.names)))

    def run(self, ):
        self.model.train(data=self.opt.data)


def parse_opt():
    """
    配置信息在：~/.config/Ultralytics/settings.yaml
    数据集加载地址：ultralytics/models/yolo/detect/train.py
    from ultralytics.data import build_dataloader, build_yolo_dataset
    """
    image_dir = 'data/test_image'
    weights = "data/model/pretrained/yolov8n-seg.pt"  # 模型文件yolov5s05_640
    # model = "yolov8n-seg1.yaml"
    model = "cfg/models/v8/yolov8-seg-custom.yaml"
    # model = "cfg/models/v8/yolov8-seg.yaml"
    # data = 'cfg/datasets/coco128-seg-local.yaml'
    data = 'cfg/cocodata/coco-data-seg.yaml'
    # data = 'cfg/cocodata/yolo-data-seg.yaml'
    hype = "cfg/default-hype.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt')
    parser.add_argument('--model', type=str, default=model, help='model.yaml path')
    parser.add_argument('--hype', type=str, default=hype, help='model.yaml path')
    parser.add_argument('--data', type=str, default=data, help='dataset')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    t = Trainer(opt)
    t.run()
