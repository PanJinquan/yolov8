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

sys.path.insert(0, (os.path.dirname(__file__)))
import argparse
import numpy as np
import cv2
import ultralytics
from easydict import EasyDict
from ultralytics import YOLO
from ultralytics.utils import downloads, ROOT, DEFAULT_CFG_PATH
from ultralytics.models.yolo.detect import DetectionTrainer  # build_dataset
from ultralytics import settings
from pybaseutils import image_utils, file_utils, yaml_utils


class Trainer(object):
    def __init__(self, opt):
        """
        GITHUB_ASSET_STEMS =['yolov8n', 'yolov8n6', 'yolov8n-cls', 'yolov8n-seg', 'yolov8n-pose',
          'yolov8s', 'yolov8s6', 'yolov8s-cls', 'yolov8s-seg', 'yolov8s-pose', 'yolov8m', 'yolov8m6',
          'yolov8m-cls', 'yolov8m-seg', 'yolov8m-pose', 'yolov8l', 'yolov8l6', 'yolov8l-cls',
          'yolov8l-seg', 'yolov8l-pose', 'yolov8x', 'yolov8x6', 'yolov8x-cls', 'yolov8x-seg', 'yolov8x-pose',
          'yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu', 'yolov5xu', 'yolov3u', 'yolov3-sppu', 'yolov3-tinyu',
          'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l', 'sam_b', 'sam_l', 'FastSAM-s', 'FastSAM-x',
          'rtdetr-l', 'rtdetr-x', 'mobile_sam']
        :param opt:
        """
        self.opt = EasyDict(opt.__dict__)
        self.device = str(self.opt.device).split(",")
        self.device = [int(i) for i in self.device]
        self.hyper_params: dict = yaml_utils.load_config(self.opt.cfg)
        self.hyper_params.update({
            "data": self.opt.data,
            "model": self.opt.model,
            "device": self.device,
            "batch": int(self.opt.batch),
            "epochs": int(self.opt.epochs),
        })
        self.env = {'settings_version': '0.0.4',
                    'datasets_dir': '',
                    'weights_dir': '',
                    'runs_dir': self.opt.output,
                    'uuid': ''
                    }
        settings.update(**self.env)
        self.model = YOLO(self.opt.model).load(self.opt.weights)  # build from YAML and transfer weights
        self.names = self.model.names
        print("settings env          :{}".format(self.env))
        print("ROOT                  :{}".format(ROOT))
        print("DEFAULT_CFG_PATH      :{}".format(DEFAULT_CFG_PATH))
        print("GITHUB_ASSET_STEMS    :{}".format(downloads.GITHUB_ASSETS_STEMS))
        print("model num class       :{}".format(len(self.names)))
        print("parser argument       :{}".format(self.opt))

    def run(self, ):
        """
        model.train(data="config.yaml", epochs=100,  imgsz=640, batch=16, name=task_name, device=[0,1])
        :return:
        """
        self.model.train(**self.hyper_params)


def parse_opt():
    """
    DDP多卡训练时，Argument命令行参数会失效
    :return:
    """
    model = "cfg/models/v8/yolov8-seg.yaml"
    weights = "data/model/pretrained/yolov8n-seg.pt"
    # data = "cfg/datasets/coco-data-seg.yaml"
    data = "cfg/datasets/coco-aije-seg.yaml"
    # data = "cfg/datasets/coco-data-seg-local.yaml"
    cfg = "cfg/segment-hyp.yaml"
    #
    # model = "cfg/models/v8/yolov8s.yaml"
    # weights = "data/model/pretrained/yolov8s.pt"
    # data = "cfg/datasets/coco-data-seg.yaml"
    # data = "cfg/datasets/voc-data-det.yaml"
    # cfg = "cfg/detect-hyp.yaml"

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--model', type=str, default=model, help='model *.yaml file')
    parser.add_argument('--weights', type=str, default=weights, help='model weights file')
    parser.add_argument('--data', type=str, default=data, help='dta *.yaml file')
    parser.add_argument('--cfg', type=str, default=cfg, help='cfg hyp file')
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs to train for')
    parser.add_argument('--device', default="3,4,5,6", type=str, help='GPU ID,--device=0,1,2')
    parser.add_argument('--workers', default=8, type=int, help='number of worker threads')
    parser.add_argument('--output', type=str, default="output", help='output')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    t = Trainer(opt)
    t.run()
