# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2024-02-04 17:21:01
    @Brief  :
"""
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


def test_obb():
    """
    oriented bounding box
    数据标注：https://zhuanlan.zhihu.com/p/430850089?utm_id=0
    :return:
    """
    # Load a model
    model = YOLO('yolov8n-obb.pt')  # load an official model
    # model = YOLO('path/to/best.pt')  # load a custom model

    # Predict with the model
    results = model.predict('data/ship.jpg',save=True)  # predict on an image
    return results

def train_obb():
    from ultralytics import YOLO
    # Load a model
    # model = YOLO('yolov8n-obb.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='cfg/datasets/dota8.yaml', epochs=100, imgsz=640)
    return results


def test_det():
    # Load a model
    model = YOLO('yolov8n.pt')
    # Predict with the model
    results = model.predict('data/bus.jpg',save=True)  # predict on an image
    return results



if __name__ == "__main__":
    test_obb()
    # train_obb()
    # test_det()
