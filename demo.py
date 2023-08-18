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
from ultralytics import YOLO
from ultralytics.engine.results import Results
from PIL import Image
from pybaseutils import image_utils, file_utils, color_utils


class YOLOv8(object):
    def __init__(self, weights):
        # 加载模型
        self.model = YOLO(weights)  # load an official model
        self.names = self.model.names
        print("weights   :{}".format(weights))
        print("name      :{}".format(self.names))

    def predict_segment(self, image, vis=True):
        """
        :param image:
        :return:
        """
        # from ultralytics.models.yolo.segment.predict import SegmentationPredictor# postprocess
        results = self.model.predict(source=image, save=False)
        if vis: self.show_result(results)
        return results

    def detect_image_dir(self, image_dir, out_dir=None, vis=True):
        # Dataloader
        dataset = file_utils.get_files_lists(image_dir, shuffle=True)
        # Run inference
        for file in dataset:
            image = cv2.imread(file)  # BGR
            # image = Image.open(file)
            results = self.predict_segment(image=image)
            # from ndarray
            # im2 = cv2.imread("bus.jpg")
            # results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

            # from list of PIL/ndarray
            # results = model.predict(source=[im1, im2])

    def show_result(self, results: Results):
        for r in results:
            image = r.orig_img
            h, w = image.shape[:2]
            boxes = np.asarray(r.boxes.xyxy.cpu().numpy(), dtype=np.float32)
            label = np.asarray(r.boxes.cls.cpu().numpy(), dtype=np.int32)
            masks = np.asarray(r.masks.masks.cpu().numpy(), dtype=np.int32)
            segms = r.masks.segments
            for i in range(len(label)):
                masks[i, :, :] = masks[i, :, :] * (label[i] + 1)
                segms[i] = [np.asarray(segms[i] * (w, h), dtype=np.int32)]
            mask = np.asarray(np.max(masks, axis=0), dtype=np.uint8)
            mask = image_utils.resize_image(mask, size=(w, h), interpolation=cv2.INTER_NEAREST)
            self.show_target_mask_image(image, mask, boxes, label, class_name=self.names)
            # self.show_target_segments_image(image, segms, boxes, label, class_name=self.names)
        return results

    @staticmethod
    def show_target_mask_image(image, mask, boxes, labels, class_name=[], thickness=2, fontScale=1.0):
        mask = np.asarray(mask, np.uint8)
        color_image, color_mask = color_utils.decode_color_image_mask(image, mask)
        color_image = image_utils.draw_image_bboxes_labels(color_image, boxes, labels, class_name=class_name,
                                                           thickness=thickness, fontScale=fontScale)
        vis_image = image_utils.image_hstack([image, mask, color_image, color_mask])
        image_utils.cv_show_image("image", color_image, delay=10)
        image_utils.cv_show_image("vis_image ", vis_image)
        return vis_image

    @staticmethod
    def show_target_segments_image(image, segms, boxes, labels, class_name=[], thickness=2, fontScale=1.0):
        color_image = image_utils.draw_image_contours(image, segms, thickness=2)
        color_image = image_utils.draw_image_bboxes_labels(color_image, boxes, labels, class_name=class_name,
                                                           thickness=thickness, fontScale=fontScale)
        image_utils.cv_show_image("image", color_image)
        return image


def parse_opt():
    image_dir = 'data/test_image'
    weights = "data/model/pretrained/yolov8n-seg.pt"  # 模型文件yolov5s05_640
    out_dir = image_dir + "_result"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt')
    parser.add_argument('--image_dir', type=str, default=image_dir, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--out_dir', type=str, default=out_dir, help='save det result image')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    d = YOLOv8(weights=opt.weights)
    d.detect_image_dir(opt.image_dir, opt.out_dir)
