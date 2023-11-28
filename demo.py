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
from pybaseutils import image_utils, file_utils, color_utils, time_utils


class YOLOv8(object):
    def __init__(self, weights: str, use_onnx=False, use_trt=False):
        """
        :param weights:
        :param use_onnx:
        :param use_trt: 预处理中使用了LetterBox进行自适应缩放图像，如果使用TRT推理，YOLOv8只适配了(640,640)
                        因此使用TRT推理效果变差
        """
        use_torch = weights.endswith(".pt") or weights.endswith(".pth")
        self.model = YOLO(weights)  # load an  model
        if use_torch: self.export_onnx(use_onnx=use_onnx, use_trt=use_trt)
        self.names = self.model.names
        print("weights   :{}".format(weights))
        print("name      :{}".format(self.names))

    def export_onnx(self, use_onnx=False, use_trt=False):
        """
        https://docs.ultralytics.com/modes/export/
        :return:
        """
        file = ""
        if use_onnx:
            self.model.export(format='onnx', simplify=True, dynamic=True)  # ONNX
            file = str(self.model.ckpt_path).replace(".pth", ".onnx").replace(".pt", ".onnx")
        if use_trt:
            self.model.export(format='engine', simplify=True, dynamic=True, half=False, device=0)  # TensorRT
            file = str(self.model.ckpt_path).replace(".pth", ".engine").replace(".pt", ".engine")
        return file

    def inference(self, image, vis=False):
        """
        :param image:
        :return:
        """
        # from ultralytics.models.yolo.segment.predict import SegmentationPredictor# postprocess
        # import ultralytics.engine.predictor
        results = self.model.predict(source=image, save=False)
        results = self.post_process(results)
        if vis: self.draw_result(results)
        return results

    def post_process(self, results):
        outs = []
        for r in results:
            image = r.orig_img
            h, w = image.shape[:2]
            boxes = np.asarray(r.boxes.xyxy.cpu().numpy(), dtype=np.float32)
            label = np.asarray(r.boxes.cls.cpu().numpy(), dtype=np.int32)
            score = np.asarray(r.boxes.conf.cpu().numpy(), dtype=np.float32)
            if r.masks is None:
                # masks = np.zeros(shape=(1, h, w), dtype=np.uint8)
                masks = None
                segms = []
            else:
                masks = np.asarray(r.masks.data.cpu().numpy(), dtype=np.int32)
                # segms = r.masks.segments
                segms = r.masks.xyn  # 归一化
                for i in range(len(label)):
                    masks[i, :, :] = masks[i, :, :] * (label[i] + 1)
                    # segms[i] = [np.asarray(segms[i] * (w, h), dtype=np.int32)]
            outs.append({"boxes": boxes, "label": label, "score": score, "segms": segms,
                         "masks": masks, "size": [w, h], "image": image,
                         })
        return outs

    def detect_image_dir(self, image_dir, out_dir=None, vis=True):
        # Dataloader
        dataset = file_utils.get_files_lists(image_dir, shuffle=False)
        # Run inference
        for file in dataset:
            image = cv2.imread(file)  # BGR
            # image = Image.open(file)
            results = self.inference(image=image, vis=vis)
            # from ndarray
            # im2 = cv2.imread("bus.jpg")
            # results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

            # from list of PIL/ndarray
            # results = model.predict(source=[im1, im2])

    def draw_result(self, results, vis=True):
        import copy
        results_ = copy.deepcopy(results)  # fix a bug: 避免对原始数据修改
        for r in results_:
            image = r['image']
            boxes = r['boxes']
            label = r['label']
            score = r['score']
            masks = r['masks']
            segms = r['segms']
            w, h = r["size"]
            if masks is None:
                self.draw_dets_result(image, boxes, score, label, class_name=self.names, vis=vis)
            else:
                # segms = r.masks.segments
                for i in range(len(segms)):
                    segms[i] = [np.asarray(segms[i] * (w, h), dtype=np.int32)]
                mask = np.asarray(np.max(masks, axis=0), dtype=np.uint8)
                mask = image_utils.resize_image(mask, size=(w, h), interpolation=cv2.INTER_NEAREST)
                self.draw_mask_result(image, mask, boxes, score, label, class_name=self.names, vis=vis)
                # self.draw_segs_result(image, segms, boxes, score, label, class_name=self.names, vis=vis)
        return results

    @staticmethod
    def draw_dets_result(image, boxes, score, labels, class_name=[], thickness=2, fontScale=1.0, vis=True):
        image = image_utils.draw_image_detection_boxes(image, boxes, score, labels, class_name=class_name,
                                                       thickness=thickness, fontScale=fontScale, drawType="chinese")
        if vis: image_utils.cv_show_image("image", image, delay=0)
        return image

    @staticmethod
    def draw_mask_result(image, mask, boxes, score, labels, class_name=[], thickness=2, fontScale=1.0, vis=True):
        color_image, color_mask = color_utils.decode_color_image_mask(image, mask)
        color_image = image_utils.draw_image_detection_boxes(color_image, boxes, score, labels, class_name=class_name,
                                                             thickness=thickness, fontScale=fontScale,
                                                             drawType="chinese")
        vis_image = image_utils.image_hstack([image, mask, color_image, color_mask])
        if vis:
            image_utils.cv_show_image("image", color_image, delay=10)
            image_utils.cv_show_image("vis_image ", vis_image)
        return vis_image

    @staticmethod
    def draw_segs_result(image, segms, boxes, score, labels, class_name=[], thickness=2, fontScale=1.0, vis=True):
        color_image = image_utils.draw_image_contours(image, segms, thickness=2)
        color_image = image_utils.draw_image_bboxes_labels(color_image, boxes, labels, class_name=class_name,
                                                           thickness=thickness, fontScale=fontScale, drawType="chinese")
        if vis: image_utils.cv_show_image("image", color_image)
        return image

    def performance(self, file="data/test.jpg", iterate=100):
        image = cv2.imread(file)  # BGR
        results = self.inference(image=image)
        for i in range(iterate):
            with time_utils.Performance("Performance"):
                results = self.inference(image=image.copy(), vis=False)


def parse_opt():
    image_dir = 'data/test_image'
    image_dir = '/media/PKing/新加卷1/SDK/base-utils/data/coco/JPEGImages'
    # weights = "data/model/pretrained/yolov8n-seg.pt"
    weights = "output/segment/outdoor/yolov8m-seg-v3/weights/best.pt"
    # weights = "output/detect/train/weights/best.engine"

    # AIJE
    image_dir = '/home/PKing/nasdata/dataset-dmai/AIJE/bug'
    weights = "output/segment/train2/weights/best.pt"
    image_dir = '/home/PKing/nasdata/dataset-dmai/AIJE/test_data/室外/badcase数据/badcase对应原始数据/hand_helmet'
    # image_dir = "data/bus.jpg"
    weights = "/home/PKing/nasdata/release/AIJE/aije-algorithm/aije-equipment-detection/app/infercore/resource/outdoor_yolov8m_equipment.pt"
    # save images
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
    # d.performance()
