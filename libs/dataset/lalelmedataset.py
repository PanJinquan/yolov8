# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-15 19:30:41
    @Brief  :
"""
import os
import numpy as np
from ultralytics.utils import RANK, colorstr
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import HELP_URL, LOGGER
from pybaseutils import image_utils, file_utils, json_utils, yaml_utils
from pybaseutils.dataloader import parser_labelme


def build_labelme_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build VOC Dataset."""
    return LabelmeDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


class LabelmeDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8

    def __init__(self, *args, data=None, task="detect", **kwargs):
        self.unique = False
        self.use_obb = task == "obb"
        self.data = data
        # self.class_dict = {n: i for i, n in data['names'].items()}
        self.class_dict = self.parser_classes(data['names'])
        # image_dir = os.path.join(os.path.dirname(kwargs['img_path']), "person")
        anno_dir = kwargs["img_path"]  # 图片目录和标注文件(*.json同目录)
        image_dir = kwargs["img_path"]
        self.data_parser = parser_labelme.LabelMeDatasets(anno_dir=anno_dir,
                                                          image_dir=image_dir,
                                                          class_name=self.class_dict,
                                                          check=False)
        super().__init__(*args, data=data, task=task, **kwargs)

    def parser_classes(self, names: dict):
        class_dict = {}
        for i, ns in names.items():
            ns = ns.split(",")
            for n in ns:
                class_dict[n] = i
        return class_dict

    def get_img_files(self, *args, **kwargs):
        """Read image files. 返回空即可"""
        file_list = []
        # for i in range(len(self.voc.image_ids)):
        #     image_file, annotation_file = self.voc.get_image_anno_file(i)
        #     file_list.append(image_file)
        return file_list

    def get_labels(self):
        """
        Returns dictionary of labels for YOLO training.
        bbox_format:
               - `xyxy` means left top and right bottom
               - `xywh` means center x, center y and width, height(yolo format)
               - `ltwh` means left top and width, height(coco format)
        """
        labels = []
        for i in range(len(self.data_parser.image_ids)):
            data_info = self.data_parser.__getitem__(i)
            im_file = data_info['image_file']
            w, h = data_info['size']
            boxes = np.array(data_info["boxes"])
            cls = np.array(data_info["labels"])
            segs = data_info["points"]
            if len(boxes) == 0: continue  # 是否要去除无目标的数据，不去除好像也没有报错
            # 将(x,y,x,y)转为(x_center y_center width height)，见ultralytics.utils.instance
            cxcywh = image_utils.xyxy2cxcywh(boxes) / (w, h, w, h)
            cls = cls.reshape(-1, 1)
            if self.use_obb: segs = image_utils.find_minAreaRect(segs)
            segs = [s / (w, h) for s in segs]
            item = {
                "im_file": im_file,
                "shape": (h, w),
                "cls": cls,
                "bboxes": cxcywh,
                "segments": segs,
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",  # YOLO中xywh指的是(x_center y_center width height)
            }
            labels.append(item)

        assert len(labels), f'No valid labels found, please check your dataset. {HELP_URL}'
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        return labels
