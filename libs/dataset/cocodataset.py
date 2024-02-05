# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-08-15 19:30:41
    @Brief  :
"""
import os
from ultralytics.utils import RANK, colorstr
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import HELP_URL, LOGGER
from pybaseutils import image_utils, file_utils, json_utils, yaml_utils
from pybaseutils.dataloader import parser_coco_ins


def build_coco_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build COCO Dataset."""
    return COCODataset(
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


class COCODataset(YOLODataset):
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
        anno_file = kwargs["img_path"]
        self.coco = parser_coco_ins.CocoInstances(anno_file, image_dir=None, class_name=self.class_dict, decode=False)
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
        # files_info = self.coco.get_files_info()
        # for data in files_info:
        #     file = os.path.join(self.coco.image_dir, data['file_name'])
        #     file_list.append(file)
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
        # for files_info, annos_info in zip(self.coco.files_info, self.coco.annos_info):
        for i in range(len(self.coco.image_ids)):
            data_info = self.coco.__getitem__(i)
            im_file = data_info["image_file"]
            w, h = data_info['size']
            # boxes, cls, mask, segs = self.coco.get_object_instance(annos_info, h, w, decode=False)
            boxes, cls, mask, segs = data_info['boxes'], data_info['label'], data_info['mask'], data_info['segs']
            # 将(x,y,x,y)转为(x_center y_center width height)，见ultralytics.utils.instance
            cxcywh = image_utils.xyxy2cxcywh(boxes) / (w, h, w, h)
            cls = cls.reshape(-1, 1)
            segs = [s[0] for s in segs]
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
