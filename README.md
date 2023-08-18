# YOLOv8

- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/tasks/detect/
- https://docs.ultralytics.com/tasks/segment
- https://docs.ultralytics.com/tasks/pose/

## install

- pip install ultralytics

## 修改读取配置文件的根目录

在

```python
from ultralytics.utils import DEFAULT_CFG_PATH
```

修改

```python
# ROOT = FILE.parents[1]  # YOLO
ROOT = Path(os.getcwd())
print("ROOT{}".format(ROOT))
```

## 增加数据集

在

```python
from ultralytics.models.yolo.detect import DetectionTrainer
```

# 修改为

```python
...
from libs.dataset.cocodataset import build_coco_dataset

# return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
return build_coco_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
```