# YOLOv8

这是在`Ultralytics`([YOLOv8]( https://github.com/ultralytics/ultralytics)) 的基础上，增加了模型训练一些特性：

- [x] 支持COCO数据训练
- [ ] 支持VOC数据训练
- [ ] 支持TensorRT推理

## 1.Requirements

- [requirements](requirements.txt),use `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

## 2.Integrations

### (1) 支持COCO数据集训练`cocodataset`

- 修改`DetectionTrainer`中`build_dataset`

```python
from ultralytics.models.yolo.detect.train import DetectionTrainer

gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
if self.data.get('data_type') == "COCO":
    from ultralytics.libs.dataset.cocodataset import build_coco_dataset

    return build_coco_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
else:
    return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
```

## 3.Train

#### (1)object detection

```bash
# object detection
#model="cfg/models/v8/yolov8s.yaml"
#weights="data/model/pretrained/yolov8s.pt"
#data="cfg/cocodata/coco-data-seg.yaml"
#cfg="cfg/detect-hyp.yaml"
python train.py --model $model --weights $weights --data $data --cfg $cfg


```

#### (2)Instance segmentation

```bash
# Instance segmentation
model="cfg/models/v8/yolov8-seg.yaml"
weights="data/model/pretrained/yolov8n-seg.pt"
data="cfg/cocodata/coco-data-seg.yaml"
cfg="cfg/segment-hyp.yaml"
python train.py --model $model --weights $weights --data $data --cfg $cfg

```

## 4.使用说明

- `ultralytics`会根据模型文件名称，来判断模型属于`n,s,m,l,x`，如果判断失败则默认为`n`,见`ultralytics/nn/tasks.py`
    ```yaml
      n: [ 0.33, 0.25, 1024 ]
      s: [ 0.33, 0.50, 1024 ]
      m: [ 0.67, 0.75, 768 ]
      l: [ 1.00, 1.00, 512 ]
      x: [ 1.00, 1.25, 512 ]
    ```

- 关于`自动混合精度`(Automatic Mixed Precision,AMP)

  在`ultralytics.utils.checks`中会检测APM， 如果检查失败，则意味着系统上的AMP存在异常，可能导致NaN丢失或0 map结果，因此在训练期间AMP将被禁用

## 5.常见错误和解决方法

- [常见错误和解决方法](docs/README.md)
- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/tasks/detect/
- https://docs.ultralytics.com/tasks/segment
- https://docs.ultralytics.com/tasks/pose/
- 训练方法： https://github.com/ultralytics/ultralytics/blob/main/docs/modes/train.md
