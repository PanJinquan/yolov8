## 常见错误和解决方法

- ConnectionResetError: [Errno 104] Connection reset by peer
  ![](docs/img001.png)
  > 这是由于内存不足导致的异常，调小batch(batch_size)即可解决
  > 

- RuntimeError: shape '[32, 67, -1]' is invalid for input of size 268800
  ![](docs/img002.png)
  > 配置文件YOLO task没有对应<br/>
  > 检测任务`task: detect`<br/>
  > 分割任务`task: segment`<br/>
  > 分类任务`task: classify`<br/>
  > pose任务`task: pose`<br/>