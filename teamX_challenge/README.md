# vision_node.cpp识别程序
项目简介：对球形、矩形、移动矩形、装甲板进行识别，对于不同的模型发送规定的点坐标，同时在窗口上进行绘制数字和边框，识别装甲板上的数字
# 使用说明
运行节点的代码：
```
ros2 launch teamX_challenge vision.launch
```
# 雨雀
雨雀链接：https://www.yuque.com/ggb0nd/zazw9y/vlip9op72mogwr1y?singleDoc# 《蓝幕战队技术文档》

VisionNode 是一个 ROS 2 节点，旨在通过图像处理与深度学习模型检测装甲板、矩形、球体等目标，并通过灯条估算物体距离。

该节点基于 YOLOv5（ONNX 格式）进行目标检测，并结合灯条的几何特性，估算目标的距离。支持矩形目标、球体目标的检测，并通过视觉信息发布检测结果。

# 特性

YOLOv5 模型检测：利用 YOLOv5 模型进行物体检测，支持自定义的目标分类。

目标类型检测：支持装甲板、球体、矩形等目标的检测。

距离估算：基于目标灯条长度估算与摄像头之间的距离。

目标排序：基于距离对目标进行排序，按从近到远的顺序发布。

ROS 2 集成：直接集成到 ROS 2 系统中，发布检测结果。



# 安装依赖
安装 OpenCV

你可以通过 ROS 2 的安装方式或者手动安装 OpenCV：
```
sudo apt install ros-<ros2-distro>-opencv
```
安装 ONNX Runtime

请参考官方文档安装 ONNX Runtime：

pip install onnxruntime

安装其他依赖
```
sudo apt install ros-<ros2-distro>-cv-bridge
```
编译工作空间

返回到工作空间根目录并编译：
```
cd ~/ros2_ws
colcon build
```


安装模型文件

确保将 YOLOv5 模型文件放置在指定路径：
```
/home/your-username/vision-node/models/best.onnx
```
配置参数

你可以通过 ROS 2 参数进行配置，以下是节点支持的参数：

话题参数：
```
camera_topic (默认: /camera/image_raw)：摄像头图像话题名称。

output_topic (默认: /vision/target)：发布检测结果的目标话题。

show_debug_window (默认: true)：是否显示调试窗口。

debug_window_name (默认: Detection Result)：调试窗口名称。

YOLOv5 参数：

yolo_model_path (默认: /home/your-username/vision-node/models/best.onnx)：YOLOv5 模型文件路径。

yolo_confidence_threshold (默认: 0.25)：YOLOv5 检测的置信度阈值。

yolo_iou_threshold (默认: 0.45)：YOLOv5 NMS（非最大抑制）的 IoU 阈值。

距离估算参数：

distance_scale_factor (默认: 1000.0)：用于估算距离的缩放因子。

reference_lightbar_length (默认: 30.0)：参考灯条的长度（像素）。

reference_distance (默认: 3.0)：参考距离（米）。

min_detection_distance (默认: 0.5)：最小检测距离（米）。

max_detection_distance (默认: 10.0)：最大检测距离（米）。

其他目标检测参数：

lightbar_min_area (默认: 10.0)：灯条最小面积。

lightbar_max_area (默认: 1000.0)：灯条最大面积。

lightbar_min_aspect_ratio (默认: 0.5)：灯条最小长宽比。

lightbar_max_aspect_ratio (默认: 80.0)：灯条最大长宽比。

形态学操作参数：

morph_kernel_size (默认: 3)：形态学操作的核大小。

球体和矩形检测参数：

sphere_min_area (默认: 500)：球体的最小面积。

sphere_min_radius (默认: 15)：球体的最小半径。

sphere_max_radius (默认: 200)：球体的最大半径。

sphere_min_circularity (默认: 0.7)：球体的最小圆度。

rect_min_area (默认: 500)：矩形的最小面积。

rect_max_area (默认: 50000)：矩形的最大面积。
```
运行

启动节点并开始图像检测：
```
ros2 run vision_node vision_node
```

该节点会订阅指定的摄像头图像话题，执行 YOLOv5 检测，并将目标检测结果通过 ROS 消息发布到指定的输出话题。

调试

可以通过参数 show_debug_window 控制是否显示图像处理的调试窗口，便于可视化检测结果。

结果

MultiObject 消息：检测到的目标（如装甲板、矩形、球体）将通过 MultiObject 消息发布，其中包含每个目标的类型和四个角点的位置。


