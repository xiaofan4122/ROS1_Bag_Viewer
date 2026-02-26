# ROS1 Bag Viewer

一个基于 Python + Tkinter 的 ROS1 Bag 文件可视化工具，支持消息浏览、数据绘图、点云降采样分析以及 LiDAR-Camera 反投影可视化。

---

## 功能特性

- **消息浏览**：加载 `.bag` 文件后自动索引所有话题，支持滑动条逐帧浏览消息内容
- **缓存加速**：首次加载自动生成 `.index` / `.cache` 二进制缓存，后续打开瞬间完成
- **动态消息解析**：通过 `genpy.dynamic` 从 bag 内嵌定义动态生成消息类，无需安装对应 ROS 包（如 `livox_ros_driver`）
- **插件系统**：可扩展的插件架构，当前内置三个插件
- **历史记录**：记忆最近打开的 bag 文件路径，方便快速重新加载

---

## 内置插件

### 反投影分析 (`reprojection_plugin`)
将 LiDAR 点云投影到相机图像上，直观验证外参标定结果。

- 弹出标定参数配置对话框，支持手动填写内参矩阵 K、畸变系数、外参矩阵 T_cam_lidar
- **命名保存**：强制要求为每组参数命名，自动持久化到 `plugins/.calib_history.json`
- **历史记忆**：左侧列表显示历史配置，点击一键填入，支持删除
- 自动检测 bag 中的图像话题和点云话题
- 查看器窗口功能：
  - 左侧：反投影结果（点云深度着色投影到图像）
  - 右上：原始图像
  - 右下：3D 点云视图（雷达坐标系）
  - 底部滚动条独立控制帧，与主窗口解耦
  - 键盘左右键逐帧切换
  - **相机帧偏移**：可将相机帧向前/向后偏移 N 帧，用于时间戳对齐
  - **时间戳显示**：实时显示相机帧和 LiDAR 帧的时间戳及差值（ms）
  - 右键菜单：保存当前帧反投影图像、保存当前帧点云（`.pcd`）、导出全部帧为视频

### 数据绘图 (`data_plotter_plugin`)
通用话题数据曲线绘图器，基于 matplotlib，支持交互式图表。

### 体素降采样测试 (`voxel_test_plugin`)
点云降采样方案测试，支持目标点数降采样、固定体素大小等多种方案，利用多进程并行计算。

---

## 环境依赖

- **ROS Noetic**（需要 source 环境）
- Python 3
- `ttkbootstrap`
- `numpy`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `cv_bridge`
- `genpy`
- `rosbag`

安装 Python 依赖：

```bash
pip install ttkbootstrap numpy opencv-python Pillow matplotlib
```

---

## 快速开始

```bash
bash run_viewer.sh
# 或
zsh run_viewer.sh
```

脚本会自动完成以下步骤：
1. 退出当前 conda 环境（如有）
2. Source ROS Noetic 环境
3. 检测并启动 roscore（如未运行）
4. 启动主程序

---

## 项目结构

```
ROS1_Bag_Viewer/
├── rosbag_viewer.py          # 主程序入口
├── reprojection_viewer.py    # 反投影查看器（独立窗口）
├── bag_cache_reader.py       # 缓存读取器（可独立使用）
├── plugin_core.py            # 插件基类与上下文定义
├── DataPlotter.py            # 通用绘图组件
├── run_viewer.sh             # 启动脚本（bash/zsh 兼容）
└── plugins/
    ├── reprojection_plugin.py    # 反投影插件
    ├── data_plotter_plugin.py    # 数据绘图插件
    ├── voxel_test_plugin.py      # 体素测试插件
    └── .calib_history.json       # 标定参数历史（自动生成）
```

---

## 插件开发

继承 `plugin_core.RosBagPluginBase`，实现以下接口即可：

```python
from plugin_core import RosBagPluginBase

class MyPlugin(RosBagPluginBase):
    def get_name(self) -> str:
        return "我的插件"

    def get_button_style(self) -> str:
        return "primary"  # ttkbootstrap 按钮样式

    def on_start(self):
        # 点击插件按钮时触发
        pass

    def on_frame_changed(self, index: int, is_high_quality: bool) -> bool:
        # 主窗口滑动条变化时触发
        # 返回 True 表示插件接管渲染，主窗口跳过默认解析
        return False
```

在 `rosbag_viewer.py` 中导入并注册插件即可加入到主界面。

---

## 独立使用缓存读取器

`BagCacheReader` 可脱离 GUI 独立使用，适合写脚本批量处理数据：

```python
from bag_cache_reader import BagCacheReader

reader = BagCacheReader('/path/to/your.bag')
reader.load_topic('/livox/lidar')

for i in range(reader.get_message_count()):
    msg, timestamp = reader.get_message(i)
    print(timestamp.to_sec(), msg)
```

> **注意**：使用前需先通过 GUI 程序对该 bag 文件完成索引缓存的生成。
