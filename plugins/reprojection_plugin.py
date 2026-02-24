# plugins/reprojection_plugin.py
import os
import yaml
import numpy as np
from tkinter import filedialog, simpledialog, messagebox
from plugin_core import RosBagPluginBase
from reprojection_viewer import ReprojectionViewer

class ReprojectionPlugin(RosBagPluginBase):
    def __init__(self, context):
        super().__init__(context)
        self.viewer_window = None

    def get_name(self) -> str:
        return "反投影分析"

    def get_button_style(self) -> str:
        return "success"

    def _auto_detect_topics(self):
        # 将你原来的自动检测逻辑移到这里
        image_candidates = []
        lidar_candidates = []
        for topic_name, info in self.context.topic_info.items():
            msg_type = info.msg_type
            if msg_type in ["sensor_msgs/Image", "sensor_msgs/CompressedImage"]:
                image_candidates.append(topic_name)
            elif "PointCloud2" in msg_type or "livox" in msg_type or "lidar" in topic_name:
                lidar_candidates.append(topic_name)
        return (image_candidates[0] if image_candidates else None, 
                lidar_candidates[0] if lidar_candidates else None)

    def on_start(self):
        # 基本上是你原来的 _open_reprojection_viewer 逻辑
        if self.viewer_window and self.viewer_window.winfo_exists():
            self.viewer_window.lift()
            return

        bag_path = self.context.bag_file_path
        expected_calib_path = os.path.splitext(bag_path)[0] + ".yaml"
        if os.path.exists(expected_calib_path):
            calib_path = expected_calib_path
        else:
            calib_path = filedialog.askopenfilename(
                title="未找到同名标定文件，请手动选择", filetypes=[("YAML files", "*.yaml")]
            )
        if not calib_path: return

        try:
            with open(calib_path, 'r') as f: calib = yaml.safe_load(f)
            K = np.array(calib['camera_matrix']['data']).reshape(3, 3)
            dist = np.array(calib['distortion_coefficients']['data'])
            T_cam_lidar = np.array(calib['T_cam_lidar']['data']).reshape(4, 4)
        except Exception as e:
            messagebox.showerror("错误", f"解析标定文件失败: {e}")
            return

        img_topic, lidar_topic = self._auto_detect_topics()
        
        # 为了简洁，省略了弹窗重新输入的逻辑，你可以直接把原代码的输入框逻辑贴过来
        if not img_topic or not lidar_topic:
            messagebox.showwarning("提示", "未能检测到完整的话题。")
            return

        self.viewer_window = ReprojectionViewer(self.context.master)
        self.viewer_window.configure_data_sources(bag_path, img_topic, lidar_topic, K, T_cam_lidar, dist)
        
        current_index = self.context.get_current_index()
        if current_index >= 0:
            self.viewer_window.update_view_by_index(current_index)

    def on_frame_changed(self, index: int, is_high_quality: bool) -> bool:
        # 当滑动条拖动时，主界面会调用这里
        if self.viewer_window and self.viewer_window.winfo_exists() and self.viewer_window.state() == 'normal':
            self.viewer_window.update_view_by_index(index, high_quality=is_high_quality)
            return True # 告诉主界面：我处理了，你别去解析那个庞大的点云字符串了
        return False