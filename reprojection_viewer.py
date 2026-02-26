# --- [新增] --- 导入所需模块
import os
import shutil
import glob
from typing import Iterable, Tuple, Optional
import tkinter as tk
from tkinter import messagebox, filedialog # 导入文件对话框和消息框

import ttkbootstrap as ttk
from ttkbootstrap.scrolled import ScrolledText
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
# Matplotlib 用于在Tkinter中嵌入3D绘图
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as mtransforms
from matplotlib import cm, colors

from bag_cache_reader import BagCacheReader

# --- [新增] --- 导入并行和线程相关模块
import concurrent.futures
import threading
import queue
import multiprocessing

# --- [新增] 独立的、用于并行处理的单帧渲染函数 ---
def render_single_frame(args):
    """
    此函数将在一个独立的子进程中执行。
    它负责读取特定索引的数据，渲染图像，并返回结果。
    """
    # 1. 从传入的参数中解包
    index, bag_path, image_topic, lidar_topic, K, T_cam_lidar, dist_coeffs = args

    try:
        # 2. 每个子进程都必须创建自己的数据读取器实例
        image_reader = BagCacheReader(bag_path)
        image_reader.load_topic(image_topic)
        lidar_reader = BagCacheReader(bag_path)
        lidar_reader.load_topic(lidar_topic)

        # 3. 获取数据 (这部分逻辑与原 update_view_by_index 类似)
        image_msg, _ = image_reader.get_message(index)
        if image_msg._type == 'sensor_msgs/CompressedImage':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            # 假设有一个 bridge 的实例，实际应用中可能需要更复杂的处理
            # 这里简化为直接调用一个假设的转换函数
            from cv_bridge import CvBridge
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        lidar_msg, _ = lidar_reader.get_message(index)
        points_lidar = np.array([[p.x, p.y, p.z] for p in lidar_msg.points])

        # 4. 执行反投影渲染 (这部分逻辑与原 _update_reprojection 完全相同)
        reprojection_img = cv_image.copy()
        if points_lidar is None or len(points_lidar) == 0:
            return index, reprojection_img  # 返回索引和图像

        # ... (这里省略了您原有的、完整的反投影计算代码，直接复制粘贴即可) ...
        h, w = cv_image.shape[:2]
        R_cam_lidar = T_cam_lidar[:3, :3].astype(np.float32)
        t_cam_lidar = T_cam_lidar[:3, 3].astype(np.float32)
        rvec, _ = cv2.Rodrigues(R_cam_lidar)
        pts = points_lidar.astype(np.float32)
        proj, _ = cv2.projectPoints(pts, rvec, t_cam_lidar, K.astype(np.float32), dist_coeffs.astype(np.float32))
        proj = proj.reshape(-1, 2)
        pts_cam = (R_cam_lidar @ pts.T).T + t_cam_lidar
        depth = pts_cam[:, 2]
        min_z, max_z = 0.2, 200.0
        u, v = proj[:, 0], proj[:, 1]
        mask = (depth > min_z) & (depth < max_z) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(mask):
            return index, reprojection_img
        u, v, z = u[mask], v[mask], depth[mask]
        order = np.argsort(z)
        u, v, z = u[order], v[order], z[order]
        z_pos = z[z > 0]
        if len(z_pos) < 10:
            zmin, zmax = (np.min(z_pos), np.max(z_pos)) if len(z_pos) > 0 else (min_z, max_z)
        else:
            zmin, zmax = np.percentile(z_pos, [5, 95])
        zmin = max(zmin, 1e-3)
        z_for_color = np.log(np.clip(z, zmin, zmax))
        norm = plt.Normalize(vmin=np.log(zmin), vmax=np.log(zmax))
        cmap = plt.get_cmap('viridis')
        cell_size = 2
        ui, vi = (u / cell_size).astype(int), (v / cell_size).astype(int)
        best = {}
        for idx_pt in range(len(z)):
            key = (ui[idx_pt], vi[idx_pt])
            if key not in best: best[key] = idx_pt
        keep_indices = np.fromiter(best.values(), dtype=int)
        u, v, z = u[keep_indices], v[keep_indices], z[keep_indices]
        zc_log = np.log(np.clip(z, zmin, zmax))
        fx = K[0, 0]
        radii = np.clip((0.004 * fx / (z + 1e-6)), 2.0, 6.0).astype(int)
        colors_bgr = [tuple(int(c * 255) for c in cmap(norm(val))[:3][::-1]) for val in zc_log]
        for i in range(len(u) - 1, -1, -1):
            cx, cy = int(round(u[i])), int(round(v[i]))
            cv2.circle(reprojection_img, (cx, cy), radii[i], colors_bgr[i], -1, cv2.LINE_AA)

        # 5. 返回帧索引和渲染好的图像
        return index, reprojection_img

    except Exception as e:
        # 在子进程中打印错误，并返回None，以便主进程知道此帧失败
        print(f"子进程处理帧 {index} 时发生错误: {e}")
        return index, None

# --- 【新】可缩放平移的图像面板 ---
class ZoomableImagePanel(ttk.Frame):
    """一个封装了缩放和平移功能的自定义图像显示控件"""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, background="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._image_id = None
        self._pil_image = None
        self.scale = 1.0

        # 绑定事件
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows & macOS
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux 滚轮向上
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux 滚轮向下
        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)

    def set_image(self, cv_image):
        """设置并显示一个新的OpenCV图像"""
        if cv_image is None:
            if self._image_id:
                self.canvas.delete(self._image_id)
            self._image_id = None
            self._pil_image = None
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self._pil_image = Image.fromarray(rgb_image)
        self.scale = 1.0  # 重置缩放
        self._render_image()

    def _render_image(self):
        """根据当前缩放比例渲染图像"""
        if self._pil_image is None: return

        w, h = self._pil_image.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        # 确保尺寸有效
        if new_w < 1 or new_h < 1:
            return

        resized_img = self._pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(resized_img)

        if self._image_id:
            self.canvas.itemconfig(self._image_id, image=self._photo_image)
        else:
            self._image_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

        # 将画布的滚动区域设置为图片大小，以便平移
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def _on_button_press(self, event):
        """记录平移的起始点"""
        self.canvas.scan_mark(event.x, event.y)

    def _on_mouse_drag(self, event):
        """执行平移"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_mouse_wheel(self, event):
        """执行缩放"""
        scale_factor = 1.1
        # Linux滚轮事件处理
        if event.num == 4:  # 向上滚动
            delta = 1
        elif event.num == 5:  # 向下滚动
            delta = -1
        else:  # Windows/macOS
            delta = event.delta

        if delta > 0:  # 放大
            self.scale *= scale_factor
        else:  # 缩小
            self.scale /= scale_factor

        # 重新渲染以应用缩放
        self._render_image()


class ReprojectionViewer(ttk.Toplevel):
    """
    一个用于可视化3D点云到2D图像反投影的GUI类。
    所有计算均在相机和雷达坐标系下进行。
    """

    def __init__(self, master):
        super().__init__(master)
        self.title("3D-2D 反投影查看器 (雷达系)")
        self.geometry("1600x900")

        # --- 内部数据 & 配置 ---
        self.image_reader = None
        self.lidar_reader = None
        # [新增] 存储数据源路径以传递给子进程
        self.bag_path = None
        self.image_topic = None
        self.lidar_topic = None
        self.K = np.eye(3)
        self.dist_coeffs = np.zeros(5)
        self.T_cam_lidar = np.eye(4)

        # [新增] 用于线程安全通信的队列
        self.export_queue = queue.Queue()

        # UI 组件
        self.image_panel = None
        self.reprojection_panel = None
        self.ax_3d = None
        self.canvas_3d = None
        self.export_button = None
        self._slider_var = None
        self._slider = None
        self._frame_label = None
        self._total_frames = 0
        self._is_dragging = False

        self._create_widgets()

    def _create_widgets(self):
        # ... (与之前版本相同)
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        self.reprojection_frame = ttk.Labelframe(main_pane, text="反投影结果 (点云 -> 图像)", padding=10)
        self.reprojection_panel = ZoomableImagePanel(self.reprojection_frame)
        self.reprojection_panel.pack(fill=tk.BOTH, expand=True)
        main_pane.add(self.reprojection_frame, weight=2)
        right_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane, weight=1)
        self.image_frame = ttk.Labelframe(right_pane, text="原始图像", padding=5)
        self.image_panel = ZoomableImagePanel(self.image_frame)
        self.image_panel.pack(fill=tk.BOTH, expand=True)
        right_pane.add(self.image_frame, weight=1)
        self.plot_frame = ttk.Labelframe(right_pane, text="3D 视图 (雷达坐标系)", padding=5)
        self.fig_3d = Figure(figsize=(5, 4), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.plot_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_3d.mpl_connect('scroll_event', self._on_3d_scroll)
        self.canvas_3d.mpl_connect('motion_notify_event', lambda e: self.canvas_3d.draw_idle() if e.button == 1 else None)

        # 右键菜单
        self._context_menu = tk.Menu(self, tearoff=0)
        self._context_menu.add_command(label="保存当前帧图像", command=self._save_current_image)
        self._context_menu.add_command(label="保存当前帧点云 (.pcd)", command=self._save_current_pcd)
        self._context_menu.add_separator()
        self._context_menu.add_command(label="导出全部帧为视频", command=self.export_video)
        # 只在反投影图像面板和3D视图上绑定右键
        self.reprojection_panel.canvas.bind("<Button-3>", self._show_context_menu)
        self.canvas_3d.get_tk_widget().bind("<Button-3>", self._show_context_menu)

        # --- 底部控制栏：滚动条 + 帧号 + 导出按钮 ---
        self.reprojection_controls = ttk.Frame(self.reprojection_frame)
        self.reprojection_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        self._frame_label = ttk.Label(self.reprojection_controls, text="0 / 0", width=12)
        self._frame_label.pack(side=tk.RIGHT, padx=(0, 4))

        self._slider_var = tk.IntVar(value=0)
        self._slider = ttk.Scale(
            self.reprojection_controls, from_=0, to=0,
            orient=tk.HORIZONTAL, variable=self._slider_var,
            command=self._on_slider_changed
        )
        self._slider.bind("<ButtonPress-1>", lambda e: setattr(self, '_is_dragging', True))
        self._slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.bind("<Left>", self._on_key)
        self.bind("<Right>", self._on_key)

        self.export_button = ttk.Button(
            self.reprojection_controls,
            text="导出视频",
            command=self.export_video,
            bootstyle="success"
        )
        self.export_button.pack(side=tk.RIGHT, padx=(0, 4))

        # 相机帧偏移量
        ttk.Label(self.reprojection_controls, text="相机偏移:").pack(side=tk.RIGHT, padx=(8, 2))
        self._offset_var = tk.IntVar(value=0)
        offset_spin = ttk.Spinbox(
            self.reprojection_controls, from_=-9999, to=9999, width=6,
            textvariable=self._offset_var, command=self._on_offset_changed
        )
        offset_spin.bind("<Return>", lambda e: self._on_offset_changed())
        offset_spin.pack(side=tk.RIGHT, padx=(0, 4))

        # 时间戳显示行
        self._ts_label = ttk.Label(self.reprojection_frame, text="相机: --  |  LiDAR: --", foreground="gray")
        self._ts_label.pack(side=tk.BOTTOM, anchor='w', padx=4, pady=(0, 2))

        right_pane.add(self.plot_frame, weight=1)

    def configure_data_sources(self, bag_path, image_topic, lidar_topic, K, T_cam_lidar, dist_coeffs):
        """配置数据源和标定参数"""
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.lidar_topic = lidar_topic
        try:
            self.image_reader = BagCacheReader(bag_path)
            self.image_reader.load_topic(image_topic)

            self.lidar_reader = BagCacheReader(bag_path)
            self.lidar_reader.load_topic(lidar_topic)

            self.K = K
            self.dist_coeffs = dist_coeffs
            self.T_cam_lidar = T_cam_lidar

            # 设置滚动条范围
            self._total_frames = self.image_reader.get_message_count()
            if self._total_frames > 0:
                self._slider.config(from_=0, to=self._total_frames - 1)
                self._frame_label.config(text=f"1 / {self._total_frames}")

            print("数据源配置成功！")
        except Exception as e:
            messagebox.showerror("配置失败", f"配置数据源时出错: {e}")

    def _on_slider_changed(self, value):
        index = int(float(value))
        self._frame_label.config(text=f"{index + 1} / {self._total_frames}")
        # 拖动时只更新帧号，不触发渲染，松开时由 _on_slider_release 统一处理
        if not self._is_dragging:
            self.update_view_by_index(index, high_quality=True)

    def _on_slider_release(self, _event):
        self._is_dragging = False
        index = int(self._slider_var.get())
        self.update_view_by_index(index, high_quality=True)

    def _on_key(self, event):
        if self._total_frames == 0:
            return
        index = int(self._slider_var.get())
        if event.keysym == 'Left':
            index = max(0, index - 1)
        elif event.keysym == 'Right':
            index = min(self._total_frames - 1, index + 1)
        else:
            return
        self._slider_var.set(index)
        self._frame_label.config(text=f"{index + 1} / {self._total_frames}")
        self.update_view_by_index(index, high_quality=True)

    def _on_offset_changed(self):
        index = int(self._slider_var.get())
        self.update_view_by_index(index, high_quality=True)

    def _show_context_menu(self, event):
        self._context_menu.post(event.x_root, event.y_root)
        self._context_menu.grab_release()
        self.bind("<Button-1>", lambda e: self._context_menu.unpost(), add="+")

    def _save_current_image(self):
        if self.reprojection_panel._pil_image is None:
            messagebox.showwarning("提示", "当前没有可保存的图像喵～")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="保存当前帧图像"
        )
        if not path:
            return
        self.reprojection_panel._pil_image.save(path)
        messagebox.showinfo("完成", f"图像已保存到:\n{path}")

    def _save_current_pcd(self):
        index = int(self._slider_var.get())
        if not self.lidar_reader:
            messagebox.showwarning("提示", "数据源未配置喵～")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pcd",
            filetypes=[("PCD files", "*.pcd")],
            title="保存当前帧点云"
        )
        if not path:
            return
        try:
            lidar_msg, _ = self.lidar_reader.get_message(index)
            if hasattr(lidar_msg, 'points'):
                pts = np.array([[p.x, p.y, p.z] for p in lidar_msg.points], dtype=np.float32)
            else:
                import sensor_msgs.point_cloud2 as pc2
                pts = np.array(list(pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
            n = len(pts)
            with open(path, 'w') as f:
                f.write("# .PCD v0.7 - Point Cloud Data\n")
                f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
                f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {n}\nDATA ascii\n")
                for p in pts:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            messagebox.showinfo("完成", f"点云已保存到:\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存点云失败: {e}")

    def compressed_img_to_cv2(self, msg):
        """将 sensor_msgs/CompressedImage 转换为 cv2 图像"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv_image

    def update_view_by_index(self, index: int, high_quality: bool = True) -> Optional[np.ndarray]:
        """根据给定的消息索引，刷新所有视图，并返回生成的反投影图像"""
        if not (self.image_reader and self.lidar_reader):
            print("警告: 数据源未配置。")
            return None

        try:
            offset = self._offset_var.get() if hasattr(self, '_offset_var') else 0
            img_count = self.image_reader.get_message_count()
            image_index = max(0, min(img_count - 1, index + offset))

            image_msg, img_ts = self.image_reader.get_message(image_index)
            if image_msg._type == 'sensor_msgs/CompressedImage':
                cv_image = self.compressed_img_to_cv2(image_msg)
            else:
                cv_image = self.image_reader.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            lidar_msg, lidar_ts = self.lidar_reader.get_message(index)
            points_lidar = np.array([[p.x, p.y, p.z] for p in lidar_msg.points])

            # 更新时间戳显示
            def fmt_ts(ts):
                try:
                    return f"{ts.to_sec():.6f}s"
                except Exception:
                    return str(ts)
            self._ts_label.config(
                text=f"相机(+{offset}帧): {fmt_ts(img_ts)}  |  LiDAR: {fmt_ts(lidar_ts)}  |  差值: {(img_ts - lidar_ts).to_sec()*1000:.1f}ms"
            )

            # 更新所有面板，并捕获返回的反投影图像
            self._update_original_image(cv_image)
            reprojected_image = self._update_reprojection(cv_image, points_lidar)  # 捕获
            self._update_3d_view(points_lidar, high_quality=high_quality)

            # 返回图像
            return reprojected_image

        except IndexError:
            error_msg = f"索引 {index} 越界，图像和点云话题的消息数量可能不匹配。"
            print(error_msg)
            self.reprojection_panel.set_image(None)
            return None
        except Exception as e:
            error_msg = f"更新视图时出错:\n{e}"
            self.reprojection_panel.set_image(None)
            print(error_msg)
            return None

    def _update_original_image(self, image):
        self.image_panel.set_image(image)

    def _update_reprojection(self, image, points_lidar):
        if image is None:
            self.reprojection_panel.set_text("无图像数据");
            return

        reprojection_img = image.copy()
        if points_lidar is None or len(points_lidar) == 0:
            self.reprojection_panel.set_image(reprojection_img);
            return

        h, w = image.shape[:2]

        # --- 1. 投影 (保持不变) ---
        R_cam_lidar = self.T_cam_lidar[:3, :3].astype(np.float32)
        t_cam_lidar = self.T_cam_lidar[:3, 3].astype(np.float32)
        K = self.K.astype(np.float32);
        dist = self.dist_coeffs.astype(np.float32)
        rvec, _ = cv2.Rodrigues(R_cam_lidar)
        pts = points_lidar.astype(np.float32)
        proj, _ = cv2.projectPoints(pts, rvec, t_cam_lidar, K, dist)
        proj = proj.reshape(-1, 2)
        pts_cam = (R_cam_lidar @ pts.T).T + t_cam_lidar
        depth = pts_cam[:, 2]

        min_z, max_z = 0.2, 200.0
        u, v = proj[:, 0], proj[:, 1]
        mask = (depth > min_z) & (depth < max_z) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

        if not np.any(mask):
            self.reprojection_panel.set_image(reprojection_img);
            return

        u, v, z = u[mask], v[mask], depth[mask]

        # --- 2. 深度排序: 近点在前，远的在后，方便后续绘制 ---
        order = np.argsort(z)  # z越小(越近)的索引在前
        u, v, z = u[order], v[order], z[order]

        # --- 3. 鲁棒色标 (保持不变) ---
        z_pos = z[z > 0]
        if len(z_pos) < 10:
            zmin, zmax = (np.min(z_pos), np.max(z_pos)) if len(z_pos) > 0 else (min_z, max_z)
        else:
            zmin, zmax = np.percentile(z_pos, [5, 95])
        zmin = max(zmin, 1e-3)
        z_for_color = np.log(np.clip(z, zmin, zmax))
        norm = plt.Normalize(vmin=np.log(zmin), vmax=np.log(zmax))
        cmap = plt.get_cmap('viridis')

        # --- 4. 稀疏化 (保持不变) ---
        cell_size = 2;
        ui, vi = (u / cell_size).astype(int), (v / cell_size).astype(int)
        best = {}
        for idx in range(len(z)):
            key = (ui[idx], vi[idx])
            # 因为现在是近的在前，所以只保留每格最前面的点
            if key not in best: best[key] = idx
        keep_indices = np.fromiter(best.values(), dtype=int)
        u, v, z = u[keep_indices], v[keep_indices], z[keep_indices]
        zc_log = np.log(np.clip(z, zmin, zmax))

        # --- 5. 自适应点大小和绘制 (保持不变) ---
        fx = K[0, 0]
        radii = np.clip((0.004 * fx / (z + 1e-6)), 2.0, 6.0).astype(int)
        colors_bgr = [tuple(int(c * 255) for c in cmap(norm(val))[:3][::-1]) for val in zc_log]
        # 先画远的，再画近的，以实现正确的遮挡
        for i in range(len(u) - 1, -1, -1):
            cx, cy = int(round(u[i])), int(round(v[i]))
            cv2.circle(reprojection_img, (cx, cy), radii[i], colors_bgr[i], -1, cv2.LINE_AA)

        # 转换为整数坐标
        u_int, v_int = u.astype(int), v.astype(int)

        self.reprojection_panel.set_image(reprojection_img)

        # 将生成的图像返回
        return reprojection_img

    def _update_3d_view(self, points_lidar, high_quality: bool = True):
        self.ax_3d.clear()

        # 降采样：高质量最多 3000 点，预览最多 800 点
        max_pts = 3000 if high_quality else 800
        if points_lidar is not None and len(points_lidar) > max_pts:
            step = len(points_lidar) // max_pts
            points_to_render = points_lidar[::step, :]
        else:
            points_to_render = points_lidar
        point_size = 1 if high_quality else 0.5

        if points_to_render is not None and len(points_to_render) > 0:
            self.ax_3d.scatter(points_to_render[:, 0], points_to_render[:, 1], points_to_render[:, 2],
                               s=point_size, c=points_to_render[:, 2], cmap='viridis', alpha=0.5)

        # ... (绘制雷达原点和相机视锥的代码与上一版本完全相同)
        T_lidar_cam = np.linalg.inv(self.T_cam_lidar)
        R_lidar_cam, pos_lidar_cam = T_lidar_cam[:3, :3], T_lidar_cam[:3, 3]
        self.ax_3d.scatter(pos_lidar_cam[0], pos_lidar_cam[1], pos_lidar_cam[2], c='red', s=100, marker='o',
                           label="Camera")
        h, w = (480, 640);
        if self.image_panel and self.image_panel._pil_image: h, w = self.image_panel._pil_image.height, self.image_panel._pil_image.width
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        corners_norm = np.array(
            [[(0 - cx) / fx, (0 - cy) / fy, 1], [(w - cx) / fx, (0 - cy) / fy, 1], [(w - cx) / fx, (h - cy) / fy, 1],
             [(0 - cx) / fx, (h - cy) / fy, 1]])
        corners_cam = corners_norm * 5.0
        corners_lidar = (R_lidar_cam @ corners_cam.T).T + pos_lidar_cam
        for i in range(4):
            self.ax_3d.plot([pos_lidar_cam[0], corners_lidar[i, 0]], [pos_lidar_cam[1], corners_lidar[i, 1]],
                            [pos_lidar_cam[2], corners_lidar[i, 2]], 'r-')
        self.ax_3d.plot(np.append(corners_lidar[:, 0], corners_lidar[0, 0]),
                        np.append(corners_lidar[:, 1], corners_lidar[0, 1]),
                        np.append(corners_lidar[:, 2], corners_lidar[0, 2]), 'r-')

        # --- 【最终修正】 使用Matplotlib的推荐方法来设置等比例轴 ---

        # 1. 收集所有需要显示的点来计算边界
        all_points_for_limits = np.vstack([pos_lidar_cam.reshape(1, 3), corners_lidar])
        if points_to_render is not None and len(points_to_render) > 0:
            all_points_for_limits = np.vstack([all_points_for_limits, points_to_render])

        # 2. 获取数据范围
        X, Y, Z = all_points_for_limits[:, 0], all_points_for_limits[:, 1], all_points_for_limits[:, 2]
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5

        # 3. 设置范围，确保是立方体
        self.ax_3d.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        self.ax_3d.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        self.ax_3d.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        self.ax_3d.set_xlabel('X (Lidar)');
        self.ax_3d.set_ylabel('Y (Lidar)');
        self.ax_3d.set_zlabel('Z (Lidar)')
        self.ax_3d.set_xticklabels([]);
        self.ax_3d.set_yticklabels([]);
        self.ax_3d.set_zticklabels([])

        self.canvas_3d.draw_idle()

    def _on_3d_scroll(self, event):
        scale_factor = 0.9 if event.button == 'up' else 1.1
        self.ax_3d.dist *= scale_factor
        self.canvas_3d.draw_idle()

    # --- [新增] --- 导出视频的主方法
    def export_video(self):
        """
        启动一个后台线程来执行并行的视频导出任务，避免GUI冻结。
        """
        if not self.image_reader or self.image_reader.get_message_count() == 0:
            messagebox.showwarning("导出失败", "数据源未配置或没有消息可供导出。")
            return

        save_path = filedialog.asksaveasfilename(
            title="将视频另存为...",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("所有文件", "*.*")]
        )
        if not save_path:
            print("视频导出被用户取消。")
            return

        # 禁用按钮，防止重复点击
        self.export_button.config(state="disabled", text="导出中...")

        # 创建并启动后台线程
        export_thread = threading.Thread(
            target=self._run_export_in_background,
            args=(save_path,),
            daemon=True
        )
        export_thread.start()

        # 启动一个定时检查器来处理从后台线程返回的消息
        self._check_export_queue()

    # --- [新增] 在后台线程中运行的实际并行处理逻辑 ---
    def _run_export_in_background(self, save_path):
        """
        这个方法在独立的线程中运行，负责管理多进程池来渲染所有帧。
        """
        try:
            output_dir = os.path.dirname(save_path)
            video_filename = os.path.basename(save_path)
            total_frames = self.image_reader.get_message_count()

            # 准备要传递给每个子进程的参数
            # 注意：我们将所有必需的数据打包，而不是传递 self
            tasks_args = []
            for i in range(total_frames):
                tasks_args.append((
                    i, self.bag_path, self.image_topic, self.lidar_topic,
                    self.K, self.T_cam_lidar, self.dist_coeffs
                ))

            print(f"开始并行渲染，共 {total_frames} 帧...")
            rendered_frames = [None] * total_frames

            # 创建一个进程池，CPU核心数可根据机器性能调整
            # os.cpu_count() 使用所有可用的CPU核心
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"使用 {num_workers} 个并行工作进程。")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                future_to_index = {executor.submit(render_single_frame, args): args[0] for args in tasks_args}

                # 当任务完成时，获取结果
                for future in concurrent.futures.as_completed(future_to_index):
                    index, image = future.result()
                    if image is not None:
                        rendered_frames[index] = image
                        print(f"  帧 {index + 1}/{total_frames} 渲染完成。")
                    else:
                        print(f"  帧 {index + 1}/{total_frames} 渲染失败。")

            print("所有帧渲染完毕，开始生成视频文件...")
            # 过滤掉渲染失败的帧
            valid_frames = [frame for frame in rendered_frames if frame is not None]

            if not valid_frames:
                raise ValueError("所有帧都渲染失败，无法创建视频。")

            final_path = self._create_video_from_frames(
                image_generator=valid_frames,
                output_dir=output_dir,
                video_filename=video_filename,
                fps=15
            )
            # 通过队列将成功消息发送回主线程
            self.export_queue.put(("success", final_path))

        except Exception as e:
            # 通过队列将错误消息发送回主线程
            print(f"后台导出线程发生错误: {e}")
            self.export_queue.put(("error", str(e)))

    # --- [新增] 定时检查队列，安全地更新GUI ---
    def _check_export_queue(self):
        """
        从主GUI线程调用，检查后台线程是否有消息，并据此更新UI。
        """
        try:
            # 非阻塞地获取消息
            message_type, data = self.export_queue.get_nowait()

            if message_type == "success":
                messagebox.showinfo("导出完成", f"视频已成功保存至:\n{data}")
            elif message_type == "error":
                messagebox.showerror("导出错误", f"创建视频时发生错误:\n{data}")

            # 任务完成，恢复按钮状态
            self.export_button.config(state="normal", text="导出视频")

        except queue.Empty:
            # 队列为空，表示后台任务仍在运行
            # 100毫秒后再次检查
            self.after(100, self._check_export_queue)

    # --- [新增] --- 视频创建的辅助方法
    def _create_video_from_frames(
            self,
            image_generator: Iterable[np.ndarray],
            output_dir: str,
            video_filename: str,
            fps: int = 15,
            codec: str = 'mp4v'
    ) -> Optional[str]:
        """
        [内部方法] 将一系列图像帧保存并编码为视频文件。
        """
        # 1. 创建用于存放临时帧的目录
        frames_subdir = os.path.join(output_dir, "temp_frames_for_video")
        if os.path.exists(frames_subdir):
            shutil.rmtree(frames_subdir)
        os.makedirs(frames_subdir, exist_ok=True)

        # 2. 迭代生成器，将每一帧图像保存为图片文件
        frame_size: Optional[Tuple[int, int]] = None
        frame_count = 0
        print("正在将视频帧保存为临时图片文件...")
        for i, frame in enumerate(image_generator):
            if frame_size is None:
                height, width, _ = frame.shape
                frame_size = (width, height)
                print(f"检测到视频尺寸: {width}x{height}")

            # 使用带前导零的命名，确保后续glob排序正确
            frame_path = os.path.join(frames_subdir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        if frame_count == 0 or frame_size is None:
            print("错误：没有有效的帧可用于创建视频。")
            shutil.rmtree(frames_subdir)
            return None

        print(f"临时图片文件保存完毕，共 {frame_count} 帧。")

        # 3. 使用OpenCV将图片序列编码成视频
        video_path = os.path.join(output_dir, video_filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        print(f"正在编码视频: {video_path}")
        # 按文件名顺序读取帧
        saved_frames = sorted(glob.glob(os.path.join(frames_subdir, "frame_*.png")))
        for frame_path in saved_frames:
            img = cv2.imread(frame_path)
            video_writer.write(img)

        video_writer.release()
        print("视频编码完成。")

        # 4. 清理临时文件
        print("正在清理临时文件...")
        shutil.rmtree(frames_subdir)
        print("清理完毕。")

        return video_path


# --- 使用示例 ---
# --- 使用示例 ---
if __name__ == '__main__':
    # 确保 matplotlib 在示例代码中被导入
    import matplotlib.pyplot as plt

    # 创建主窗口和ReprojectionViewer实例
    root = ttk.Window(themename="flatly")
    root.title("主控制窗口")

    viewer = ReprojectionViewer(root)


    # --- 生成模拟数据 ---
    def generate_dummy_data():
        print("正在生成新的模拟数据...")
        # (这部分函数内容与之前完全相同)
        img_size = (480, 640)
        image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        cv2.putText(image, "Original Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        fx, fy = 320, 320
        cx, cy = img_size[1] / 2, img_size[0] / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(5)
        cam_pos = np.array([5, -5, 3])
        target = np.array([0, 0, 0])
        up_vector = np.array([0, 0, 1])
        z_c = - (target - cam_pos) / np.linalg.norm(target - cam_pos)
        x_c = np.cross(up_vector, z_c) / np.linalg.norm(np.cross(up_vector, z_c))
        y_c = np.cross(z_c, x_c)
        R_world_cam = np.vstack([x_c, y_c, z_c])
        R_cam_world = R_world_cam.T
        t_cam_world = -R_cam_world @ cam_pos
        T_cam_world = np.eye(4)
        T_cam_world[:3, :3] = R_cam_world
        T_cam_world[:3, 3] = t_cam_world
        points_world = np.random.rand(1000, 3)
        points_world[:, 0] *= 10;
        points_world[:, 1] *= 10;
        points_world[:, 2] *= 5
        return image, points_world, K, T_cam_world, dist_coeffs


    # --- 控制按钮 ---
    def update_viewer():
        # --- 【核心修正】 ---
        # 不再调用旧的 update_data 方法
        # 而是直接调用内部的更新方法来展示模拟数据
        img, pts, K_mat, T_mat, dist = generate_dummy_data()

        # 将外参从 World->Cam 转换为 Cam->Lidar (在模拟中, Lidar就是World)
        # T_cam_lidar 就是 T_cam_world

        viewer.K = K_mat
        viewer.dist_coeffs = dist
        viewer.T_cam_lidar = T_mat  # 这里T_mat就是Cam<-World（你模拟里Lidar=World）

        viewer._update_original_image(img)
        viewer._update_reprojection(img, pts.astype(np.float32))  # OpenCV更稳用float32
        viewer._update_3d_view(pts)


    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack()
    ttk.Button(control_frame, text="加载/刷新模拟数据", command=update_viewer, bootstyle="primary").pack()

    # 第一次加载
    root.after(100, update_viewer)

    root.mainloop()