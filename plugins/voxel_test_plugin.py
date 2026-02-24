import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
import numpy as np
import pickle
import threading
import multiprocessing as mp
import tempfile
import uuid
import os
import open3d as o3d  # --- 引入强大的 Open3D ---

import sensor_msgs.point_cloud2 as pc2

import matplotlib
# matplotlib.use('TkAgg')  # 移除，避免与 data_plotter_plugin 冲突
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from plugin_core import RosBagPluginBase

class VoxelTestWindow(ttk.Toplevel):
    def __init__(self, master, points_array, error_msg=None, viewer=None, topic=None, index_data=None, current_index=None):
        super().__init__(master)
        self.title("体素降采样幂律关系分析")
        self.geometry("1000x600")
        self.points = points_array
        self.error_msg = error_msg
        self.viewer = viewer
        self.topic = topic
        self.index_data = index_data
        self.current_index = current_index
        self.cache_path = self.viewer._get_cache_paths(self.topic)[1] if self.viewer and self.topic else None
        
        self.voxel_sizes = np.linspace(0.01, 0.2, 50)  # 更密集的体素大小，从 0.01 到 2.0，共 30 个点
        self.point_counts = []
        
        self._create_ui()
        self._update_show_btn_state()
        
        if self.error_msg:
            self.status_var.set(f"数据提取失败: {self.error_msg}")
            self._show_error_in_plot()
        elif len(self.points) == 0:
            self.status_var.set("当前帧点云为空！")
            self._show_error_in_plot()
        else:
            self._update_status()
            # --- 核心修复 1: 开启独立后台线程进行密集计算，坚决不卡 GUI ---
            threading.Thread(target=self._run_analysis, daemon=True).start()

    def _create_ui(self):
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(fill="x")
        
        self.status_var = tk.StringVar(value=f"正在使用 Open3D 高速分析 {len(self.points)} 个点，请稍候...")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Noto Sans CJK SC", 12, "bold")).pack(side="top", fill="x")
        self.total_var = tk.StringVar(value=self._format_total_frames())
        ttk.Label(control_frame, textvariable=self.total_var, font=("Noto Sans CJK SC", 10)).pack(side="top", fill="x")

        if self.index_data and len(self.index_data) > 1:
            btn_frame = ttk.Frame(control_frame)
            btn_frame.pack(side="top", pady=(5, 0))
            self.prev_btn = ttk.Button(btn_frame, text="上一帧", command=self._prev_frame, bootstyle="outline")
            self.prev_btn.pack(side="left", padx=(0, 10))
            self.next_btn = ttk.Button(btn_frame, text="下一帧", command=self._next_frame, bootstyle="outline")
            self.next_btn.pack(side="left")
            ttk.Label(btn_frame, text="跳转帧:").pack(side="left", padx=(12, 4))
            self.jump_var = tk.StringVar()
            self.jump_entry = ttk.Entry(btn_frame, textvariable=self.jump_var, width=8)
            self.jump_entry.pack(side="left")
            self.jump_btn = ttk.Button(btn_frame, text="跳转", command=self._jump_to_frame, bootstyle="outline")
            self.jump_btn.pack(side="left", padx=(6, 0))

        self.show_pcd_btn = ttk.Button(control_frame, text="显示点云", command=self._show_point_cloud, bootstyle="primary")
        self.show_pcd_btn.pack(side="top", pady=(8, 0), anchor="w")
        
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill="both", expand=True)
        
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax_linear = self.fig.add_subplot(121)
        self.ax_log = self.fig.add_subplot(122)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        # --- 核心修复 2: 删除了之前重复的 canvas.pack() ---

    def _update_status(self):
        valid_count, near_count = self._count_valid_points(self.points)
        if self.index_data:
            self.status_var.set(
                f"正在分析帧 {self.current_index + 1}/{len(self.index_data)}: "
                f"{len(self.points)} 个点 (有效 {valid_count}, 近原点 {near_count})"
            )
        else:
            self.status_var.set(
                f"正在使用 Open3D 高速分析 {len(self.points)} 个点 "
                f"(有效 {valid_count}, 近原点 {near_count})，请稍候..."
            )
        self._update_show_btn_state()

    def _update_show_btn_state(self):
        if not hasattr(self, "show_pcd_btn"):
            return
        has_points = self.points is not None and len(self.points) > 0 and not self.error_msg
        self.show_pcd_btn.config(state="normal" if has_points else "disabled")

    def _prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_and_update()

    def _next_frame(self):
        if self.current_index < len(self.index_data) - 1:
            self.current_index += 1
            self._load_and_update()

    def _jump_to_frame(self):
        if not self.index_data or len(self.index_data) == 0:
            messagebox.showwarning("提示", "当前没有可跳转的帧。")
            return
        raw = self.jump_var.get().strip()
        if not raw:
            messagebox.showwarning("提示", "请输入要跳转的帧号。")
            return
        try:
            target = int(raw)
        except ValueError:
            messagebox.showwarning("提示", "帧号必须是整数。")
            return
        if target < 1 or target > len(self.index_data):
            messagebox.showwarning("提示", f"帧号超出范围 (1-{len(self.index_data)})。")
            return
        self.current_index = target - 1
        self._load_and_update()

    def _load_and_update(self):
        try:
            self.status_var.set(f"加载帧 {self.current_index + 1} 中...")
            offset, size = self.index_data[self.current_index]
            with open(self.cache_path, 'rb') as f:
                f.seek(offset)
                msg_type, raw_data, timestamp = pickle.loads(f.read(size))

            msg = self.viewer._deserialize_raw_message(msg_type, raw_data, self.topic)

            if msg_type == "sensor_msgs/PointCloud2":
                points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            elif msg_type in ("livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"):
                points = np.array([[p.x, p.y, p.z] for p in msg.points])
            else:
                raise ValueError(f"不支持的消息类型: {msg_type}")

            self.points = points
            self.total_var.set(self._format_total_frames())
            self._update_status()
            threading.Thread(target=self._run_analysis, daemon=True).start()

        except Exception as e:
            self.status_var.set(f"加载帧失败: {e}")
            self._show_error_in_plot()
        self._update_show_btn_state()
        self.ax_linear.clear()
        self.ax_log.clear()
        self.ax_linear.text(0.5, 0.5, self.status_var.get(), ha='center', va='center', transform=self.ax_linear.transAxes, fontsize=12)
        self.ax_log.text(0.5, 0.5, self.status_var.get(), ha='center', va='center', transform=self.ax_log.transAxes, fontsize=12)
        self.fig.tight_layout()
        self.canvas.draw()

    def _run_analysis(self):
        try:
            total_points = len(self.points)
            if total_points == 0:
                self.after(0, lambda: self.status_var.set("当前帧点云为空！"))
                return

            # --- 核心修复 3: 使用 Open3D 进行 O(N) 级别的极速降采样 ---
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)

            counts = []
            for v in self.voxel_sizes:
                downpcd = pcd.voxel_down_sample(voxel_size=v)
                counts.append(len(downpcd.points))
            
            self.point_counts = np.array(counts)
            
            # --- 核心修复 4: 计算完成后，必须通过 after() 将绘图任务安全地丢回主线程 ---
            self.after(0, self._plot_results)
            
        except Exception as e:
            self.after(0, lambda: self.status_var.set(f"分析出错: {e}"))

    def _plot_results(self):
        self.ax_linear.clear()
        self.ax_log.clear()

        # 子图 1: 线性比例
        self.ax_linear.plot(self.voxel_sizes, self.point_counts, 'o-', color='#2c3e50', linewidth=2, markersize=6)
        self.ax_linear.set_title("Voxel Size vs Point Count")
        self.ax_linear.set_xlabel("Voxel Size v (m)")
        self.ax_linear.set_ylabel("Number of Points N")
        self.ax_linear.grid(True, linestyle='--', alpha=0.7)

        # 子图 2: 双对数比例与分形维度拟合
        valid_mask = self.point_counts > 0
        if not np.any(valid_mask):
            self.ax_log.text(0.5, 0.5, "无有效数据进行对数拟合", ha='center', va='center', transform=self.ax_log.transAxes)
            self.status_var.set(f"分析完成！当前帧原始点数: {len(self.points)}。无法进行分形维度拟合（点数为0）。")
        else:
            log_v = np.log10(self.voxel_sizes[valid_mask])
            log_N = np.log10(self.point_counts[valid_mask])
            
            coeffs = np.polyfit(log_v, log_N, 1)
            k = coeffs[0]  # 拟合指数
            C = coeffs[1]
            fit_log_N = k * log_v + C
            
            self.ax_log.plot(log_v, log_N, 'o', color='#e74c3c', label='Actual Data')
            self.ax_log.plot(log_v, fit_log_N, '--', color='#2980b9', label=f'Fit Exp: {k:.2f}')
            
            self.ax_log.set_title("Log-Log Plot (Empirical Dimension)")
            self.ax_log.set_xlabel("Log10(v)")
            self.ax_log.set_ylabel("Log10(N)")
            self.ax_log.legend()
            self.ax_log.grid(True, linestyle='--', alpha=0.7)
            
            valid_count, near_count = self._count_valid_points(self.points)
            self.status_var.set(
                f"分析完成！当前帧 {self.current_index + 1 if self.current_index is not None else ''} "
                f"原始点数: {len(self.points)} (有效 {valid_count}, 近原点 {near_count})。"
                f"环境分形维度拟合指数约为: {k:.3f}"
            )

        self.fig.tight_layout()
        self.canvas.draw()

    def _count_valid_points(self, points):
        if points is None or len(points) == 0:
            return 0, 0
        distances = np.linalg.norm(points, axis=1)
        near_mask = distances < 0.01
        near_count = int(np.sum(near_mask))
        valid_count = int(len(points) - near_count)
        return valid_count, near_count

    def _format_total_frames(self):
        if self.index_data:
            return f"总帧数: {len(self.index_data)}"
        return "总帧数: N/A"

    def _show_point_cloud(self):
        if self.points is None or len(self.points) == 0:
            messagebox.showwarning("提示", "当前帧没有点云可显示。")
            return
        try:
            points = np.asarray(self.points)
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f"rosbag_viewer_points_{uuid.uuid4().hex}.npy")
            np.save(tmp_path, points)
            proc = mp.Process(target=_open_point_cloud_window_process, args=(tmp_path,), daemon=True)
            proc.start()
        except Exception as e:
            messagebox.showerror("错误", f"显示点云失败: {e}")


def _open_point_cloud_window_process(points_path):
    try:
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
    except Exception as e:
        print(f"[voxel_test_plugin] Open3D GUI 模块不可用: {e}")
        return

    try:
        points = np.load(points_path)
    except Exception as e:
        print(f"[voxel_test_plugin] 读取点云文件失败: {e}")
        return
    finally:
        try:
            os.remove(points_path)
        except Exception:
            pass

    if points is None or len(points) == 0:
        return

    center = np.array([0.0, 0.0, 0.0])
    distances_all = np.linalg.norm(points - center, axis=1)
    near_mask = distances_all < 0.01
    near_count = int(np.sum(near_mask))
    points = points[~near_mask]
    distances = distances_all[~near_mask]

    app = gui.Application.instance
    app.initialize()
    _configure_gui_font(app)

    window = app.create_window("点云查看与半径统计", 1024, 768)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0.05, 0.05, 0.05, 1.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if hasattr(rendering, "MaterialRecord"):
        material = rendering.MaterialRecord()
        sphere_material = rendering.MaterialRecord()
    else:
        material = rendering.Material()
        sphere_material = rendering.Material()
    material.shader = "defaultUnlit"
    sphere_material.shader = "defaultLitTransparency"
    sphere_material.base_color = [0.9, 0.9, 0.95, 0.2]
    scene.scene.add_geometry("pcd", pcd, material)
    bounds = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
    title = gui.Label("半径统计（以点云中心为圆心）")
    panel.add_child(title)

    slider = gui.Slider(gui.Slider.DOUBLE)
    slider.set_limits(0.1, 10.0)
    slider.double_value = 1.0
    panel.add_child(slider)

    counts_label = gui.Label("")
    panel.add_child(counts_label)

    def update_counts(r):
        inside = int(np.sum(distances <= r))
        outside = int(len(distances) - inside)
        counts_label.text = (
            f"r = {r:.2f} m | 内: {inside} | 外: {outside} | 近原点(<1cm): {near_count}"
        )

    def on_slider_changed(value):
        update_counts(value)
        try:
            scene.scene.remove_geometry("sphere")
        except Exception:
            pass
        try:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=value, resolution=32)
            sphere.translate(center)
            sphere.compute_vertex_normals()
            scene.scene.add_geometry("sphere", sphere, sphere_material)
        except Exception:
            pass

    slider.set_on_value_changed(on_slider_changed)
    update_counts(slider.double_value)
    try:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=slider.double_value, resolution=32)
        sphere.translate(center)
        sphere.compute_vertex_normals()
        scene.scene.add_geometry("sphere", sphere, sphere_material)
    except Exception:
        pass

    window.add_child(scene)
    window.add_child(panel)

    panel_width = 280

    def on_layout(layout_context):
        r = window.content_rect
        panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        scene.frame = gui.Rect(r.x, r.y, max(1, r.width - panel_width), r.height)

    window.set_on_layout(on_layout)

    app.run()


def _configure_gui_font(app):
    try:
        import open3d.visualization.gui as gui
    except Exception:
        return
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 兼容不同系统路径重复无害
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_desc = gui.FontDescription()
                font_desc.add_typeface_for_language(path, "zh")
                app.set_font(app.DEFAULT_FONT_ID, font_desc)
                return
            except Exception:
                continue


class VoxelTestPlugin(RosBagPluginBase):
    def __init__(self, context):
        super().__init__(context)
        self.test_window = None

    def get_name(self) -> str:
        return "体素降采样分析"

    def get_button_style(self) -> str:
        return "warning"

    def on_start(self):
        print("VoxelTestPlugin.on_start called")  # 调试输出
        if self.test_window and self.test_window.winfo_exists():
            self.test_window.lift()
            return

        topic = self.context.get_current_topic()
        index = self.context.get_current_index()

        print(f"Current topic: {topic}, index: {index}")  # 调试输出

        if not topic:
            messagebox.showwarning("警告", "请先选择一个话题。")
            return

        info = self.context.topic_info.get(topic)
        supported_types = ["sensor_msgs/PointCloud2", "livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"]
        if not info or info.msg_type not in supported_types:
            # 列出所有支持的点云话题
            available_topics = [t for t, i in self.context.topic_info.items() if i.msg_type in supported_types]
            if available_topics:
                topic_list = "\n".join(available_topics)
                messagebox.showwarning("警告", f"当前话题 '{topic}' 不支持。该插件仅支持以下点云话题：\n{topic_list}\n请先选择其中一个点云话题。")
            else:
                messagebox.showwarning("警告", "该插件仅支持 sensor_msgs/PointCloud2、livox_ros_driver2/CustomMsg 和 livox_ros_driver/CustomMsg 类型的话题，但包中没有此类话题。")
            return

        viewer = self.context._viewer
        index_data = viewer.topic_indices.get(topic)
        if not index_data or index < 0 or index >= len(index_data):
            messagebox.showwarning("警告", "无法获取当前帧的数据，请确保索引已完成。")
            return

        try:
            print("Attempting to extract point cloud data")  # 调试输出
            _, cache_path = viewer._get_cache_paths(topic)
            offset, size = index_data[index]
            with open(cache_path, 'rb') as f:
                f.seek(offset)
                msg_type, raw_data, timestamp = pickle.loads(f.read(size))

            msg = viewer._deserialize_raw_message(msg_type, raw_data, topic)

            if msg_type == "sensor_msgs/PointCloud2":
                points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            elif msg_type in ("livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"):
                points = np.array([[p.x, p.y, p.z] for p in msg.points])
            else:
                raise ValueError(f"不支持的消息类型: {msg_type}")

            print(f"Extracted {len(points)} points")  # 调试输出
            
            self.test_window = VoxelTestWindow(self.context.master, points, viewer=viewer, topic=topic, index_data=index_data, current_index=index)
            print("VoxelTestWindow created successfully")  # 调试输出

        except Exception as e:
            print(f"Error in on_start: {e}")  # 调试输出
            # 即使失败，也创建窗口显示错误
            self.test_window = VoxelTestWindow(self.context.master, np.array([]), error_msg=str(e))
