import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
import numpy as np
import pickle
import threading
import concurrent.futures
import multiprocessing as mp
import tempfile
import uuid
import os
import open3d as o3d  # --- 引入强大的 Open3D ---
import genpy.dynamic

import sensor_msgs.point_cloud2 as pc2

import matplotlib
# matplotlib.use('TkAgg')  # 移除，避免与 data_plotter_plugin 冲突
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from plugin_core import RosBagPluginBase

# ---- 目标点数多进程计算支持 ----
_WORKER_CACHE_PATH = None
_WORKER_MSG_DEF_MAP = None
_WORKER_TARGET_N = None
_WORKER_VMIN = 0.02
_WORKER_VMAX = 10.0
_WORKER_TOL = 0.01
_WORKER_FIXED_VOXEL = None
_WORKER_MULTI_THRESHOLDS = (3.0, 10.0)
_WORKER_MULTI_VOXELS = (0.1, 0.3, 0.5)
_WORKER_CLASS_CACHE = {}


def _init_target_worker(cache_path, msg_def_map, target_n, v_min, v_max, tol):
    global _WORKER_CACHE_PATH, _WORKER_MSG_DEF_MAP, _WORKER_TARGET_N, _WORKER_VMIN, _WORKER_VMAX, _WORKER_TOL
    _WORKER_CACHE_PATH = cache_path
    _WORKER_MSG_DEF_MAP = msg_def_map
    _WORKER_TARGET_N = target_n
    _WORKER_VMIN = v_min
    _WORKER_VMAX = v_max
    _WORKER_TOL = tol


def _init_fixed_voxel_worker(cache_path, msg_def_map, voxel_size):
    global _WORKER_CACHE_PATH, _WORKER_MSG_DEF_MAP, _WORKER_FIXED_VOXEL
    _WORKER_CACHE_PATH = cache_path
    _WORKER_MSG_DEF_MAP = msg_def_map
    _WORKER_FIXED_VOXEL = voxel_size


def _init_multires_worker(cache_path, msg_def_map):
    global _WORKER_CACHE_PATH, _WORKER_MSG_DEF_MAP
    _WORKER_CACHE_PATH = cache_path
    _WORKER_MSG_DEF_MAP = msg_def_map


def _get_dynamic_msg_class_in_worker(msg_type):
    if msg_type in _WORKER_CLASS_CACHE:
        return _WORKER_CLASS_CACHE[msg_type]
    msg_def = _WORKER_MSG_DEF_MAP.get(msg_type)
    if not msg_def:
        raise ValueError(f"缺少消息定义: {msg_type}")
    generated = genpy.dynamic.generate_dynamic(msg_type, msg_def)
    msg_class = generated.get(msg_type)
    if not msg_class:
        raise ValueError(f"动态生成消息类失败: {msg_type}")
    _WORKER_CLASS_CACHE[msg_type] = msg_class
    return msg_class


def _load_points_from_cache_worker(offset, size):
    with open(_WORKER_CACHE_PATH, 'rb') as f:
        f.seek(offset)
        msg_type, raw_data, timestamp = pickle.loads(f.read(size))
    msg_class = _get_dynamic_msg_class_in_worker(msg_type)
    msg = msg_class()
    msg.deserialize(raw_data)
    if msg_type == "sensor_msgs/PointCloud2":
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    elif msg_type in ("livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"):
        points = np.array([[p.x, p.y, p.z] for p in msg.points])
    else:
        raise ValueError(f"不支持的消息类型: {msg_type}")
    return points


def _filter_near_origin_worker(points):
    if points is None or len(points) == 0:
        return points
    distances = np.linalg.norm(points, axis=1)
    return points[distances >= 0.01]


def _find_voxel_for_target_worker(points, target_n):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    raw_count = len(points)
    if target_n >= raw_count:
        return None, raw_count

    def count_at(v):
        down = pcd.voxel_down_sample(voxel_size=v)
        return len(down.points)

    high_count = count_at(_WORKER_VMAX)
    if high_count > target_n:
        return None, high_count

    low = _WORKER_VMIN
    high = _WORKER_VMAX
    low_count = count_at(low)
    if low_count <= target_n:
        return low, low_count

    while (high - low) > _WORKER_TOL:
        mid = (low + high) / 2.0
        mid_count = count_at(mid)
        if mid_count > target_n:
            low = mid
        else:
            high = mid
    final_v = (low + high) / 2.0
    final_count = count_at(final_v)
    return final_v, final_count


def _target_worker_task(args):
    i, offset, size = args
    points = _load_points_from_cache_worker(offset, size)
    points = _filter_near_origin_worker(points)
    if points is None or len(points) == 0:
        return i, float("nan"), float("nan")
    v, count = _find_voxel_for_target_worker(points, _WORKER_TARGET_N)
    if v is None:
        return i, float("nan"), float("nan")
    return i, v, count - _WORKER_TARGET_N


def _fixed_voxel_worker_task(args):
    i, offset, size = args
    points = _load_points_from_cache_worker(offset, size)
    points = _filter_near_origin_worker(points)
    if points is None or len(points) == 0:
        return i, float("nan")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(voxel_size=_WORKER_FIXED_VOXEL)
    return i, float(len(down.points))


def _multires_worker_task(args):
    i, offset, size = args
    points = _load_points_from_cache_worker(offset, size)
    points = _filter_near_origin_worker(points)
    if points is None or len(points) == 0:
        return i, float("nan")

    d = np.linalg.norm(points, axis=1)
    t1, t2 = _WORKER_MULTI_THRESHOLDS
    v1, v2, v3 = _WORKER_MULTI_VOXELS

    masks = [
        d <= t1,
        (d > t1) & (d <= t2),
        d > t2
    ]
    voxels = [v1, v2, v3]
    total = 0
    for mask, v in zip(masks, voxels):
        if not np.any(mask):
            continue
        p = points[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        down = pcd.voxel_down_sample(voxel_size=v)
        total += len(down.points)
    return i, float(total)


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
        self._target_task_running = False
        self._fixed_task_running = False
        self._multires_task_running = False
        self._adaptive_multires_task_running = False
        self._adaptive_single_task_running = False
        self.cn_font = self._get_chinese_font()
        
        self.voxel_sizes = np.linspace(0.02, 0.5, 50)  # 更密集的体素大小，从 0.02 到 0.5，共 50 个点
        self.point_counts = []
        self._hover_lines = []
        self._hover_annots = {}
        
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

            ttk.Label(btn_frame, text="目标点数 N:").pack(side="left", padx=(12, 4))
            self.target_var = tk.StringVar()
            self.target_entry = ttk.Entry(btn_frame, textvariable=self.target_var, width=10)
            self.target_entry.pack(side="left")
            self.target_btn = ttk.Button(btn_frame, text="计算体素", command=self._start_target_scan, bootstyle="outline")
            self.target_btn.pack(side="left", padx=(6, 0))

            ttk.Label(btn_frame, text="体素大小 v:").pack(side="left", padx=(12, 4))
            self.fixed_voxel_var = tk.StringVar()
            self.fixed_voxel_entry = ttk.Entry(btn_frame, textvariable=self.fixed_voxel_var, width=8)
            self.fixed_voxel_entry.pack(side="left")
            self.fixed_voxel_btn = ttk.Button(btn_frame, text="统计点数", command=self._start_fixed_voxel_scan, bootstyle="outline")
            self.fixed_voxel_btn.pack(side="left", padx=(6, 0))

            self.multires_btn = ttk.Button(btn_frame, text="多级体素统计", command=self._start_multires_scan, bootstyle="outline")
            self.multires_btn.pack(side="left", padx=(12, 0))
            self.adaptive_multires_btn = ttk.Button(btn_frame, text="自适应多级体素", command=self._start_adaptive_multires_scan, bootstyle="outline")
            self.adaptive_multires_btn.pack(side="left", padx=(12, 0))
            self.adaptive_single_btn = ttk.Button(btn_frame, text="自适应单级体素", command=self._start_adaptive_single_scan, bootstyle="outline")
            self.adaptive_single_btn.pack(side="left", padx=(12, 0))

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
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self._setup_plot_context_menu()

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
        if hasattr(self, "target_btn"):
            self.target_btn.config(state="disabled" if self._target_task_running else "normal")
        if hasattr(self, "fixed_voxel_btn"):
            self.fixed_voxel_btn.config(state="disabled" if self._fixed_task_running else "normal")
        if hasattr(self, "multires_btn"):
            self.multires_btn.config(state="disabled" if self._multires_task_running else "normal")
        if hasattr(self, "adaptive_multires_btn"):
            self.adaptive_multires_btn.config(state="disabled" if self._adaptive_multires_task_running else "normal")
        if hasattr(self, "adaptive_single_btn"):
            self.adaptive_single_btn.config(state="disabled" if self._adaptive_single_task_running else "normal")

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
        line_linear, = self.ax_linear.plot(self.voxel_sizes, self.point_counts, 'o-', color='#2c3e50', linewidth=2, markersize=6)
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
            
            line_log_actual, = self.ax_log.plot(log_v, log_N, 'o', color='#e74c3c', label='Actual Data')
            line_log_fit, = self.ax_log.plot(log_v, fit_log_N, '--', color='#2980b9', label=f'Fit Exp: {k:.2f}')
            
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

        self._hover_lines = []
        self._hover_lines.append((self.ax_linear, line_linear))
        if 'line_log_actual' in locals():
            self._hover_lines.append((self.ax_log, line_log_actual))
        if 'line_log_fit' in locals():
            self._hover_lines.append((self.ax_log, line_log_fit))

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

    def _start_target_scan(self):
        if self._target_task_running:
            return
        if not self.index_data or len(self.index_data) == 0:
            messagebox.showwarning("提示", "当前没有可计算的帧。")
            return
        raw = self.target_var.get().strip() if hasattr(self, "target_var") else ""
        if not raw:
            messagebox.showwarning("提示", "请输入目标点数 N。")
            return
        try:
            target_n = int(raw)
        except ValueError:
            messagebox.showwarning("提示", "目标点数必须是整数。")
            return
        if target_n <= 0:
            messagebox.showwarning("提示", "目标点数必须大于 0。")
            return

        self._target_task_running = True
        self._update_show_btn_state()
        threading.Thread(target=self._compute_target_voxels_all_frames, args=(target_n,), daemon=True).start()

    def _start_fixed_voxel_scan(self):
        if self._fixed_task_running:
            return
        if not self.index_data or len(self.index_data) == 0:
            messagebox.showwarning("提示", "当前没有可计算的帧。")
            return
        raw = self.fixed_voxel_var.get().strip() if hasattr(self, "fixed_voxel_var") else ""
        if not raw:
            messagebox.showwarning("提示", "请输入体素大小 v。")
            return
        try:
            voxel_size = float(raw)
        except ValueError:
            messagebox.showwarning("提示", "体素大小必须是数字。")
            return
        if voxel_size <= 0:
            messagebox.showwarning("提示", "体素大小必须大于 0。")
            return

        self._fixed_task_running = True
        self._update_show_btn_state()
        threading.Thread(target=self._compute_fixed_voxel_all_frames, args=(voxel_size,), daemon=True).start()

    def _compute_target_voxels_all_frames(self, target_n):
        total_frames = len(self.index_data)
        voxel_sizes = [float("nan")] * total_frames
        errors = [float("nan")] * total_frames
        frame_ids = list(range(1, total_frames + 1))

        cpu_count = os.cpu_count() or 1
        max_workers = max(1, min(10, cpu_count - 2))
        completed = 0

        if not self.viewer or not self.topic:
            self.after(0, lambda: self.status_var.set("无法计算：Viewer 或 topic 不可用。"))
            return
        msg_type = self.viewer.topic_info[self.topic].msg_type
        msg_def = self.viewer._get_msg_def_for_topic(self.topic, msg_type)
        msg_def_map = {msg_type: msg_def}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_target_worker,
            initargs=(self.cache_path, msg_def_map, target_n, 0.02, 10.0, 0.01)
        ) as executor:
            futures = [
                executor.submit(_target_worker_task, (i, offset, size))
                for i, (offset, size) in enumerate(self.index_data)
            ]
            for fut in concurrent.futures.as_completed(futures):
                if not self._target_task_running:
                    return
                try:
                    i, v, err = fut.result()
                    voxel_sizes[i] = v
                    errors[i] = err
                except Exception as e:
                    print(f"[voxel_test_plugin] 目标点数计算失败: {e}")
                completed += 1
                if completed % 5 == 0 or completed == total_frames:
                    self.after(0, lambda c=completed: self.status_var.set(
                        f"计算目标点数中... 已完成 {c}/{total_frames}"))

        def finalize():
            self._target_task_running = False
            self._update_show_btn_state()
            self._plot_target_results(frame_ids, voxel_sizes, errors, target_n)
            self.status_var.set(f"计算完成！目标点数 N={target_n}，已处理 {total_frames} 帧。")

        self.after(0, finalize)

    def _compute_fixed_voxel_all_frames(self, voxel_size):
        total_frames = len(self.index_data)
        counts = [float("nan")] * total_frames
        frame_ids = list(range(1, total_frames + 1))

        cpu_count = os.cpu_count() or 1
        max_workers = max(1, min(10, cpu_count - 2))
        completed = 0

        if not self.viewer or not self.topic:
            self.after(0, lambda: self.status_var.set("无法计算：Viewer 或 topic 不可用。"))
            return
        msg_type = self.viewer.topic_info[self.topic].msg_type
        msg_def = self.viewer._get_msg_def_for_topic(self.topic, msg_type)
        msg_def_map = {msg_type: msg_def}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_fixed_voxel_worker,
            initargs=(self.cache_path, msg_def_map, voxel_size)
        ) as executor:
            futures = [
                executor.submit(_fixed_voxel_worker_task, (i, offset, size))
                for i, (offset, size) in enumerate(self.index_data)
            ]
            for fut in concurrent.futures.as_completed(futures):
                if not self._fixed_task_running:
                    return
                try:
                    i, count = fut.result()
                    counts[i] = count
                except Exception as e:
                    print(f"[voxel_test_plugin] 固定体素统计失败: {e}")
                completed += 1
                if completed % 5 == 0 or completed == total_frames:
                    self.after(0, lambda c=completed: self.status_var.set(
                        f"固定体素统计中... 已完成 {c}/{total_frames}"))

        def finalize():
            self._fixed_task_running = False
            self._update_show_btn_state()
            self._plot_fixed_voxel_results(frame_ids, counts, voxel_size)
            self.status_var.set(f"计算完成！体素 v={voxel_size:.3f} m，已处理 {total_frames} 帧。")

        self.after(0, finalize)

    def _start_multires_scan(self):
        if self._multires_task_running:
            return
        if not self.index_data or len(self.index_data) == 0:
            messagebox.showwarning("提示", "当前没有可计算的帧。")
            return
        self._multires_task_running = True
        self._update_show_btn_state()
        threading.Thread(target=self._compute_multires_all_frames, daemon=True).start()

    def _compute_multires_all_frames(self):
        total_frames = len(self.index_data)
        counts = [float("nan")] * total_frames
        frame_ids = list(range(1, total_frames + 1))

        cpu_count = os.cpu_count() or 1
        max_workers = max(1, min(10, cpu_count - 2))
        completed = 0

        if not self.viewer or not self.topic:
            self.after(0, lambda: self.status_var.set("无法计算：Viewer 或 topic 不可用。"))
            return
        msg_type = self.viewer.topic_info[self.topic].msg_type
        msg_def = self.viewer._get_msg_def_for_topic(self.topic, msg_type)
        msg_def_map = {msg_type: msg_def}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_multires_worker,
            initargs=(self.cache_path, msg_def_map)
        ) as executor:
            futures = [
                executor.submit(_multires_worker_task, (i, offset, size))
                for i, (offset, size) in enumerate(self.index_data)
            ]
            for fut in concurrent.futures.as_completed(futures):
                if not self._multires_task_running:
                    return
                try:
                    i, count = fut.result()
                    counts[i] = count
                except Exception as e:
                    print(f"[voxel_test_plugin] 多级体素统计失败: {e}")
                completed += 1
                if completed % 5 == 0 or completed == total_frames:
                    self.after(0, lambda c=completed: self.status_var.set(
                        f"多级体素统计中... 已完成 {c}/{total_frames}"))

        def finalize():
            self._multires_task_running = False
            self._update_show_btn_state()
            self._plot_multires_results(frame_ids, counts)
            self.status_var.set(f"计算完成！多级体素统计已处理 {total_frames} 帧。")

        self.after(0, finalize)

    def _start_adaptive_multires_scan(self):
        if self._adaptive_multires_task_running:
            return
        if not self.index_data or len(self.index_data) == 0:
            messagebox.showwarning("提示", "当前没有可计算的帧。")
            return
        self._adaptive_multires_task_running = True
        self._update_show_btn_state()
        threading.Thread(target=self._compute_adaptive_multires_all_frames, daemon=True).start()

    def _compute_adaptive_multires_all_frames(self):
        target_n = 2500
        alpha = 1.5
        base_min = 0.02
        base_max = 0.8
        total_frames = len(self.index_data)
        counts = [float("nan")] * total_frames
        base_vs = [float("nan")] * total_frames
        frame_ids = list(range(1, total_frames + 1))

        # 初始体素大小（经验值），用于第 1 帧
        base_v = 0.5

        for i, (offset, size) in enumerate(self.index_data):
            if not self._adaptive_multires_task_running:
                return
            self.after(0, lambda i=i: self.status_var.set(f"自适应多级体素中... 帧 {i + 1}/{total_frames}"))
            try:
                points = self._load_points_from_cache(offset, size)
                points = self._filter_near_origin(points)
                if points is None or len(points) == 0:
                    counts[i] = float("nan")
                    base_vs[i] = float("nan")
                    continue

                count = self._apply_multires(points, base_v)
                counts[i] = count
                base_vs[i] = base_v

                # 更新下一帧的体素大小
                if count > 0:
                    base_v = base_v * (count / target_n) ** (1.0 / alpha)
                    base_v = max(base_min, min(base_max, base_v))
            except Exception as e:
                print(f"[voxel_test_plugin] 自适应多级体素失败: {e}")

        def finalize():
            self._adaptive_multires_task_running = False
            self._update_show_btn_state()
            self._plot_adaptive_multires_results(frame_ids, counts, base_vs, target_n, alpha)
            self.status_var.set(f"计算完成！自适应多级体素已处理 {total_frames} 帧。")

        self.after(0, finalize)

    def _start_adaptive_single_scan(self):
        if self._adaptive_single_task_running:
            return
        if not self.index_data or len(self.index_data) == 0:
            messagebox.showwarning("提示", "当前没有可计算的帧。")
            return
        self._adaptive_single_task_running = True
        self._update_show_btn_state()
        threading.Thread(target=self._compute_adaptive_single_all_frames, daemon=True).start()

    def _compute_adaptive_single_all_frames(self):
        target_n = 2500
        alpha = 1.5
        v_min = 0.02
        v_max = 0.8
        total_frames = len(self.index_data)
        counts = [float("nan")] * total_frames
        vs = [float("nan")] * total_frames
        frame_ids = list(range(1, total_frames + 1))

        v = 0.5

        for i, (offset, size) in enumerate(self.index_data):
            if not self._adaptive_single_task_running:
                return
            self.after(0, lambda i=i: self.status_var.set(f"自适应单级体素中... 帧 {i + 1}/{total_frames}"))
            try:
                points = self._load_points_from_cache(offset, size)
                points = self._filter_near_origin(points)
                if points is None or len(points) == 0:
                    counts[i] = float("nan")
                    vs[i] = float("nan")
                    continue

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                down = pcd.voxel_down_sample(voxel_size=v)
                count = len(down.points)
                counts[i] = count
                vs[i] = v

                if count > 0:
                    v = v * (count / target_n) ** (1.0 / alpha)
                    v = max(v_min, min(v_max, v))
            except Exception as e:
                print(f"[voxel_test_plugin] 自适应单级体素失败: {e}")

        def finalize():
            self._adaptive_single_task_running = False
            self._update_show_btn_state()
            self._plot_adaptive_single_results(frame_ids, counts, vs, target_n, alpha)
            self.status_var.set(f"计算完成！自适应单级体素已处理 {total_frames} 帧。")

        self.after(0, finalize)

    def _load_points_from_cache(self, offset, size):
        if not self.viewer or not self.topic:
            raise RuntimeError("Viewer 或 topic 不可用")
        with open(self.cache_path, 'rb') as f:
            f.seek(offset)
            msg_type, raw_data, timestamp = pickle.loads(f.read(size))
        msg = self.viewer._deserialize_raw_message(msg_type, raw_data, self.topic)
        if msg_type == "sensor_msgs/PointCloud2":
            return np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if msg_type in ("livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"):
            return np.array([[p.x, p.y, p.z] for p in msg.points])
        raise ValueError(f"不支持的消息类型: {msg_type}")

    def _filter_near_origin(self, points):
        if points is None or len(points) == 0:
            return points
        distances = np.linalg.norm(points, axis=1)
        return points[distances >= 0.01]

    def _find_voxel_for_target(self, points, target_n, v_min=0.02, v_max=10.0, tol=0.01):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        raw_count = len(points)
        if target_n >= raw_count:
            return None, raw_count

        def count_at(v):
            down = pcd.voxel_down_sample(voxel_size=v)
            return len(down.points)

        high_count = count_at(v_max)
        if high_count > target_n:
            return None, high_count

        low = v_min
        high = v_max
        low_count = count_at(low)
        if low_count <= target_n:
            return low, low_count

        while (high - low) > tol:
            mid = (low + high) / 2.0
            mid_count = count_at(mid)
            if mid_count > target_n:
                low = mid
            else:
                high = mid
        final_v = (low + high) / 2.0
        final_count = count_at(final_v)
        return final_v, final_count

    def _plot_target_results(self, frame_ids, voxel_sizes, errors, target_n):
        self.ax_linear.clear()
        self.ax_log.clear()

        font_kwargs = {'fontproperties': self.cn_font} if self.cn_font else {}
        line_linear, = self.ax_linear.plot(frame_ids, voxel_sizes, '-', color='#2c3e50', linewidth=1.5)
        self.ax_linear.set_title(f"目标点数 N={target_n} 的体素大小", **font_kwargs)
        self.ax_linear.set_xlabel("帧序号", **font_kwargs)
        self.ax_linear.set_ylabel("Voxel Size (m)", **font_kwargs)
        self.ax_linear.grid(True, linestyle='--', alpha=0.6)

        line_log, = self.ax_log.plot(frame_ids, errors, '-', color='#e74c3c', linewidth=1.5)
        self.ax_log.set_title("降采样后点数误差 (实际 - 目标)", **font_kwargs)
        self.ax_log.set_xlabel("帧序号", **font_kwargs)
        self.ax_log.set_ylabel("点数误差", **font_kwargs)
        self.ax_log.grid(True, linestyle='--', alpha=0.6)

        self._hover_lines = [
            (self.ax_linear, line_linear),
            (self.ax_log, line_log),
        ]

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_fixed_voxel_results(self, frame_ids, counts, voxel_size):
        self.ax_linear.clear()
        self.ax_log.clear()

        font_kwargs = {'fontproperties': self.cn_font} if self.cn_font else {}
        line_linear, = self.ax_linear.plot(frame_ids, counts, '-', color='#2c3e50', linewidth=1.5)
        self.ax_linear.set_title(f"体素 v={voxel_size:.3f} m 的点数", **font_kwargs)
        self.ax_linear.set_xlabel("帧序号", **font_kwargs)
        self.ax_linear.set_ylabel("点数", **font_kwargs)
        self.ax_linear.grid(True, linestyle='--', alpha=0.6)

        self.ax_log.text(0.5, 0.5, " ", ha='center', va='center', transform=self.ax_log.transAxes)

        self._hover_lines = [
            (self.ax_linear, line_linear),
        ]

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_multires_results(self, frame_ids, counts):
        self.ax_linear.clear()
        self.ax_log.clear()

        font_kwargs = {'fontproperties': self.cn_font} if self.cn_font else {}
        line_linear, = self.ax_linear.plot(frame_ids, counts, '-', color='#2c3e50', linewidth=1.5)
        self.ax_linear.set_title("多级体素滤波后的点数", **font_kwargs)
        self.ax_linear.set_xlabel("帧序号", **font_kwargs)
        self.ax_linear.set_ylabel("点数", **font_kwargs)
        self.ax_linear.grid(True, linestyle='--', alpha=0.6)

        self.ax_log.text(0.5, 0.5, " ", ha='center', va='center', transform=self.ax_log.transAxes)

        self._hover_lines = [
            (self.ax_linear, line_linear),
        ]

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_adaptive_multires_results(self, frame_ids, counts, base_vs, target_n, alpha):
        self.ax_linear.clear()
        self.ax_log.clear()

        font_kwargs = {'fontproperties': self.cn_font} if self.cn_font else {}
        line_counts, = self.ax_linear.plot(frame_ids, counts, '-', color='#2c3e50', linewidth=1.5)
        self.ax_linear.axhline(target_n, color='#27ae60', linestyle='--', linewidth=1.0)
        self.ax_linear.set_title(f"自适应多级体素点数 (N_target={target_n}, α={alpha})", **font_kwargs)
        self.ax_linear.set_xlabel("帧序号", **font_kwargs)
        self.ax_linear.set_ylabel("点数", **font_kwargs)
        self.ax_linear.grid(True, linestyle='--', alpha=0.6)

        line_base, = self.ax_log.plot(frame_ids, base_vs, '-', color='#8e44ad', linewidth=1.5)
        self.ax_log.set_title("基础体素大小 v_base", **font_kwargs)
        self.ax_log.set_xlabel("帧序号", **font_kwargs)
        self.ax_log.set_ylabel("v_base (m)", **font_kwargs)
        self.ax_log.grid(True, linestyle='--', alpha=0.6)

        self._hover_lines = [
            (self.ax_linear, line_counts),
            (self.ax_log, line_base),
        ]

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_adaptive_single_results(self, frame_ids, counts, vs, target_n, alpha):
        self.ax_linear.clear()
        self.ax_log.clear()

        font_kwargs = {'fontproperties': self.cn_font} if self.cn_font else {}
        line_counts, = self.ax_linear.plot(frame_ids, counts, '-', color='#2c3e50', linewidth=1.5)
        self.ax_linear.axhline(target_n, color='#27ae60', linestyle='--', linewidth=1.0)
        self.ax_linear.set_title(f"自适应单级体素点数 (N_target={target_n}, α={alpha})", **font_kwargs)
        self.ax_linear.set_xlabel("帧序号", **font_kwargs)
        self.ax_linear.set_ylabel("点数", **font_kwargs)
        self.ax_linear.grid(True, linestyle='--', alpha=0.6)

        line_v, = self.ax_log.plot(frame_ids, vs, '-', color='#8e44ad', linewidth=1.5)
        self.ax_log.set_title("体素大小 v", **font_kwargs)
        self.ax_log.set_xlabel("帧序号", **font_kwargs)
        self.ax_log.set_ylabel("v (m)", **font_kwargs)
        self.ax_log.grid(True, linestyle='--', alpha=0.6)

        self._hover_lines = [
            (self.ax_linear, line_counts),
            (self.ax_log, line_v),
        ]

        self.fig.tight_layout()
        self.canvas.draw()

    def _apply_multires(self, points, base_v):
        d = np.linalg.norm(points, axis=1)
        t1, t2 = _WORKER_MULTI_THRESHOLDS
        v1, v2, v3 = base_v, base_v * 2.0, base_v * 4.0
        masks = [
            d <= t1,
            (d > t1) & (d <= t2),
            d > t2
        ]
        voxels = [v1, v2, v3]
        total = 0
        for mask, v in zip(masks, voxels):
            if not np.any(mask):
                continue
            p = points[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p)
            down = pcd.voxel_down_sample(voxel_size=v)
            total += len(down.points)
        return total

    def _setup_plot_context_menu(self):
        self._plot_menu = tk.Menu(self, tearoff=0)
        self._plot_menu.add_command(label="保存曲线数据到 CSV", command=self._save_plot_data_to_csv)
        self.canvas.get_tk_widget().bind("<Button-3>", self._on_plot_right_click)

    def _on_plot_right_click(self, event):
        try:
            self._plot_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._plot_menu.grab_release()

    def _save_plot_data_to_csv(self):
        try:
            from tkinter import filedialog
        except Exception:
            messagebox.showerror("错误", "无法打开文件对话框。")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存曲线数据",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        try:
            lines = []
            for ax in self.fig.axes:
                for line in ax.lines:
                    x = line.get_xdata()
                    y = line.get_ydata()
                    label = line.get_label() if line.get_label() != "_nolegend_" else "series"
                    lines.append((label, x, y))

            if not lines:
                messagebox.showwarning("提示", "当前没有可保存的曲线数据。")
                return

            max_len = max(len(x) for _, x, _ in lines)
            headers = []
            for i, (label, _, _) in enumerate(lines, start=1):
                headers.append(f"{label}_x_{i}")
                headers.append(f"{label}_y_{i}")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for idx in range(max_len):
                    row = []
                    for _, x, y in lines:
                        row.append(str(x[idx]) if idx < len(x) else "")
                        row.append(str(y[idx]) if idx < len(y) else "")
                    f.write(",".join(row) + "\n")

            messagebox.showinfo("完成", f"已保存曲线数据到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    def _get_or_create_annot(self, ax):
        annot = self._hover_annots.get(ax)
        if annot is None:
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color="#555"),
            )
            annot.set_visible(False)
            self._hover_annots[ax] = annot
        return annot

    def _on_hover(self, event):
        if event.inaxes is None:
            for annot in self._hover_annots.values():
                annot.set_visible(False)
            self.canvas.draw_idle()
            return

        ax = event.inaxes
        candidates = [line for a, line in self._hover_lines if a is ax]
        if not candidates:
            return

        best = None
        best_dist = 8.0  # pixels
        for line in candidates:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) == 0:
                continue
            pts = np.column_stack([xdata, ydata])
            disp = ax.transData.transform(pts)
            dx = disp[:, 0] - event.x
            dy = disp[:, 1] - event.y
            d = np.hypot(dx, dy)
            idx = int(np.argmin(d))
            if d[idx] < best_dist:
                best_dist = d[idx]
                best = (line, idx)

        annot = self._get_or_create_annot(ax)
        if best is None:
            annot.set_visible(False)
            self.canvas.draw_idle()
            return

        line, idx = best
        x = float(line.get_xdata()[idx])
        y = float(line.get_ydata()[idx])
        annot.xy = (x, y)
        annot.set_text(f"x={x:.3f}, y={y:.3f}")
        annot.set_visible(True)
        self.canvas.draw_idle()

    def _get_chinese_font(self):
        font_paths = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    from matplotlib.font_manager import FontProperties
                    return FontProperties(fname=path, size=11)
                except Exception:
                    return None
        return None


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
