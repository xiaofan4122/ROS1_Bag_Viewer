# plugins/imu_lidar_conflict_plugin.py
import bisect
import concurrent.futures
import os
import threading
import tkinter as tk
from tkinter import messagebox

import numpy as np
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import matplotlib
import matplotlib.font_manager as _fm
_fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

import open3d as o3d
import sensor_msgs.point_cloud2 as pc2

from bag_cache_reader import BagCacheReader, CacheNotFoundError
from plugin_core import RosBagPluginBase


# ------------------------------------------------------------------ #
#  工具函数
# ------------------------------------------------------------------ #
def _points_from_msg(msg, msg_type: str):
    """统一解析点云消息为 (N,3) float32 ndarray"""
    try:
        if msg_type == "sensor_msgs/PointCloud2":
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            return np.array(pts, dtype=np.float32) if pts else None
        elif msg_type in ("livox_ros_driver/CustomMsg", "livox_ros_driver2/CustomMsg"):
            pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
            return pts if len(pts) > 0 else None
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ #
#  模块级并行工作函数（必须在模块级以支持多进程/线程池 pickle）
# ------------------------------------------------------------------ #
def _icp_imu_worker(msg_type, msg_def, raw0, raw1, imu_segment, gravity_unit, g_mag):
    """
    处理一对相邻LiDAR帧：在子进程内反序列化点云 + ICP + IMU积分。
    参数全部为可pickle的基础类型，适合 ProcessPoolExecutor。
    """
    import roslib.message
    import genpy.dynamic

    # 优先用 roslib 查找，找不到则用 msg_def 动态生成
    cls = roslib.message.get_message_class(msg_type)
    if cls is None:
        generated = genpy.dynamic.generate_dynamic(msg_type, msg_def)
        cls = generated.get(msg_type)

    def _deserialize(raw):
        if cls is None:
            return None
        msg = cls()
        msg.deserialize(raw)
        return msg

    pts_src = _points_from_msg(_deserialize(raw0), msg_type)
    pts_tgt = _points_from_msg(_deserialize(raw1), msg_type)

    # --- ICP ---
    dz_lidar, fitness = 0.0, 0.0
    if (pts_src is not None and pts_tgt is not None
            and len(pts_src) >= 100 and len(pts_tgt) >= 100):
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(pts_src)
        pcd_src = pcd_src.voxel_down_sample(voxel_size=0.2)

        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(pts_tgt)
        pcd_tgt = pcd_tgt.voxel_down_sample(voxel_size=0.2)

        if len(pcd_src.points) > 50 and len(pcd_tgt.points) > 50:
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            result = o3d.pipelines.registration.registration_icp(
                pcd_src, pcd_tgt, 1.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria
            )
            fitness = result.fitness
            dz_lidar = result.transformation[2, 3] if result.fitness >= 0.3 else 0.0

    # --- IMU 积分 ---
    dz_imu, gyro_z_mean = 0.0, 0.0
    if len(imu_segment) >= 2:
        vel_z = 0.0
        disp_z = 0.0
        prev_az, prev_t = None, None
        gyro_zs = []

        for (t, ax, ay, az_raw, wz) in imu_segment:
            raw_a = np.array([ax, ay, az_raw])
            # 投影到重力方向，减去重力大小，得到垂直方向真实加速度
            az = float(np.dot(raw_a, gravity_unit)) - g_mag
            gyro_zs.append(wz)

            if prev_t is not None:
                dt = t - prev_t
                if 0 < dt < 1.0:
                    vel_z += 0.5 * (prev_az + az) * dt
                    disp_z += vel_z * dt

            prev_az, prev_t = az, t

        dz_imu = disp_z
        gyro_z_mean = float(np.mean(gyro_zs)) if gyro_zs else 0.0

    return dz_lidar, dz_imu, gyro_z_mean, fitness


# ------------------------------------------------------------------ #
#  主窗口
# ------------------------------------------------------------------ #
class ImuLidarConflictWindow(ttk.Toplevel):

    def __init__(self, master, bag_path, topic_info):
        super().__init__(master)
        self.title("IMU / LiDAR 模态冲突检测（电梯场景）")
        self.geometry("1100x780")

        self.bag_path = bag_path
        self.topic_info = topic_info

        # 状态变量
        self.imu_topic_var    = tk.StringVar()
        self.lidar_topic_var  = tk.StringVar()
        self.threshold_var    = tk.StringVar(value="0.05")
        self.progress_var     = tk.IntVar(value=0)
        self.status_var       = tk.StringVar(value="就绪")

        # 计算结果缓存（用于阈值实时重绘）
        self._result = None

        self._imu_topics, self._lidar_topics = self._auto_detect_topics()
        self._build_ui()

    # ---------------------------------------------------------------- #
    #  话题自动检测
    # ---------------------------------------------------------------- #
    def _auto_detect_topics(self):
        imu_topics, lidar_topics = [], []
        for t, info in self.topic_info.items():
            mt = info.msg_type
            if mt == "sensor_msgs/Imu":
                imu_topics.append(t)
            elif mt in ("sensor_msgs/PointCloud2",
                        "livox_ros_driver/CustomMsg",
                        "livox_ros_driver2/CustomMsg"):
                lidar_topics.append(t)
        if imu_topics:
            self.imu_topic_var.set(imu_topics[0])
        if lidar_topics:
            self.lidar_topic_var.set(lidar_topics[0])
        return imu_topics, lidar_topics

    # ---------------------------------------------------------------- #
    #  UI 构建
    # ---------------------------------------------------------------- #
    def _build_ui(self):
        # --- 顶部控制区 ---
        ctrl = ttk.Labelframe(self, text="配置", padding=8)
        ctrl.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(ctrl, text="IMU 话题:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ctrl, textvariable=self.imu_topic_var,
                     values=self._imu_topics, width=32).grid(row=0, column=1, padx=5)

        ttk.Label(ctrl, text="LiDAR 话题:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        ttk.Combobox(ctrl, textvariable=self.lidar_topic_var,
                     values=self._lidar_topics, width=32).grid(row=0, column=3, padx=5)

        ttk.Label(ctrl, text="冲突阈值 (m):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        thresh_entry = ttk.Entry(ctrl, textvariable=self.threshold_var, width=10)
        thresh_entry.grid(row=1, column=1, sticky="w", padx=5, pady=(6, 0))
        thresh_entry.bind("<Return>", lambda e: self._redraw_threshold())

        self._start_btn = ttk.Button(ctrl, text="▶  开始计算",
                                     command=self._on_start_clicked, bootstyle="warning")
        self._start_btn.grid(row=1, column=2, padx=(15, 5), pady=(6, 0))

        ttk.Button(ctrl, text="重新绘制阈值线",
                   command=self._redraw_threshold, bootstyle="secondary-outline").grid(
            row=1, column=3, padx=5, pady=(6, 0), sticky="w")

        # --- 进度区 ---
        prog_frame = ttk.Frame(self, padding=(10, 2))
        prog_frame.pack(fill="x")
        ttk.Progressbar(prog_frame, variable=self.progress_var,
                        maximum=100, bootstyle="warning-striped").pack(fill="x", pady=2)
        ttk.Label(prog_frame, textvariable=self.status_var, foreground="gray").pack(anchor="w")

        # --- 图表区 ---
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

    # ---------------------------------------------------------------- #
    #  计算入口
    # ---------------------------------------------------------------- #
    def _on_start_clicked(self):
        if not self.imu_topic_var.get() or not self.lidar_topic_var.get():
            messagebox.showwarning("提示", "请先选择 IMU 和 LiDAR 话题喵～", parent=self)
            return
        self._start_btn.config(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("初始化...")
        threading.Thread(target=self._run_computation, daemon=True).start()

    def _update_progress(self, value: int, text: str):
        self.progress_var.set(value)
        self.status_var.set(text)

    # ---------------------------------------------------------------- #
    #  后台计算线程
    # ---------------------------------------------------------------- #
    def _run_computation(self):
        try:
            imu_topic   = self.imu_topic_var.get()
            lidar_topic = self.lidar_topic_var.get()

            # 阶段1：重力估计
            self.after(0, lambda: self._update_progress(3, "估计重力向量（前50帧）..."))
            imu_reader = BagCacheReader(self.bag_path)
            gravity_unit, g_mag = self._estimate_gravity(imu_reader, imu_topic)

            # 阶段2：一次性加载所有原始数据
            self.after(0, lambda: self._update_progress(10, "加载 LiDAR 帧..."))
            lidar_reader = BagCacheReader(self.bag_path)
            lidar_frames, lidar_msg_type, lidar_msg_def = self._load_lidar_frames(lidar_reader, lidar_topic)

            self.after(0, lambda: self._update_progress(20, "加载 IMU 数据..."))
            imu_reader2 = BagCacheReader(self.bag_path)
            imu_data = self._load_imu_data(imu_reader2, imu_topic)

            # 阶段3：并行 ICP + IMU 积分
            self.after(0, lambda: self._update_progress(30, "并行 ICP 配准中..."))
            results = self._parallel_process_pairs(lidar_frames, lidar_msg_type, lidar_msg_def, imu_data, gravity_unit, g_mag)

            self.after(0, lambda: self._on_computation_done(results))

        except CacheNotFoundError as e:
            self.after(0, lambda err=e: self._on_error(
                f"缓存未找到，请先在主界面对该话题生成缓存喵～\n{err}"))
        except Exception as e:
            self.after(0, lambda err=e: self._on_error(str(err)))

    # ---------------------------------------------------------------- #
    #  算法：重力估计
    # ---------------------------------------------------------------- #
    def _estimate_gravity(self, reader, imu_topic, n=50):
        reader.load_topic(imu_topic)
        count = min(n, reader.get_message_count())
        accum = np.zeros(3)
        for i in range(count):
            msg, _ = reader.get_message(i)
            a = msg.linear_acceleration
            accum += np.array([a.x, a.y, a.z])
        g_vec = accum / count
        g_mag = np.linalg.norm(g_vec)
        return g_vec / (g_mag + 1e-9), g_mag

    # ---------------------------------------------------------------- #
    #  数据加载：一次性读取所有帧到内存
    # ---------------------------------------------------------------- #
    def _load_lidar_frames(self, reader, topic):
        reader.load_topic(topic)
        total = reader.get_message_count()
        msg_type, msg_def = reader.get_msg_def(topic)
        frames = []
        for i in range(total):
            _, raw_data, ts = reader.get_raw(i)
            frames.append((ts.to_sec(), msg_type, raw_data))
        self.after(0, lambda: self._update_progress(19, f"LiDAR 加载完成: {total} 帧"))
        return frames, msg_type, msg_def

    def _load_imu_data(self, reader, topic):
        reader.load_topic(topic)
        total = reader.get_message_count()
        data = []
        for i in range(total):
            msg, ts = reader.get_message(i)
            a = msg.linear_acceleration
            w = msg.angular_velocity
            data.append((ts.to_sec(), a.x, a.y, a.z, w.z))
        self.after(0, lambda: self._update_progress(29, f"IMU 加载完成: {total} 条"))
        return data

    # ---------------------------------------------------------------- #
    #  并行处理：每对相邻LiDAR帧独立提交线程池
    # ---------------------------------------------------------------- #
    def _parallel_process_pairs(self, lidar_frames, lidar_msg_type, lidar_msg_def, imu_data, gravity_unit, g_mag):
        """
        对每对相邻LiDAR帧 (i, i+1)，找出对应时间窗口内的IMU数据，
        并行执行 ICP + IMU积分。
        返回 list of (t0, dz_lidar, dz_imu, gyro_z_mean)，按帧顺序排列。
        """
        n = len(lidar_frames)
        if n < 2:
            return []

        imu_times = [d[0] for d in imu_data]

        # 构建任务列表，直接传 raw bytes，子进程里再解析
        task_args = []
        task_t0   = []
        for i in range(n - 1):
            t0, _, raw0 = lidar_frames[i]
            t1, _, raw1 = lidar_frames[i + 1]
            lo = bisect.bisect_left(imu_times, t0)
            hi = bisect.bisect_right(imu_times, t1)
            imu_seg = imu_data[lo:hi]
            task_args.append((lidar_msg_type, lidar_msg_def, raw0, raw1, imu_seg, gravity_unit, g_mag))
            task_t0.append(t0)

        total     = len(task_args)
        results   = [None] * total
        completed = 0
        last_pct  = -1
        n_workers = max(1, os.cpu_count() or 1)

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_icp_imu_worker, *args): idx
                for idx, args in enumerate(task_args)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                dz_lidar, dz_imu, gyro_z_mean, _ = future.result()
                results[idx] = (task_t0[idx], dz_lidar, dz_imu, gyro_z_mean)
                completed += 1
                pct = 30 + int(completed / total * 69)
                if pct != last_pct:  # 只在百分比变化时才更新UI
                    last_pct = pct
                    self.after(0, lambda v=pct, c=completed, tot=total:
                               self._update_progress(v, f"ICP 配准: {c}/{tot}"))

        return results

    # ---------------------------------------------------------------- #
    #  计算完成回调
    # ---------------------------------------------------------------- #
    def _on_computation_done(self, results):
        self._update_progress(100, "计算完成")
        self._start_btn.config(state="normal")
        self._result = results
        self._plot_results(results)

    def _on_error(self, msg):
        self._update_progress(0, f"错误: {msg}")
        self._start_btn.config(state="normal")
        messagebox.showerror("计算失败", msg, parent=self)

    def _redraw_threshold(self):
        if self._result:
            self._plot_results(self._result)

    # ---------------------------------------------------------------- #
    #  绘图
    # ---------------------------------------------------------------- #
    def _plot_results(self, results):
        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            threshold = 0.05

        if not results:
            return

        times      = np.array([r[0] for r in results])
        dz_lidar   = np.array([r[1] for r in results])
        dz_imu     = np.array([r[2] for r in results])
        gyro_z     = np.array([r[3] for r in results])

        t_rel   = times - times[0]
        diff    = np.abs(dz_lidar - dz_imu)
        conflict = diff > threshold

        # --- 子图1：每帧相对 dz 对比 ---
        self.ax1.clear()
        self.ax1.plot(t_rel, dz_lidar, color='darkorange', lw=1.2,
                      marker='.', ms=3, label='LiDAR ICP dz')
        self.ax1.plot(t_rel, dz_imu,   color='steelblue',  lw=1.2,
                      marker='.', ms=3, label='IMU 积分 dz')
        self.ax1.axhline(0, color='gray', lw=0.5)
        self.ax1.set_ylabel("相对 Z 位移 (m)")
        self.ax1.set_title("每帧相对 Z 位移：LiDAR ICP vs IMU 积分")
        self.ax1.legend(fontsize=9)
        self.ax1.grid(True, alpha=0.35)

        # --- 子图2：差值绝对值 ---
        self.ax2.clear()
        self.ax2.plot(t_rel, diff, color='crimson', lw=1.0,
                      label='|LiDAR_dz - IMU_dz|')
        self.ax2.axhline(threshold, color='red', ls='--', lw=1.5,
                         label=f'阈值 {threshold} m')
        self.ax2.fill_between(t_rel, diff, threshold, where=conflict,
                              alpha=0.25, color='red', label='冲突区间')
        n_conflict = int(np.sum(conflict))
        self.ax2.set_title(f"差值绝对值  |  冲突帧数: {n_conflict} / {len(t_rel)}")
        self.ax2.set_ylabel("|差值| (m)")
        self.ax2.legend(fontsize=9)
        self.ax2.grid(True, alpha=0.35)

        # --- 子图3：IMU 角速度Z（每区间均值）---
        self.ax3.clear()
        self.ax3.plot(t_rel, gyro_z, color='mediumpurple', lw=0.8,
                      label='IMU ω_z 均值 (rad/s)')
        self.ax3.axhline(0, color='gray', lw=0.5)
        self.ax3.set_xlabel("时间 (s)")
        self.ax3.set_ylabel("ω_z (rad/s)")
        self.ax3.set_title("IMU 角速度 Z（辅助判断旋转干扰）")
        self.ax3.legend(fontsize=9)
        self.ax3.grid(True, alpha=0.35)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()


# ------------------------------------------------------------------ #
#  插件入口
# ------------------------------------------------------------------ #
class ImuLidarConflictPlugin(RosBagPluginBase):

    def __init__(self, context):
        super().__init__(context)
        self._window = None

    def get_name(self) -> str:
        return "IMU/LiDAR冲突检测"

    def get_button_style(self) -> str:
        return "warning"

    def on_start(self):
        if self._window and self._window.winfo_exists():
            self._window.lift()
            return
        self._window = ImuLidarConflictWindow(
            self.context.master,
            self.context.bag_file_path,
            self.context.topic_info
        )

    def on_frame_changed(self, index: int, is_high_quality: bool) -> bool:
        return False
