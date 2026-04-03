import os
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import sensor_msgs.point_cloud2 as pc2
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from plugin_core import RosBagPluginBase


class AdaptiveVoxelPController:
    class Config:
        def __init__(self, target_points=1500.0, alpha=1.2, min_voxel=0.03, max_voxel=0.8):
            self.target_points = target_points
            self.alpha = alpha
            self.min_voxel = min_voxel
            self.max_voxel = max_voxel

    def __init__(self, initial_voxel=0.1, cfg=None):
        self.cfg = cfg or self.Config()
        self.voxel_ = initial_voxel
        self._clamp_state()

    def voxel(self):
        return self.voxel_

    def update(self, n_points):
        if self.cfg.alpha <= 0.0 or self.cfg.target_points <= 0.0:
            return self.voxel_
        ratio = float(n_points) / self.cfg.target_points
        scale = np.power(max(ratio, 1e-6), 1.0 / self.cfg.alpha)
        self.voxel_ *= scale
        self._clamp_state()
        return self.voxel_

    def _clamp_state(self):
        if self.cfg.alpha <= 0.0:
            self.cfg.alpha = 1.0
        if self.cfg.target_points <= 0.0:
            self.cfg.target_points = 1.0
        if self.cfg.min_voxel <= 0.0:
            self.cfg.min_voxel = 0.001
        if self.cfg.max_voxel < self.cfg.min_voxel:
            self.cfg.max_voxel = self.cfg.min_voxel
        self.voxel_ = float(np.clip(self.voxel_, self.cfg.min_voxel, self.cfg.max_voxel))


class ElevatorDoorDetectorWindow(ttk.Toplevel):
    ROI_CONFIG = {
        "y_limit": 1.2,
        "z_min": -0.3,
        "z_max": 2.3,
        "x_min": 0.2,
        "x_max": 6.0,
    }
    VOXEL_CONFIG = {
        "target_points": 1500.0,
        "alpha": 1.2,
        "min_voxel": 0.03,
        "max_voxel": 0.8,
        "initial_voxel": 0.08,
    }

    def __init__(self, master, viewer, topic, index_data):
        super().__init__(master)
        self.title("电梯门关闭检测")
        self.geometry("1280x860")

        self.viewer = viewer
        self.topic = topic
        self.index_data = index_data or []
        self.cache_path = self.viewer._get_cache_paths(self.topic)[1]
        self.msg_type = self.viewer.topic_info[self.topic].msg_type
        self.cn_font = self._get_chinese_font()

        self.status_var = tk.StringVar(value="准备分析电梯门关闭状态...")
        self.config_vars = {
            "max_distance_threshold": tk.StringVar(value="3.20"),
            "window_size": tk.StringVar(value="5"),
            "closed_percent": tk.StringVar(value="80"),
        }

        self.metrics = None
        self.analysis_running = False
        self._hover_lines = []
        self._hover_annots = {}
        self._right_click_ax = None

        self._create_ui()
        self._set_initial_message()

    def _create_ui(self):
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, textvariable=self.status_var, font=("Noto Sans CJK SC", 12, "bold")).pack(
            side="top", fill="x"
        )

        settings = ttk.Frame(control_frame)
        settings.pack(side="top", fill="x", pady=(8, 4))

        self._add_setting(settings, "最大距离阈值:", "max_distance_threshold", 0)
        self._add_setting(settings, "统计窗口:", "window_size", 1)
        self._add_setting(settings, "关闭百分比(%):", "closed_percent", 2)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side="top", fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(btn_frame, text="开始检测", command=self._start_analysis, bootstyle="success")
        self.run_btn.pack(side="left")
        self.reset_btn = ttk.Button(btn_frame, text="恢复默认参数", command=self._reset_defaults, bootstyle="outline")
        self.reset_btn.pack(side="left", padx=(8, 0))

        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax_state = self.fig.add_subplot(221)
        self.ax_distance = self.fig.add_subplot(222)
        self.ax_delta = self.fig.add_subplot(223)
        self.ax_voxel = self.fig.add_subplot(224)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._setup_plot_context_menu()

    def _add_setting(self, parent, label, key, column):
        ttk.Label(parent, text=label).grid(row=0, column=column * 2, padx=(0, 4), pady=2, sticky="e")
        ttk.Entry(parent, textvariable=self.config_vars[key], width=8).grid(
            row=0, column=column * 2 + 1, padx=(0, 12), pady=2, sticky="w"
        )

    def _reset_defaults(self):
        defaults = {
            "max_distance_threshold": "3.20",
            "window_size": "5",
            "closed_percent": "80",
        }
        for key, value in defaults.items():
            self.config_vars[key].set(value)

    def _set_initial_message(self):
        for ax in (self.ax_state, self.ax_distance, self.ax_delta, self.ax_voxel):
            ax.clear()
            ax.text(0.5, 0.5, "点击“开始检测”分析当前话题", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()

    def _start_analysis(self):
        if self.analysis_running:
            return
        if not self.index_data:
            messagebox.showwarning("提示", "当前话题没有可分析的帧。", parent=self)
            return

        try:
            params = {
                "max_distance_threshold": float(self.config_vars["max_distance_threshold"].get().strip()),
                "window_size": int(self.config_vars["window_size"].get().strip()),
                "closed_percent": float(self.config_vars["closed_percent"].get().strip()),
            }
        except ValueError:
            messagebox.showwarning("提示", "参数格式错误，请输入数字。", parent=self)
            return

        if params["max_distance_threshold"] <= 0:
            messagebox.showwarning("提示", "最大距离阈值必须大于 0。", parent=self)
            return
        if params["window_size"] <= 0:
            messagebox.showwarning("提示", "统计窗口必须大于 0。", parent=self)
            return
        if not (0.0 < params["closed_percent"] <= 100.0):
            messagebox.showwarning("提示", "关闭百分比必须在 0 到 100 之间。", parent=self)
            return

        self.analysis_running = True
        self.run_btn.config(state="disabled")
        self.status_var.set(f"正在分析 {len(self.index_data)} 帧点云的关门状态...")
        threading.Thread(target=self._run_analysis, args=(params,), daemon=True).start()

    def _run_analysis(self, params):
        controller = AdaptiveVoxelPController(
            initial_voxel=self.VOXEL_CONFIG["initial_voxel"],
            cfg=AdaptiveVoxelPController.Config(
                target_points=self.VOXEL_CONFIG["target_points"],
                alpha=self.VOXEL_CONFIG["alpha"],
                min_voxel=self.VOXEL_CONFIG["min_voxel"],
                max_voxel=self.VOXEL_CONFIG["max_voxel"],
            ),
        )

        frame_ids = []
        max_distances = []
        p95_distances = []
        p99_distances = []
        voxel_values = []
        roi_counts = []

        total = len(self.index_data)
        for i, (offset, size) in enumerate(self.index_data):
            try:
                points = self._load_points_from_cache(offset, size)
                roi_points = self._extract_roi(points)
                frame_ids.append(i + 1)

                if len(roi_points) == 0:
                    max_distances.append(np.nan)
                    p95_distances.append(np.nan)
                    p99_distances.append(np.nan)
                    voxel_values.append(controller.voxel())
                    roi_counts.append(0)
                    continue

                voxel_values.append(controller.voxel())
                sampled = self._voxel_downsample(roi_points, controller.voxel())
                controller.update(len(sampled))
                x_vals = sampled[:, 0]

                max_distances.append(float(np.nanmax(x_vals)) if len(x_vals) else np.nan)
                p95_distances.append(float(np.nanpercentile(x_vals, 95)) if len(x_vals) else np.nan)
                p99_distances.append(float(np.nanpercentile(x_vals, 99)) if len(x_vals) else np.nan)
                roi_counts.append(len(sampled))

                if i % 10 == 0 or i == total - 1:
                    self.after(0, lambda c=i + 1: self.status_var.set(f"正在分析电梯门关闭状态... {c}/{total}"))
            except Exception as exc:
                print(f"[elevator_door_detector_plugin] frame {i + 1} failed: {exc}")
                frame_ids.append(i + 1)
                max_distances.append(np.nan)
                p95_distances.append(np.nan)
                p99_distances.append(np.nan)
                voxel_values.append(controller.voxel())
                roi_counts.append(0)

        raw_closed = []
        threshold = params["max_distance_threshold"]
        for max_distance in max_distances:
            raw_closed.append(1.0 if np.isfinite(max_distance) and max_distance <= threshold else 0.0)

        closed_scores = []
        closed_ratios = []
        state_values = []
        required_ratio = params["closed_percent"] / 100.0
        window = params["window_size"]

        for i, max_distance in enumerate(max_distances):
            if not np.isfinite(max_distance):
                closed_scores.append(np.nan)
            else:
                closed_scores.append(float(threshold - max_distance))

            begin = max(0, i - window + 1)
            window_flags = raw_closed[begin:i + 1]
            ratio = float(np.mean(window_flags)) if window_flags else np.nan
            closed_ratios.append(ratio)
            state_values.append(1.0 if np.isfinite(ratio) and ratio >= required_ratio else 0.0)

        self.metrics = {
            "frame": np.asarray(frame_ids, dtype=float),
            "max_distance": np.asarray(max_distances, dtype=float),
            "p95_distance": np.asarray(p95_distances, dtype=float),
            "p99_distance": np.asarray(p99_distances, dtype=float),
            "closed_score": np.asarray(closed_scores, dtype=float),
            "closed_ratio": np.asarray(closed_ratios, dtype=float),
            "raw_closed": np.asarray(raw_closed, dtype=float),
            "door_closed": np.asarray(state_values, dtype=float),
            "voxel": np.asarray(voxel_values, dtype=float),
            "roi_points": np.asarray(roi_counts, dtype=float),
            "closed_threshold": threshold,
            "params": params,
        }
        self.after(0, self._finalize_analysis)

    def _finalize_analysis(self):
        self.analysis_running = False
        self.run_btn.config(state="normal")
        self._plot_metrics()

    def _plot_metrics(self):
        metrics = self.metrics
        if not metrics:
            return

        frame = metrics["frame"]
        threshold = metrics["closed_threshold"]
        font_kwargs = {"fontproperties": self.cn_font} if self.cn_font else {}

        self.ax_state.clear()
        self.ax_distance.clear()
        self.ax_delta.clear()
        self.ax_voxel.clear()

        line_state, = self.ax_state.plot(frame, metrics["door_closed"], "-", color="#27ae60", linewidth=1.8, label="door_closed")
        line_raw, = self.ax_state.plot(frame, metrics["raw_closed"], "-", color="#7f8c8d", linewidth=1.2, label="raw_closed")
        self.ax_state.set_title("电梯门关闭状态", **font_kwargs)
        self.ax_state.set_xlabel("帧序号", **font_kwargs)
        self.ax_state.set_ylabel("状态", **font_kwargs)
        self.ax_state.set_ylim(-0.1, 1.2)
        self.ax_state.grid(True, linestyle="--", alpha=0.6)
        self.ax_state.legend(prop=self.cn_font)

        line_max, = self.ax_distance.plot(frame, metrics["max_distance"], "-", color="#2c3e50", linewidth=1.6, label="max_distance")
        line_p95, = self.ax_distance.plot(frame, metrics["p95_distance"], "-", color="#2980b9", linewidth=1.4, label="p95_distance")
        line_p99, = self.ax_distance.plot(frame, metrics["p99_distance"], "-", color="#16a085", linewidth=1.2, label="p99_distance")
        if np.isfinite(threshold):
            self.ax_distance.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.2, label="closed_threshold")
        self.ax_distance.set_title("最大距离关门判据", **font_kwargs)
        self.ax_distance.set_xlabel("帧序号", **font_kwargs)
        self.ax_distance.set_ylabel("距离 x (m)", **font_kwargs)
        self.ax_distance.grid(True, linestyle="--", alpha=0.6)
        self.ax_distance.legend(prop=self.cn_font)

        line_ratio, = self.ax_delta.plot(frame, metrics["closed_ratio"], "-", color="#16a085", linewidth=1.5, label="closed_ratio")
        line_ratio_thr, = self.ax_delta.plot(
            frame,
            np.full_like(frame, metrics["params"]["closed_percent"] / 100.0, dtype=float),
            "-",
            color="#c0392b",
            linewidth=1.2,
            label="closed_percent_threshold",
        )
        self.ax_delta.set_title("窗口关闭百分比", **font_kwargs)
        self.ax_delta.set_xlabel("帧序号", **font_kwargs)
        self.ax_delta.set_ylabel("比例", **font_kwargs)
        self.ax_delta.set_ylim(-0.05, 1.05)
        self.ax_delta.grid(True, linestyle="--", alpha=0.6)
        self.ax_delta.legend(prop=self.cn_font)

        line_voxel, = self.ax_voxel.plot(frame, metrics["voxel"], "-", color="#8e44ad", linewidth=1.5, label="adaptive_voxel")
        line_roi, = self.ax_voxel.plot(frame, metrics["roi_points"], "-", color="#7f8c8d", linewidth=1.2, label="roi_points")
        self.ax_voxel.set_title("自适应体素与 ROI 点数", **font_kwargs)
        self.ax_voxel.set_xlabel("帧序号", **font_kwargs)
        self.ax_voxel.set_ylabel("体素 / 点数", **font_kwargs)
        self.ax_voxel.grid(True, linestyle="--", alpha=0.6)
        self.ax_voxel.legend(prop=self.cn_font)

        self._hover_lines = [
            (self.ax_state, line_state),
            (self.ax_state, line_raw),
            (self.ax_distance, line_max),
            (self.ax_distance, line_p95),
            (self.ax_distance, line_p99),
            (self.ax_delta, line_ratio),
            (self.ax_delta, line_ratio_thr),
            (self.ax_voxel, line_voxel),
            (self.ax_voxel, line_roi),
        ]

        closed_frames = int(np.nansum(metrics["door_closed"] > 0.5))
        self.status_var.set(
            f"分析完成: 最大距离阈值 {threshold:.3f} m, 判定关闭 {closed_frames}/{len(frame)} 帧"
        )

        self.fig.tight_layout()
        self.canvas.draw()

    def _load_points_from_cache(self, offset, size):
        with open(self.cache_path, "rb") as f:
            f.seek(offset)
            msg_type, raw_data, _ = pickle.loads(f.read(size))
        msg = self.viewer._deserialize_raw_message(msg_type, raw_data, self.topic)
        if msg_type == "sensor_msgs/PointCloud2":
            points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64)
        elif msg_type in ("livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"):
            points = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float64)
        else:
            raise ValueError(f"不支持的消息类型: {msg_type}")
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return points

    def _extract_roi(self, points):
        if points is None or len(points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        roi = self.ROI_CONFIG
        mask = np.isfinite(points).all(axis=1)
        mask &= np.linalg.norm(points, axis=1) >= 0.05
        mask &= points[:, 0] >= roi["x_min"]
        mask &= points[:, 0] <= roi["x_max"]
        mask &= np.abs(points[:, 1]) <= roi["y_limit"]
        mask &= points[:, 2] >= roi["z_min"]
        mask &= points[:, 2] <= roi["z_max"]
        return points[mask]

    def _voxel_downsample(self, points, voxel):
        if len(points) == 0 or voxel <= 0:
            return points
        coords = np.floor(points / voxel).astype(np.int64)
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        unique_idx.sort()
        return points[unique_idx]

    def _setup_plot_context_menu(self):
        self._plot_menu = tk.Menu(self, tearoff=0)
        self._plot_menu.add_command(label="保存曲线数据到 CSV", command=self._save_plot_data_to_csv)
        self.canvas.get_tk_widget().bind("<Button-3>", self._on_plot_right_click)

    def _on_plot_right_click(self, event):
        self._right_click_ax = None
        x_fig = event.x / self.canvas.get_tk_widget().winfo_width()
        y_fig = 1.0 - event.y / self.canvas.get_tk_widget().winfo_height()
        for ax in self.fig.axes:
            if ax.get_position().contains(x_fig, y_fig):
                self._right_click_ax = ax
                break
        try:
            self._plot_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._plot_menu.grab_release()

    def _save_plot_data_to_csv(self):
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="保存曲线数据",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return

        try:
            target_axes = [self._right_click_ax] if self._right_click_ax is not None else self.fig.axes
            lines = []
            for ax in target_axes:
                for line in ax.lines:
                    label = line.get_label() if line.get_label() != "_nolegend_" else "series"
                    lines.append((label, line.get_xdata(), line.get_ydata()))

            if not lines:
                messagebox.showwarning("提示", "当前没有可保存的曲线数据。", parent=self)
                return

            max_len = max(len(x) for _, x, _ in lines)
            headers = []
            for i, (label, _, _) in enumerate(lines, start=1):
                headers.extend([f"{label}_x_{i}", f"{label}_y_{i}"])

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for idx in range(max_len):
                    row = []
                    for _, x, y in lines:
                        row.append(str(x[idx]) if idx < len(x) else "")
                        row.append(str(y[idx]) if idx < len(y) else "")
                    f.write(",".join(row) + "\n")

            messagebox.showinfo("完成", f"已保存曲线数据到:\n{file_path}", parent=self)
        except Exception as exc:
            messagebox.showerror("错误", f"保存失败: {exc}", parent=self)

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
        candidates = [line for line_ax, line in self._hover_lines if line_ax is ax]
        if not candidates:
            return

        best = None
        best_dist = 8.0
        for line in candidates:
            xdata = np.asarray(line.get_xdata(), dtype=np.float64)
            ydata = np.asarray(line.get_ydata(), dtype=np.float64)
            if len(xdata) == 0 or len(ydata) == 0:
                continue
            finite = np.isfinite(xdata) & np.isfinite(ydata)
            if not np.any(finite):
                continue
            pts = np.column_stack([xdata[finite], ydata[finite]])
            disp = ax.transData.transform(pts)
            dx = disp[:, 0] - event.x
            dy = disp[:, 1] - event.y
            dist = np.hypot(dx, dy)
            idx_local = int(np.argmin(dist))
            if dist[idx_local] < best_dist:
                idx = np.flatnonzero(finite)[idx_local]
                best_dist = dist[idx_local]
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
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    from matplotlib.font_manager import FontProperties

                    return FontProperties(fname=path, size=11)
                except Exception:
                    return None
        return None


class ElevatorDoorDetectorPlugin(RosBagPluginBase):
    def __init__(self, context):
        super().__init__(context)
        self.window = None

    def get_name(self) -> str:
        return "电梯门检测"

    def get_button_style(self) -> str:
        return "warning"

    def on_start(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return

        viewer = self.context._viewer
        topic = self.context.get_current_topic()
        if not topic:
            messagebox.showwarning("提示", "请先选择一个话题。")
            return
        if viewer.topic_status.get(topic) != "已完成":
            messagebox.showwarning("提示", "当前话题仍在索引中，请稍后再试。")
            return

        msg_type = viewer.topic_info[topic].msg_type
        if msg_type not in ("sensor_msgs/PointCloud2", "livox_ros_driver2/CustomMsg", "livox_ros_driver/CustomMsg"):
            messagebox.showwarning("提示", f"当前话题类型为 {msg_type}，插件仅支持点云话题。")
            return

        self.window = ElevatorDoorDetectorWindow(
            self.context.master,
            viewer=viewer,
            topic=topic,
            index_data=viewer.topic_indices.get(topic, []),
        )
