# plugins/reprojection_plugin.py
import os
import json
import numpy as np
import tkinter as tk
from tkinter import messagebox
from plugin_core import RosBagPluginBase
from reprojection_viewer import ReprojectionViewer

# 历史配置保存路径（与插件同目录）
CALIB_STORE_PATH = os.path.join(os.path.dirname(__file__), ".calib_history.json")


class CalibrationDialog(tk.Toplevel):
    """内外参填写对话框，支持命名保存与历史记忆"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("相机标定参数配置")
        self.resizable(False, False)
        self.result = None  # 确认后为 (name, K, dist, T_cam_lidar)，取消为 None

        self._history = self._load_history()
        self._build_ui()
        self.grab_set()
        self.wait_window()

    # ------------------------------------------------------------------ #
    #  持久化
    # ------------------------------------------------------------------ #
    def _load_history(self):
        if os.path.exists(CALIB_STORE_PATH):
            try:
                with open(CALIB_STORE_PATH, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_history(self):
        with open(CALIB_STORE_PATH, 'w') as f:
            json.dump(self._history, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    #  UI 构建
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # ---- 左侧：历史列表 ----
        left = tk.Frame(self, padx=8, pady=8)
        left.grid(row=0, column=0, sticky='ns')

        tk.Label(left, text="历史配置", font=('', 10, 'bold')).pack(anchor='w')

        self._listbox = tk.Listbox(left, width=22, height=20, selectmode=tk.SINGLE)
        self._listbox.pack(fill='both', expand=True)
        self._listbox.bind('<<ListboxSelect>>', self._on_select)
        for name in self._history:
            self._listbox.insert(tk.END, name)

        tk.Button(left, text="删除选中", command=self._delete_selected).pack(
            fill='x', pady=(4, 0)
        )

        # ---- 右侧：表单 ----
        right = tk.Frame(self, padx=12, pady=8)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)

        row = 0

        # 配置名称（必填）
        tk.Label(right, text="配置名称（必填）", font=('', 9, 'bold')).grid(
            row=row, column=0, sticky='w'
        )
        row += 1
        self._name_var = tk.StringVar()
        tk.Entry(right, textvariable=self._name_var, width=52).grid(
            row=row, column=0, sticky='ew', pady=(0, 10)
        )
        row += 1

        # 内参矩阵 K
        tk.Label(right, text="内参矩阵 K（3×3，行优先，共 9 个值）", font=('', 9, 'bold')).grid(
            row=row, column=0, sticky='w'
        )
        row += 1
        tk.Label(right, text="格式：fx  0  cx  0  fy  cy  0  0  1", fg='gray').grid(
            row=row, column=0, sticky='w'
        )
        row += 1
        self._k_var = tk.StringVar(value="1000 0 640 0 1000 360 0 0 1")
        tk.Entry(right, textvariable=self._k_var, width=52).grid(
            row=row, column=0, sticky='ew', pady=(0, 10)
        )
        row += 1

        # 畸变系数
        tk.Label(right, text="畸变系数 dist（5 个值：k1 k2 p1 p2 k3）", font=('', 9, 'bold')).grid(
            row=row, column=0, sticky='w'
        )
        row += 1
        self._dist_var = tk.StringVar(value="0 0 0 0 0")
        tk.Entry(right, textvariable=self._dist_var, width=52).grid(
            row=row, column=0, sticky='ew', pady=(0, 10)
        )
        row += 1

        # 外参矩阵 T_cam_lidar
        tk.Label(right, text="外参矩阵 T_cam_lidar（4×4，行优先，共 16 个值）", font=('', 9, 'bold')).grid(
            row=row, column=0, sticky='w'
        )
        row += 1
        tk.Label(right, text="格式：旋转矩阵 R(3×3) + 平移 t(3×1) 组成的齐次矩阵", fg='gray').grid(
            row=row, column=0, sticky='w'
        )
        row += 1
        self._t_var = tk.StringVar(value="1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1")
        tk.Entry(right, textvariable=self._t_var, width=52).grid(
            row=row, column=0, sticky='ew', pady=(0, 16)
        )
        row += 1

        # 按钮行
        btn_row = tk.Frame(right)
        btn_row.grid(row=row, column=0, sticky='e')
        tk.Button(btn_row, text="取消", padx=10, command=self.destroy).pack(side='right', padx=(4, 0))
        tk.Button(
            btn_row, text="保存并确认", padx=10,
            bg='#4CAF50', fg='white', command=self._on_ok
        ).pack(side='right')

    # ------------------------------------------------------------------ #
    #  事件处理
    # ------------------------------------------------------------------ #
    def _on_select(self, _event):
        sel = self._listbox.curselection()
        if not sel:
            return
        name = self._listbox.get(sel[0])
        cfg = self._history.get(name, {})
        self._name_var.set(name)
        self._k_var.set(' '.join(str(v) for v in cfg.get('K', [])))
        self._dist_var.set(' '.join(str(v) for v in cfg.get('dist', [])))
        self._t_var.set(' '.join(str(v) for v in cfg.get('T_cam_lidar', [])))

    def _delete_selected(self):
        sel = self._listbox.curselection()
        if not sel:
            return
        name = self._listbox.get(sel[0])
        if messagebox.askyesno("确认删除", f"确定要删除配置 '{name}' 吗？", parent=self):
            self._history.pop(name, None)
            self._save_history()
            self._listbox.delete(sel[0])

    def _parse_floats(self, s, count, label):
        try:
            vals = [float(x) for x in s.split()]
        except ValueError:
            raise ValueError(f"「{label}」包含非数字字符")
        if len(vals) != count:
            raise ValueError(f"「{label}」需要 {count} 个数值，当前输入了 {len(vals)} 个")
        return vals

    def _on_ok(self):
        name = self._name_var.get().strip()
        if not name:
            messagebox.showwarning("提示", "配置名称不能为空，请先填写名称喵～", parent=self)
            return
        try:
            k_vals = self._parse_floats(self._k_var.get(), 9, "内参矩阵 K")
            dist_vals = self._parse_floats(self._dist_var.get(), 5, "畸变系数")
            t_vals = self._parse_floats(self._t_var.get(), 16, "外参矩阵 T_cam_lidar")
        except ValueError as e:
            messagebox.showerror("输入错误", str(e), parent=self)
            return

        K = np.array(k_vals).reshape(3, 3)
        dist = np.array(dist_vals)
        T = np.array(t_vals).reshape(4, 4)

        # 保存到历史
        self._history[name] = {
            'K': k_vals,
            'dist': dist_vals,
            'T_cam_lidar': t_vals,
        }
        self._save_history()

        # 如果是新名称则追加到列表
        existing = list(self._listbox.get(0, tk.END))
        if name not in existing:
            self._listbox.insert(tk.END, name)

        self.result = (name, K, dist, T)
        self.destroy()


# ====================================================================== #
#  插件主体
# ====================================================================== #
class ReprojectionPlugin(RosBagPluginBase):
    def __init__(self, context):
        super().__init__(context)
        self.viewer_window = None

    def get_name(self) -> str:
        return "反投影分析"

    def get_button_style(self) -> str:
        return "success"

    def _auto_detect_topics(self):
        image_candidates = []
        lidar_candidates = []
        for topic_name, info in self.context.topic_info.items():
            msg_type = info.msg_type
            if msg_type in ["sensor_msgs/Image", "sensor_msgs/CompressedImage"]:
                image_candidates.append(topic_name)
            elif "PointCloud2" in msg_type or "livox" in msg_type or "lidar" in topic_name:
                lidar_candidates.append(topic_name)
        return (
            image_candidates[0] if image_candidates else None,
            lidar_candidates[0] if lidar_candidates else None,
        )

    def on_start(self):
        if self.viewer_window and self.viewer_window.winfo_exists():
            self.viewer_window.lift()
            return

        # 弹出内外参填写对话框
        dlg = CalibrationDialog(self.context.master)
        if dlg.result is None:
            return  # 用户取消

        _name, K, dist, T_cam_lidar = dlg.result

        img_topic, lidar_topic = self._auto_detect_topics()
        if not img_topic or not lidar_topic:
            messagebox.showwarning("提示", "未能自动检测到完整的图像/点云话题。", parent=self.context.master)
            return

        bag_path = self.context.bag_file_path
        self.viewer_window = ReprojectionViewer(self.context.master)
        self.viewer_window.configure_data_sources(
            bag_path, img_topic, lidar_topic, K, T_cam_lidar, dist
        )

        current_index = self.context.get_current_index()
        if current_index >= 0:
            self.viewer_window.update_view_by_index(current_index)

    def on_frame_changed(self, index: int, is_high_quality: bool) -> bool:
        return False
