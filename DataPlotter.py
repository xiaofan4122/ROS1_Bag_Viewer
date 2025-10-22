# ------------------ 第 1 部分: 导入和配置 (包含一个关键修改) ------------------
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.font_manager import FontProperties
from ttkbootstrap.dialogs import Messagebox  # 保留，用于main函数的启动错误
from ttkbootstrap.toast import ToastNotification  # --- NEW: 导入 ToastNotification ---

try:
    import rosbag
    from bag_cache_reader import BagCacheReader
except ImportError as e:
    print(f"导入错误: {e}\n请确保已 source ROS 工作空间并安装了 rospkg。")
    exit()


# ------------------ 第 2 部分: 优化后的 DataPlotter 类 ------------------

class DataPlotter(ttk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("通用数据绘图器 (可交互)")
        self.geometry("1200x800")

        # ... (内部数据和状态变量与之前版本相同) ...
        self.bag_path = None
        self.all_topics = []
        self.data_reader = None
        self.current_topic = tk.StringVar()
        self.current_field = tk.StringVar()
        self.plotted_lines = {}
        self.cn_font = self._get_chinese_font()
        self._pan_pressed = False
        self._pan_start_x = None
        self._pan_start_y = None
        self._initial_view_limits = None

        self._create_widgets()
        self._configure_plot()
        self._connect_events()

    # --- NEW: 创建一个健壮的、支持窗口相对定位的Toast显示函数 ---
    def _show_toast(self, title, message, bootstyle="info", duration=2000):
        """
        在当前窗口(self)的右下角显示一个Toast通知。
        """
        # 1. 确保窗口的几何布局是最新的，以便获取准确的坐标和尺寸
        self.update_idletasks()

        # 2. 获取当前窗口左上角的屏幕坐标 (x, y)
        win_x = self.winfo_x()
        win_y = self.winfo_y()

        # 3. 获取当前窗口的宽度和高度
        win_width = self.winfo_width()
        win_height = self.winfo_height()

        # 获取整个电脑屏幕的宽度和高度
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # 4. 定义Toast距离窗口边缘的间距
        padding = 20

        # 5. 计算Toast右下角的目标屏幕坐标
        #    目标x = 窗口x + 窗口宽度 - 间距
        #    目标y = 窗口y + 窗口高度 - 间距
        pos_x = screen_width - (win_x + win_width - padding)
        pos_y = screen_height - (win_y + win_height - padding)

        # 6. 创建并显示Toast，使用计算出的绝对屏幕坐标
        toast = ToastNotification(
            title=title,
            message=message,
            duration=duration,
            bootstyle=bootstyle,
            position=(pos_x, pos_y, 'se'),  # 'se' 表示将Toast的东南(右下)角对齐到 (pos_x, pos_y)
            alert=True,  # 为警告和错误播放提示音
        )
        toast.show_toast()

    # ... (_get_chinese_font, _create_widgets, _configure_plot, 等方法保持不变) ...
    def _get_chinese_font(self):
        font_paths = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                      '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
        for path in font_paths:
            if os.path.exists(path): return FontProperties(fname=path, size=12)
        return None

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10);
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Labelframe(main_frame, text="绘图控制", padding=10);
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        control_frame.columnconfigure(1, weight=1);
        control_frame.columnconfigure(3, weight=1)
        ttk.Label(control_frame, text="选择话题:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.topic_combo = ttk.Combobox(control_frame, textvariable=self.current_topic, state="readonly");
        self.topic_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.topic_combo.bind("<<ComboboxSelected>>", self.on_topic_selected)
        ttk.Label(control_frame, text="选择数据字段:").grid(row=0, column=2, padx=(15, 5), pady=5, sticky="w")
        self.field_combo = ttk.Combobox(control_frame, textvariable=self.current_field, state="disabled");
        self.field_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        button_frame = ttk.Frame(control_frame);
        button_frame.grid(row=0, column=4, padx=(15, 5), pady=5)
        self.plot_button = ttk.Button(button_frame, text="绘制", command=self._plot_selected_data, bootstyle="success");
        self.plot_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = ttk.Button(button_frame, text="重置视图 (R)", command=self._reset_view,
                                       bootstyle="info-outline");
        self.reset_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = ttk.Button(button_frame, text="清空图表", command=self._clear_plot,
                                       bootstyle="danger-outline");
        self.clear_button.pack(side=tk.LEFT, padx=5)
        plot_frame = ttk.Frame(main_frame);
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10, 6), dpi=100);
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame);
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame);
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _configure_plot(self):
        font_kwargs = {'fontproperties': self.cn_font} if self.cn_font else {};
        self.ax.set_title("数据显示", **font_kwargs)
        self.ax.set_xlabel("相对时间戳 (s)", **font_kwargs);
        self.ax.set_ylabel("数值", **font_kwargs);
        self.ax.grid(True)
        self.fig.tight_layout();
        self.canvas.draw();
        self._initial_view_limits = (self.ax.get_xlim(), self.ax.get_ylim())

    def _connect_events(self):
        self.canvas.mpl_connect('scroll_event', self._on_scroll);
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release);
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('pick_event', self._on_pick);
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.bind('<KeyPress-r>', self._reset_view_event_handler);
        self.bind('<KeyPress-R>', self._reset_view_event_handler)

    def _reset_view(self):
        if not self.plotted_lines:
            if self._initial_view_limits: self.ax.set_xlim(self._initial_view_limits[0]); self.ax.set_ylim(
                self._initial_view_limits[1])
            self.canvas.draw_idle();
            return
        visible_lines = [line for line in self.plotted_lines.values() if line.get_visible()]
        if not visible_lines: self.canvas.draw_idle(); return
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        for line in visible_lines:
            x_data, y_data = line.get_xdata(), line.get_ydata()
            if len(x_data) > 0: min_x, max_x = min(min_x, np.nanmin(x_data)), max(max_x, np.nanmax(x_data))
            if len(y_data) > 0: min_y, max_y = min(min_y, np.nanmin(y_data)), max(max_y, np.nanmax(y_data))
        if np.isinf(min_x): return
        x_margin = (max_x - min_x) * 0.05 if max_x > min_x else 0.5;
        y_margin = (max_y - min_y) * 0.05 if max_y > min_y else 0.5
        self.ax.set_xlim(min_x - x_margin, max_x + x_margin);
        self.ax.set_ylim(min_y - y_margin, max_y + y_margin);
        self.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key in ('r', 'R'): self._reset_view()

    def _reset_view_event_handler(self, event=None):
        self._reset_view()

    def _on_scroll(self, event):
        ax = self.ax;
        if event.inaxes != ax or event.xdata is None: return
        x, y = event.xdata, event.ydata;
        x_min, x_max = ax.get_xlim();
        y_min, y_max = ax.get_ylim();
        scale_factor = 1.1 if event.button == 'up' else 1 / 1.1
        ax.set_xlim(x - (x - x_min) * scale_factor, x + (x_max - x) * scale_factor);
        ax.set_ylim(y - (y - y_min) * scale_factor, y + (y_max - y) * scale_factor)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes == self.ax and event.button in (2, 3):
            self._pan_pressed = True;
            self._pan_start_x, self._pan_start_y = event.xdata, event.ydata
            self.canvas.get_tk_widget().config(cursor="fleur")

    def _on_release(self, event):
        if self._pan_pressed: self._pan_pressed = False; self.canvas.get_tk_widget().config(cursor="")

    def _on_motion(self, event):
        if self._pan_pressed and event.inaxes == self.ax and event.xdata is not None:
            dx, dy = event.xdata - self._pan_start_x, event.ydata - self._pan_start_y
            x_min, x_max = self.ax.get_xlim();
            y_min, y_max = self.ax.get_ylim()
            self.ax.set_xlim(x_min - dx, x_max - dx);
            self.ax.set_ylim(y_min - dy, y_max - dy);
            self.canvas.draw_idle()

    def _on_pick(self, event):
        legline = event.artist;
        origline = self.plotted_lines.get(legline.get_label())
        if origline is None: return
        visible = not origline.get_visible();
        origline.set_visible(visible);
        legline.set_alpha(1.0 if visible else 0.2);
        self.fig.canvas.draw()

    # --- 以下是所有被修正为使用 Toast 的调用 ---

    def _plot_selected_data(self):
        topic = self.current_topic.get();
        field = self.current_field.get()
        if not topic or not field:
            # --- MODIFIED ---
            self._show_toast("操作无效", "请先选择一个话题和一个数据字段。", bootstyle="warning")
            return
        plot_key = f"{topic}/{field}"
        if plot_key in self.plotted_lines:
            # --- MODIFIED ---
            self._show_toast("提示", f"数据 '{plot_key}' 已在图上。", bootstyle="info")
            return
        try:
            timestamps, values = [], []
            self.data_reader.load_topic(topic)
            msg_count = self.data_reader.get_message_count()
            if msg_count == 0: return
            _, first_ts_ros = self.data_reader.get_message(0);
            start_time = first_ts_ros.to_sec()
            for i in range(msg_count):
                msg, ts_ros = self.data_reader.get_message(i);
                value = self._get_nested_attr(msg, field)
                if value is not None: timestamps.append(ts_ros.to_sec() - start_time); values.append(value)
            if not values:
                # --- MODIFIED ---
                self._show_toast("无数据", f"无法从话题中提取有效数据。", bootstyle="warning")
                return
            line, = self.ax.plot(timestamps, values, label=plot_key)
            self.plotted_lines[plot_key] = line
            legend = self.ax.legend(prop=self.cn_font)
            for legline in legend.get_lines(): legline.set_picker(True); legline.set_pickradius(5)
            self.canvas.draw()
            self._reset_view()
        except Exception as e:
            # --- MODIFIED ---
            self._show_toast("绘图失败", f"发生错误: {e}", bootstyle="danger")

    def _clear_plot(self):
        self.ax.clear();
        self.plotted_lines = {};
        self._configure_plot();
        self._reset_view()
        # --- NEW: 添加一个Toast通知 ---
        self._show_toast("操作完成", "图表已清空。", bootstyle="success")

    def configure_data_sources(self, bag_path, all_topics):
        self.bag_path = bag_path;
        self.all_topics = sorted(all_topics)
        self.topic_combo['values'] = self.all_topics
        try:
            self.data_reader = BagCacheReader(self.bag_path)
        except Exception as e:
            # --- MODIFIED ---
            self._show_toast("初始化失败", f"Reader错误: {e}", bootstyle="danger", duration=5000)
            self.destroy()

    def on_topic_selected(self, event=None):
        topic = self.current_topic.get()
        if not topic: return
        self.field_combo.set('');
        self.current_field.set('')
        try:
            self.data_reader.load_topic(topic)
            msg, _ = self.data_reader.get_message(0)
            fields = self._find_plottable_fields(msg)
            if fields:
                self.field_combo['values'] = sorted(fields);
                self.field_combo.config(state="readonly")
            else:
                self.field_combo['values'] = [];
                self.field_combo.config(state="disabled")
                # --- MODIFIED ---
                self._show_toast("提示", f"话题中没有可绘制的数值数据。", bootstyle="warning")
        except Exception as e:
            # --- MODIFIED ---
            self._show_toast("解析失败", f"解析话题时出错: {e}", bootstyle="danger", duration=5000)
            self.field_combo.config(state="disabled")

    def _find_plottable_fields(self, msg, prefix=''):
        fields = []
        if not hasattr(msg, '__slots__'): return []
        for slot in msg.__slots__:
            value = getattr(msg, slot)
            if isinstance(value, (int, float)):
                fields.append(f"{prefix}{slot}")
            elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0 and isinstance(value[0], (int, float)):
                for i in range(len(value)): fields.append(f"{prefix}{slot}[{i}]")
            elif hasattr(value, '__slots__') and slot != 'header':
                fields.extend(self._find_plottable_fields(value, prefix=f"{prefix}{slot}."))
        return fields

    def _get_nested_attr(self, obj, attr_string):
        try:
            parts = attr_string.replace(']', '').replace('[', '.').split('.');
            value = obj
            for part in parts: value = value[int(part)] if part.isdigit() else getattr(value, part)
            return value
        except (AttributeError, IndexError, TypeError):
            return None


# ------------------ 第 3 部分: 用于独立测试的 main 函数 (保持不变) ------------------
# main函数中的启动错误更适合用阻塞的Messagebox，因为它发生在主窗口建立之前。

if __name__ == '__main__':
    root = ttk.Window(themename="litera")
    root.withdraw()
    bag_path = filedialog.askopenfilename(
        title="请选择一个 Bag 文件进行测试",
        filetypes=[("ROS Bag files", "*.bag"), ("All files", "*.*")]
    )
    if not bag_path:
        print("未选择文件，测试退出。")
    else:
        try:
            topics = []
            with rosbag.Bag(bag_path, 'r') as bag:
                topics = list(bag.get_type_and_topic_info()[1].keys())
            if not topics:
                Messagebox.show_warning("这个 bag 文件中没有找到任何话题。", title="文件无内容", parent=root)
                root.destroy()
            else:
                plotter = DataPlotter(master=root)
                plotter.configure_data_sources(bag_path, topics)


                def on_closing():
                    root.quit()
                    root.destroy()


                plotter.protocol("WM_DELETE_WINDOW", on_closing)
                root.mainloop()
        except Exception as e:
            Messagebox.show_error(f"测试启动失败: \n{e}", title="严重错误", parent=root)
            if 'root' in locals() and root.winfo_exists():
                root.destroy()