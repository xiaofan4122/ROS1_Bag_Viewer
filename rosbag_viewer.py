# --- Start of Replacement Block ---
import numpy as np
import yaml
from tkinter import filedialog, simpledialog
import psutil
import ttkbootstrap as ttk
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.dialogs import Messagebox
# --- 新增结束 ---
import rospy
import rosbag
import threading
import json
import pickle
import os
import hashlib
import re
import concurrent.futures
import genpy.dynamic
import time
import queue
import struct

# --- 增加这些导入 ---
from plugin_core import ViewerContext
from plugins.reprojection_plugin import ReprojectionPlugin
from plugins.data_plotter_plugin import DataPlotterPlugin
from plugins.voxel_test_plugin import VoxelTestPlugin
from plugins.imu_lidar_conflict_plugin import ImuLidarConflictPlugin
# --- 消息解析辅助函数 (整合版) ---

BUILTIN_TYPES = {
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "float32", "float64", "string", "time", "duration", "char", "byte"
}


def type_info_enrich(type_name: str):
    """
    根据消息类型名称，解析出包名、短类型名，并判断是否为内置类型。
    这是一个为保证脚本独立性而重新实现的函数。
    """
    is_builtin = type_name in BUILTIN_TYPES
    short_type = type_name
    package = None
    if "/" in type_name:
        package, short_type = type_name.split("/", 1)
    return package, short_type, is_builtin


def parse_msg_section(raw_lines):
    fields = []
    constants = []
    const_re = re.compile(r"^\s*(?P<type>[A-Za-z0-9_/]+)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+?)\s*$")
    field_re = re.compile(
        r"^\s*(?P<type>[A-Za-z0-9_/]+)(?P<array>\[(?P<alen>\d*)\])?\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$")

    for raw in raw_lines:
        line = raw.split("#", 1)[0].strip()
        if not line: continue
        m = const_re.match(line)
        if m:
            constants.append({"type": m.group("type"), "name": m.group("name"), "value": m.group("value").strip()})
            continue
        m = field_re.match(line)
        if m:
            arr = m.group("array") is not None
            alen_txt = m.group("alen")
            arr_len = None if (alen_txt is None or alen_txt == "") else int(alen_txt)
            fields.append({"type": m.group("type"), "name": m.group("name"), "is_array": arr, "array_len": arr_len})
    return {"fields": fields, "constants": constants}


def split_msg_definitions(concatenated_def: str, primary_type: str):
    lines = concatenated_def.splitlines()
    blocks, current = [], []
    sep_re = re.compile(r"^=+$")
    for ln in lines:
        if sep_re.match(ln):
            if current: blocks.append(current)
            current = []
        else:
            current.append(ln)
    if current: blocks.append(current)

    out = {primary_type: [ln for ln in blocks[0] if ln.strip()]}
    for block in blocks[1:]:
        if block and block[0].startswith("MSG: "):
            tname = block[0][5:].strip()
            out[tname] = [ln for ln in block[1:] if ln.strip()]
    return out


def build_type_schemas_from_definition(concatenated_def: str, primary_type: str):
    """
    唯一的、功能完善的消息结构解析函数。
    """
    sections = split_msg_definitions(concatenated_def, primary_type)
    result = {"types": {}}
    for tname, lines in sections.items():
        parsed = parse_msg_section(lines)
        deps, has_header = [], False
        for f in parsed["fields"]:
            pkg, short, is_builtin = type_info_enrich(f["type"])
            if not is_builtin and f["type"] not in ("time", "duration"): deps.append(f["type"])
            if f["type"] in ("std_msgs/Header", "Header"): has_header = True
            f.update({"package": pkg, "short_type": short, "is_builtin": is_builtin})
        result["types"][tname] = {"fields": parsed["fields"], "constants": parsed["constants"],
                                  "has_header": has_header, "dependencies": sorted(set(deps))}
    return result


# --- End of Replacement Block ---

class RosBagViewer(ttk.Toplevel):
    MAX_WORKERS = min(4, os.cpu_count() or 1)
    MAX_DISPLAY_LEN = 10000
    CACHE_DIR = ".rosbag_cache"
    UI_POLL_INTERVAL = 100  # ms, 每秒轮询10次
    # --- 【核心改进 1】 定义索引条目的二进制格式 ---
    # >: Big-endian, Q: 8-byte unsigned integer. Total 16 bytes per entry.
    INDEX_FORMAT = '>QQ'
    INDEX_ENTRY_SIZE = struct.calcsize(INDEX_FORMAT)

    def __init__(self, master, bag_file):
        super().__init__(master)
        self.protocol("WM_DELETE_WINDOW", self.close)

        # --- 步骤1: 只进行最基础的变量初始化 ---
        self.bag_file_path = os.path.abspath(bag_file)
        self.image_viewer = None
        self.current_topic = None
        self.topic_indices = {}
        self.topic_status = {}
        self.topics = []
        self.topic_info = {}

        self.bag = None
        self.bag_lock = threading.Lock()
        self.indexing_executor = None
        self.ui_task_executor = None
        self.is_closing = threading.Event()
        self.ui_update_queue = queue.Queue()
        self.is_slider_dragging = False  # 新增状态变量，用于跟踪拖动状态
        self._msg_def_cache = {}  # (topic, msg_type) -> msg_def
        self._dynamic_class_cache = {}  # msg_type -> class

        # --- 新增：初始化插件系统 ---
        self.context = ViewerContext(self)
        self.plugins = [
            ReprojectionPlugin(self.context),
            DataPlotterPlugin(self.context),
            VoxelTestPlugin(self.context),
            # ImuLidarConflictPlugin(self.context)
        ]

        # --- 步骤2: 立即创建UI骨架 ---
        self.title("智能 ROS Bag 查看器 (最终修复版)")
        self.geometry("1200x800")
        self.minsize(900, 700)
        self.bold_font = font.Font(family="Noto Sans CJK SC", size=11, weight="bold")

        # 创建UI组件，此时它们内部的数据为空
        self._create_widgets()
        self._configure_layout()

        # --- 步骤3: 安排一个短延迟后，再开始加载数据和后台任务 ---
        self.after(50, self.initialize_ros_and_backend)

    def initialize_ros_and_backend(self):
        """在UI窗口稳定后，初始化所有ROS、后台组件，并填充UI数据"""
        try:
            rospy.init_node("rosbag_viewer", anonymous=True)
            self.bag = rosbag.Bag(self.bag_file_path, 'r')
            topic_info_dict = self.bag.get_type_and_topic_info()[1]
            self.topic_info = topic_info_dict
            self.topics = sorted(list(topic_info_dict.keys()))
            self.topic_status = {topic: "等待中" for topic in self.topics}
            self.current_topic = self.topics[0] if self.topics else None
        except Exception as e:
            messagebox.showerror("初始化失败", f"处理 Bag 文件时出错: {e}")
            self.destroy()
            return

        # --- 使用加载到的数据填充UI组件 ---
        self.topic_combo.config(values=self.topics)
        if self.current_topic:
            self.topic_combo.set(self.current_topic)

        # --- 初始化线程池 ---
        self.indexing_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS,
            thread_name_prefix='IndexingWorker',
            initializer=self._set_low_priority
        )
        self.ui_task_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix='UITaskWorker'
        )

        # --- 启动后台任务 ---
        self.after(100, self._start_background_indexing)
        self.after(self.UI_POLL_INTERVAL, self._ui_poller)

    @staticmethod
    def _set_low_priority():
        """【新】线程初始化函数，用于降低线程优先级"""
        p = psutil.Process(os.getpid())
        if os.name == 'nt':  # Windows
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:  # Linux, macOS
            p.nice(10)  # nice值越高，优先级越低

    # def _open_image_viewer(self):
    #     """打开或显示图像查看器窗口"""
    #     if self.image_viewer is None or not self.image_viewer.winfo_exists():
    #         # 确保 ImageViewer 已经被导入
    #         # from image_viewer_file import ImageViewer # (如果它在另一个文件里)
    #         self.image_viewer = ImageViewer(self.master, self)
    #     else:
    #         self.image_viewer.deiconify()  # 如果已存在但被隐藏，则重新显示
    #
    #     self.image_viewer.lift()  # 提到顶层
    #     self.image_viewer.focus_force()  # 获取焦点以响应键盘事件
    #
    #     # 立即更新一次图像，以显示当前滚动条位置的内容
    #     if self.topic_status.get(self.current_topic) == "已完成":
    #         self.on_slider_changed(self.slider_var.get())

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=(15, 15));
        self.main_frame = main_frame
        top_frame = ttk.Frame(main_frame);
        self.top_frame = top_frame

        topic_selection_frame = ttk.Frame(top_frame)
        self.topic_selection_frame = topic_selection_frame  # 提前赋值
        self.topic_label = ttk.Label(topic_selection_frame, text="选择话题:", font=self.bold_font)

        # 创建时 self.topics 可能为空，没关系，后面会用 config 更新
        self.topic_combo = ttk.Combobox(topic_selection_frame, values=self.topics, state="readonly",
                                        font=("Noto Sans CJK SC", 11))
        if self.current_topic: self.topic_combo.set(self.current_topic)
        self.topic_combo.bind("<<ComboboxSelected>>", self.on_topic_selected)


        # 改为动态生成插件按钮：
        for plugin in self.plugins:
            btn = ttk.Button(
                self.topic_selection_frame, 
                text=plugin.get_name(), 
                command=plugin.on_start, 
                bootstyle=plugin.get_button_style()
            )
            btn.pack(side="left", padx=(10, 0))

        self.status_label = ttk.Label(top_frame, text="状态: 初始化...", anchor="w")
        # ... (其余创建代码不变)
        self.progress_frame = ttk.Frame(top_frame)
        self.progress_label = ttk.Label(self.progress_frame, text="总进度:")
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate")
        slider_frame = ttk.Frame(top_frame)
        self.slider_label = ttk.Label(slider_frame, text="消息:")
        self.slider_var = tk.IntVar()
        self.slider = ttk.Scale(slider_frame, from_=1, to=1, orient="horizontal", variable=self.slider_var,
                                command=self.on_slider_changed)
        self.slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.prev_btn = ttk.Button(slider_frame, text="<<", command=self._prev_frame, width=3, bootstyle="outline")
        self.next_btn = ttk.Button(slider_frame, text=">>", command=self._next_frame, width=3, bootstyle="outline")
        self.count_label = ttk.Label(slider_frame, text="N/A", width=15)
        self.slider_frame = slider_frame
        self.slider.config(state="disabled")
        paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL);
        self.paned_window = paned_window
        value_pane = ttk.Frame(paned_window, padding=(0, 10, 0, 0))
        self.value_label = ttk.Label(value_pane, text="消息内容", font=self.bold_font)
        self.value_text = ScrolledText(value_pane, wrap="word", height=10, font=("Noto Sans CJK SC Mono", 10), autohide=True)
        self._update_text_widget(self.value_text, "等待后台索引完成...")
        paned_window.add(value_pane, weight=1)
        schema_pane = ttk.Frame(paned_window, padding=(0, 10, 0, 0))
        self.schema_label = ttk.Label(schema_pane, text="消息结构", font=self.bold_font)
        self.schema_text = ScrolledText(schema_pane, wrap="word", height=10, font=("Noto Sans CJK SC Mono", 10), autohide=True)
        paned_window.add(schema_pane, weight=1)

    def _on_slider_press(self, event=None):
        """当鼠标在滚动条上按下时调用"""
        self.is_slider_dragging = True

    def _on_slider_release(self, event=None):
        """当鼠标在滚动条上释放时调用"""
        self.is_slider_dragging = False
        # 释放后，立即以高质量模式更新一次最终位置
        self.on_slider_changed(self.slider_var.get())

    def _next_frame(self):
        current = int(self.slider_var.get())
        max_val = int(self.slider.cget("to"))
        if current < max_val:
            self.slider_var.set(current + 1)
            self.on_slider_changed(current + 1)

    def _prev_frame(self):
        current = int(self.slider_var.get())
        if current > 1:
            self.slider_var.set(current - 1)
            self.on_slider_changed(current - 1)

    def _configure_layout(self):
        # ... (与之前版本相同)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid(row=0, column=0, sticky="ew")
        self.topic_selection_frame.pack(fill="x", expand=True, pady=5)
        self.topic_label.pack(side="left", padx=(0, 10))
        self.topic_combo.pack(side="left", fill="x", expand=True)
        self.status_label.pack(fill="x", expand=True, pady=(5, 0))
        self.progress_frame.pack(fill="x", expand=True, pady=(5, 0))
        self.progress_label.pack(side="left", padx=(0, 10))
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.slider_frame.pack(fill="x", expand=True, pady=(5, 0))
        self.slider_label.pack(side="left", padx=(0, 10))
        self.prev_btn.pack(side="left", padx=(0, 5))
        self.slider.pack(side="left", fill="x", expand=True)
        self.next_btn.pack(side="left", padx=(5, 10))
        self.count_label.pack(side="left", padx=(10, 0))
        self.paned_window.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.value_label.pack(side="top", anchor="w")
        self.value_text.pack(side="top", fill="both", expand=True, pady=(5, 0))
        self.schema_label.pack(side="top", anchor="w")
        self.schema_text.pack(side="top", fill="both", expand=True, pady=(5, 0))

    def _get_cache_paths(self, topic_name):
        """根据bag文件和topic名生成确定性的、人类可读的缓存文件路径"""
        # 1. 获取不含扩展名的原始文件名, 例如 "avia_outdoor.bag" -> "avia_outdoor"
        base_filename = os.path.basename(self.bag_file_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        # 2. 清理文件名，替换掉所有不安全字符
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', filename_without_ext)
        # 3. 保留基于完整路径的哈希值，以确保唯一性 (防止不同路径下的同名文件冲突)
        bag_hash = hashlib.md5(self.bag_file_path.encode()).hexdigest()[:16]
        # 4. 清理topic名
        safe_topic_name = re.sub(r'[^a-zA-Z0-9_-]', '_', topic_name)
        # 5. 组合成新的、更具可读性的基础名
        base_name = f"{safe_filename}_{bag_hash}_{safe_topic_name}"
        return os.path.join(self.CACHE_DIR, f"{base_name}.index"), \
            os.path.join(self.CACHE_DIR, f"{base_name}.cache")

    def _ui_poller(self):
        """
        【已修改】定期从队列中取出更新，并处理特殊的 SHUTDOWN_COMPLETE 指令。
        """
        updates = {}
        # 循环直到队列为空
        while not self.ui_update_queue.empty():
            try:
                topic_name, status = self.ui_update_queue.get_nowait()

                # 【核心修改】检查是否是关闭指令
                if topic_name == "SHUTDOWN_COMPLETE":
                    print("主线程收到关闭指令，正在销毁根窗口...")
                    # 销毁主应用程序的根窗口，这将终止 mainloop 并结束程序。
                    self.master.destroy()
                    return  # 立即返回，因为GUI即将不存在

                # 如果是常规更新，则保留最新的一个
                updates[topic_name] = status
            except queue.Empty:
                break  # 队列已处理完毕

        # 应用常规状态更新
        for topic_name, status in updates.items():
            self._update_topic_status(topic_name, status)

        # 只要程序没关闭，就继续安排下一次轮询
        if not self.is_closing.is_set():
            self.after(self.UI_POLL_INTERVAL, self._ui_poller)

    def _start_background_indexing(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        for topic in self.topics:
            # --- 【核心改进 2】 任务提交到索引专用池 ---
            future = self.indexing_executor.submit(self._check_and_process_topic, topic)
            future.add_done_callback(self._on_indexing_complete)
        self._update_ui_for_topic()

    def _check_and_process_topic(self, topic):
        index_path, _ = self._get_cache_paths(topic)
        if os.path.exists(index_path):
            try:
                # --- 【核心改进 2】 从二进制索引文件加载 ---
                with open(index_path, 'rb') as f:
                    index_data = f.read()

                # 计算有多少条目
                num_entries = len(index_data) // self.INDEX_ENTRY_SIZE
                # 逐个解包
                index = [struct.unpack(self.INDEX_FORMAT,
                                       index_data[i * self.INDEX_ENTRY_SIZE:(i + 1) * self.INDEX_ENTRY_SIZE]) for i in
                         range(num_entries)]
                return topic, index, None
            except Exception as e:
                print(f"加载二进制索引 {topic} 失败: {e}, 将重新建立。")
        return self._process_topic_thread(topic)

    def _process_topic_thread(self, topic_name):
        self.ui_update_queue.put((topic_name, "索引中..."))
        index_path, cache_path = self._get_cache_paths(topic_name)
        try:
            with self.bag_lock:
                total_count = self.bag.get_message_count(topic_filters=[topic_name])
                messages_gen = self.bag.read_messages(topics=[topic_name], raw=True)

            last_update_time = time.time()
            # --- 【核心改进 3】 同时打开两个文件进行流式写入 ---
            with open(cache_path, 'wb') as cache_f, open(index_path, 'wb') as index_f:
                with self.bag_lock:
                    for i, (_, msg_tuple, t) in enumerate(messages_gen):
                        if self.is_closing.is_set(): return topic_name, None, "任务被用户取消"

                        # 写入数据文件
                        offset = cache_f.tell()
                        data_to_pickle = (msg_tuple[0], msg_tuple[1], t)
                        pickle.dump(data_to_pickle, cache_f)
                        size = cache_f.tell() - offset

                        # 立刻将索引条目写入索引文件
                        index_f.write(struct.pack(self.INDEX_FORMAT, offset, size))

                        current_time = time.time()
                        if current_time - last_update_time > 0.1:
                            last_update_time = current_time
                            progress_percent = (i + 1) / total_count if total_count > 0 else 0
                            self.ui_update_queue.put((topic_name, f"索引中... {progress_percent:.1%}"))

            # 索引完成后，重新加载它以确保一致性
            return self._check_and_process_topic(topic_name)
        except Exception as e:
            return topic_name, None, str(e)

    def _on_indexing_complete(self, future):
        if self.is_closing.is_set(): return
        try:
            topic_name, index, error = future.result()
            if error:
                self.ui_update_queue.put((topic_name, f"失败: {error[:30]}"))
            else:
                self.topic_indices[topic_name] = index
                self.ui_update_queue.put((topic_name, "已完成"))
        except concurrent.futures.CancelledError:
            print("一个索引任务被取消。")

    def _update_topic_status(self, topic_name, status):
        self.topic_status[topic_name] = status
        # --- 【核心 Bug 修复】 ---
        # 检查字典中的's'，而不是外部的'status'变量
        done_count = sum(1 for s in self.topic_status.values() if s == "已完成" or s.startswith("失败"))
        total_topics = len(self.topics)
        self.progress_bar['value'] = (done_count / total_topics) * 100 if total_topics > 0 else 0
        if topic_name == self.current_topic: self._update_ui_for_topic()

    def on_topic_selected(self, event=None):
        # ... (与之前版本相同)
        self.current_topic = self.topic_combo.get()
        self._update_ui_for_topic()

    def _update_ui_for_topic(self):
        # ... (与之前版本相同)
        if not self.current_topic: return
        status = self.topic_status.get(self.current_topic, "未知")
        self.status_label.config(text=f"当前话题 '{self.current_topic}' 状态: {status}")
        if status == "已完成":
            self.slider_frame.pack(fill="x", expand=True, pady=(5, 0))
            self.slider_label.pack(side="left", padx=(0, 10));
            self.slider.pack(side="left", fill="x", expand=True);
            self.count_label.pack(side="left", padx=(10, 0))
            index = self.topic_indices.get(self.current_topic, [])
            message_count = len(index)
            if message_count > 0:
                self.slider.config(to=message_count, state="normal");
                self.slider_var.set(1);
                self.on_slider_changed(1)
            else:
                self.slider.config(state="disabled");
                self._update_text_widget(self.value_text, "此话题下没有消息。")
            self.update_schema_display()
        else:
            self.slider_frame.pack_forget();
            self._update_text_widget(self.value_text, f"正在等待后台索引完成... 状态: {status}");
            self.count_label.config(text="N/A")

    def on_slider_changed(self, value):
        index = int(float(value)) - 1
        index_data = self.topic_indices.get(self.current_topic)

        if not (index_data and 0 <= index < len(index_data)):
            return

        is_high_quality = not self.is_slider_dragging
        total = len(index_data)
        
        # --- 插件事件广播 ---
        intercepted = False
        for plugin in self.plugins:
            if plugin.on_frame_changed(index, is_high_quality):
                intercepted = True

        # 如果有插件接管了当前的高性能渲染（比如反投影插件正在工作），则主界面只做简单UI更新
        if intercepted:
            self.count_label.config(text=f"{index + 1} / {total}")
            self._update_text_widget(self.value_text, f"插件正在接管渲染...\n当前帧: {index + 1}\n模式: {'高质量' if is_high_quality else '预览'}")
            return  # 提前结束，跳过后续耗时的 pickle 读取和字符串格式化

        # --- 如果没有插件接管，则执行默认的完整格式化流程 ---
        try:
            _, cache_path = self._get_cache_paths(self.current_topic)
            offset, size = index_data[index]
            with open(cache_path, 'rb') as f:
                f.seek(offset)
                msg_type, raw_data, timestamp = pickle.loads(f.read(size))

            reconstructed_msg = self._deserialize_raw_message(msg_type, raw_data, self.current_topic)

            self.count_label.config(text=f"{index + 1} / {total}")
            self._update_text_widget(self.value_text, f"时间戳: {timestamp.to_sec():.4f}\n\n正在格式化消息...")

            future = self.ui_task_executor.submit(self._format_message_in_background, reconstructed_msg, index, timestamp)
            future.add_done_callback(self._on_formatting_complete)

        except Exception as e:
            self._update_text_widget(self.value_text, f"读取或重建缓存失败: {e}")

    def _format_message_in_background(self, msg, index, t):
        """
        【已修改】在后台线程执行耗时的 str(msg) 转换。
        增加了对大型数组（如点云）的智能摘要功能。
        """
        try:
            # --- 智能摘要逻辑 ---
            # 检查消息是否含有 'points' 字段，并且它是一个长列表
            if hasattr(msg, 'points') and isinstance(msg.points, list) and len(msg.points) > 50:
                # 创建一个摘要，而不是完整的字符串
                summary_lines = []
                # 添加消息头和其他非'points'字段
                for field in msg.__slots__:
                    if field != 'points':
                        field_value = getattr(msg, field)
                        summary_lines.append(f"{field}: {field_value}")
                # 添加 'points' 字段的摘要信息
                summary_lines.append(f"points: ")
                point_count = len(msg.points)
                # 只显示前50个点
                for i, point in enumerate(msg.points[:50]):
                    summary_lines.append(f"  - [{i}]: {str(point).replace(chr(10), ' ')}")  # 换行符替换为空格
                if point_count > 50:
                    summary_lines.append(f"  ... [以及另外 {point_count - 50} 个点]")
                msg_str = "\n".join(summary_lines)
            else:
                # 对于其他类型的消息，仍然使用标准转换
                msg_str = str(msg)

            return msg_str, index, t, None
        except Exception as e:
            return None, index, t, str(e)

    def _on_formatting_complete(self, future):
        """【新】后台格式化完成后的回调函数，在主线程中安全更新UI"""
        if self.is_closing.is_set(): return

        try:
            msg_str, index, t, error = future.result()

            # 检查滑块当前位置是否与完成的任务匹配，防止旧任务的结果覆盖新界面
            if self.slider_var.get() != index + 1:
                return

            if error:
                self._update_text_widget(self.value_text, f"格式化消息时出错: {error}")
            else:
                self.display_message(index, msg_str, t, is_preformatted=True)
        except Exception as e:
            print(f"格式化回调函数出错: {e}")

    def display_message(self, index, msg_or_str, t, is_preformatted=False):
        """【已修改】显示消息内容，可接收预格式化的字符串"""
        if is_preformatted:
            msg_str = msg_or_str
        else:  # 向后兼容，以防万一
            msg_str = str(msg_or_str)

        if len(msg_str) > self.MAX_DISPLAY_LEN:
            msg_str = msg_str[:self.MAX_DISPLAY_LEN] + f"\n\n[... 内容过长，已截断。完整长度: {len(msg_str)} 字符 ...]"

        timestamp = t.to_sec()
        total = len(self.topic_indices.get(self.current_topic, []))
        display_content = f"时间戳: {timestamp:.4f} (消息 {index + 1}/{total})\n\n{msg_str}"
        self._update_text_widget(self.value_text, display_content)
        self.count_label.config(text=f"{index + 1} / {total}")

    # 在 RosBagViewer 类中
    def update_schema_display(self):
        try:
            with self.bag_lock:
                # 确保你正在获取连接信息
                connections = self.bag._get_connections(topics=[self.current_topic])
                connection = next(connections)
                msg_def = connection.msg_def
                # 获取主消息类型，例如 "sensor_msgs/Image"
                primary_type = connection.datatype

                # 调用我们唯一的、功能完善的解析函数
            schema = build_type_schemas_from_definition(msg_def, primary_type)
            # 使用 json.dumps 美化输出
            schema_str = json.dumps(schema, indent=4)
            self._update_text_widget(self.schema_text, schema_str)
        except Exception as e:
            self._update_text_widget(self.schema_text, f"无法获取结构信息.\n\nError: {e}")

    def _get_msg_def_for_topic(self, topic_name, msg_type):
        key = (topic_name, msg_type)
        if key in self._msg_def_cache:
            return self._msg_def_cache[key]
        with self.bag_lock:
            connections = self.bag._get_connections(topics=[topic_name])
            for conn in connections:
                if conn.datatype == msg_type:
                    self._msg_def_cache[key] = conn.msg_def
                    return conn.msg_def
        raise ValueError(f"无法获取消息定义: topic={topic_name}, type={msg_type}")

    def _get_dynamic_msg_class(self, msg_type, topic_name):
        if msg_type in self._dynamic_class_cache:
            return self._dynamic_class_cache[msg_type]
        msg_def = self._get_msg_def_for_topic(topic_name, msg_type)
        generated = genpy.dynamic.generate_dynamic(msg_type, msg_def)
        msg_class = generated.get(msg_type)
        if not msg_class:
            raise ValueError(f"动态生成消息类失败: {msg_type}")
        self._dynamic_class_cache[msg_type] = msg_class
        return msg_class

    def _deserialize_raw_message(self, msg_type, raw_data, topic_name):
        msg_class = self._get_dynamic_msg_class(msg_type, topic_name)
        msg = msg_class()
        msg.deserialize(raw_data)
        return msg
    def _update_text_widget(self, widget, content):
        # ... (与之前版本相同)
        widget.text.config(state="normal");
        widget.delete(1.0, tk.END);
        widget.insert(tk.END, content);
        widget.text.config(state="disabled")

    def close(self):
        if self.is_closing.is_set(): return
        self.is_closing.set()
        print("正在关闭应用...")
        self.status_label.config(text="正在关闭，请等待后台任务结束...")
        self.topic_combo.config(state="disabled")
        self.slider.config(state="disabled")
        self.update_idletasks()

        shutdown_thread = threading.Thread(target=self._shutdown_worker)
        shutdown_thread.start()

    def _shutdown_worker(self):
        """
        【已修改】在后台线程中安全地关闭所有非GUI资源。
        完成后，通过队列通知主线程来销毁GUI。
        """
        print("正在关闭UI任务线程池...")
        if self.ui_task_executor:
            self.ui_task_executor.shutdown(wait=True)

        print("正在关闭索引任务线程池...")
        if self.indexing_executor:
            self.indexing_executor.shutdown(wait=True)

        print("所有后台任务已完成。")
        with self.bag_lock:
            if self.bag:
                self.bag.close()
                print("Bag 文件已关闭。")

        # 【核心修改】不再直接调用 after()。
        # 而是向主线程的轮询器发送一个明确的指令。
        self.ui_update_queue.put(("SHUTDOWN_COMPLETE", None))
import tkinter as tk
from tkinter import filedialog, messagebox, font
import ttkbootstrap as ttk
import json
import os

class Launcher(ttk.Toplevel): # <--- 继承 Toplevel
    """
    应用程序启动器窗口。
    """
    HISTORY_FILE = "history.json"
    MAX_HISTORY = 15

    def __init__(self, master): # <--- 接收 master
        super().__init__(master) # <--- 调用父类初始化
        self.title("ROS Bag 查看器 - 启动器")
        # ... (其余 __init__ 代码不变)
        self.minsize(500, 180)  # 2. (可选) 设置一个合理的最小宽度和高度
        self.history = self._load_history()
        self.selected_file_path = tk.StringVar()
        self._create_widgets()
        self._setup_right_click_menu()


    def _load_history(self):
        """从 JSON 文件加载历史记录"""
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_history(self):
        """将当前历史记录保存到 JSON 文件"""
        with open(self.HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _add_to_history(self, path):
        """添加一条新记录到历史顶部，并保持列表长度"""
        if path in self.history:
            self.history.remove(path)
        self.history.insert(0, path)
        self.history = self.history[:self.MAX_HISTORY]
        self._save_history()

    def _create_widgets(self):
        """创建UI组件"""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame, text="选择最近使用的文件，或浏览新文件：", font=("Noto Sans CJK SC", 12)).pack(anchor="w")

        # --- 历史记录下拉框 ---
        combo_frame = ttk.Frame(main_frame)
        combo_frame.pack(fill="x", expand=True, pady=10)

        self.history_combo = ttk.Combobox(combo_frame, textvariable=self.selected_file_path, values=self.history,
                                          font=("Noto Sans CJK SC", 10))
        self.history_combo.pack(side="left", fill="x", expand=True)
        self.history_combo.bind("<<ComboboxSelected>>", self._on_validate_selection)
        self.history_combo.bind("<KeyRelease>", self._on_validate_selection)  # 文本变化时也验证

        # --- 按钮 ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", expand=True, pady=10)

        browse_button = ttk.Button(button_frame, text="浏览...", command=self._browse_file, width=15)
        browse_button.pack(side="left", padx=(0, 10))

        self.open_button = ttk.Button(button_frame, text="打开", command=self._open_selected_file, state="disabled",
                                      bootstyle="primary")
        self.open_button.pack(side="right")

        if self.history:
            self.selected_file_path.set(self.history[0])
            self._on_validate_selection()

    def _setup_right_click_menu(self):
        """设置右键删除菜单"""
        self.right_click_menu = tk.Menu(self, tearoff=0)
        self.right_click_menu.add_command(label="从历史记录中删除", command=self._delete_selected_history)

        # 将右键事件绑定到 Combobox 的输入框部分
        self.history_combo.bind("<Button-3>", lambda e: self.right_click_menu.post(e.x_root, e.y_root))

    def _delete_selected_history(self):
        """删除当前选中的历史记录"""
        path_to_delete = self.selected_file_path.get()
        if path_to_delete in self.history:
            self.history.remove(path_to_delete)
            self._save_history()

            # 更新下拉框
            self.history_combo['values'] = self.history
            self.selected_file_path.set("")  # 清空选择
            self.open_button.config(state="disabled")
            messagebox.showinfo("成功", f"已从历史记录中删除:\n{path_to_delete}")
        else:
            messagebox.showwarning("提示", "此路径不在历史记录中。")

    def _browse_file(self):
        """打开文件选择对话框"""
        file_path = filedialog.askopenfilename(
            title="选择一个 ROS Bag 文件",
            filetypes=[("ROS Bag Files", "*.bag"), ("All Files", "*.*")]
        )
        if file_path:
            self.selected_file_path.set(file_path)
            self._on_validate_selection()

    def _on_validate_selection(self, event=None):
        """检查当前路径是否有效，并更新“打开”按钮的状态"""
        path = self.selected_file_path.get()
        if path and os.path.exists(path) and path.endswith('.bag'):
            self.open_button.config(state="normal")
        else:
            self.open_button.config(state="disabled")

    def _open_selected_file(self):
        path = self.selected_file_path.get()
        if not os.path.exists(path):
            messagebox.showerror("错误", f"文件不存在:\n{path}")
            return

        self._add_to_history(path)

        # 【核心修改】
        # 1. 销毁自己 (Launcher 窗口)
        self.destroy()

        # 2. 创建新的查看器窗口，将 master 传递过去
        #    注意：不再调用 mainloop()
        app = RosBagViewer(self.master, path)


# --- 新的主程序入口 ---
if __name__ == '__main__':
    try:
        # 1. 创建一个应用的主根窗口，并应用主题
        root = ttk.Window(themename="flatly")
        # 2. 立刻将它隐藏。它的唯一作用就是持有主事件循环(mainloop)和主题
        root.withdraw()
        # 3. 创建我们的 Launcher 窗口，它的父窗口是隐藏的 root
        launcher = Launcher(root)
        # 4. 启动整个应用程序唯一的、持久的事件循环
        root.mainloop()
    except Exception as e:
        messagebox.showerror("严重错误", f"程序启动失败:\n\n{e}")
