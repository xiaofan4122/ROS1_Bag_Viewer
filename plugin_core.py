# plugin_core.py
import abc

class ViewerContext:
    """提供给插件的上下文，安全地暴露主程序的状态和数据"""
    def __init__(self, viewer_instance):
        self._viewer = viewer_instance

    @property
    def master(self):
        return self._viewer.master

    @property
    def bag_file_path(self):
        return self._viewer.bag_file_path

    @property
    def topics(self):
        return self._viewer.topics

    @property
    def topic_info(self):
        return self._viewer.topic_info

    def get_current_topic(self):
        return self._viewer.current_topic

    def get_current_index(self):
        # 返回从 0 开始的真实索引
        return self._viewer.slider_var.get() - 1


class RosBagPluginBase(abc.ABC):
    """所有插件必须继承的基类"""
    def __init__(self, context: ViewerContext):
        self.context = context

    @abc.abstractmethod
    def get_name(self) -> str:
        """插件在 UI 上显示的名称"""
        pass

    def get_button_style(self) -> str:
        """按钮样式 (默认 primary，可选 success, info, warning 等)"""
        return "primary"

    @abc.abstractmethod
    def on_start(self):
        """点击插件按钮时执行的操作"""
        pass

    def on_frame_changed(self, index: int, is_high_quality: bool) -> bool:
        """
        当主界面滑动条改变时触发。
        返回值: 如果返回 True，表示该插件接管了当前帧的渲染，
        主程序将跳过耗时的默认文本解析和格式化（用于提升拖动时的性能）。
        """
        return False