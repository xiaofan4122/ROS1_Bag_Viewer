import os
import struct
import pickle
import hashlib
import re
import roslib.message
from cv_bridge import CvBridge  # 仍然需要它来处理图像消息


class CacheNotFoundError(Exception):
    """当找不到指定话题的缓存文件时抛出此异常。"""
    pass


class BagCacheReader:
    """
    一个用于读取由 RosBagViewer 生成的持久化缓存的独立数据访问类。

    这个类不包含任何GUI代码，可以被任何需要访问bag数据的脚本独立使用。
    它通过读取预先生成的.index和.cache文件来提供对消息的快速、随机访问。
    """
    CACHE_DIR = ".rosbag_cache"
    INDEX_FORMAT = '>QQ'
    INDEX_ENTRY_SIZE = struct.calcsize(INDEX_FORMAT)

    def __init__(self, bag_file_path: str):
        """
        初始化读取器。

        :param bag_file_path: 原始 .bag 文件的路径。
        """
        if not os.path.exists(bag_file_path):
            raise FileNotFoundError(f"指定的 bag 文件不存在: {bag_file_path}")

        self.bag_file_path = os.path.abspath(bag_file_path)
        self.bridge = CvBridge()  # 用于图像消息转换

        self.current_topic = None
        self.current_topic_index = []
        self.current_cache_path = None

    def _get_cache_paths(self, topic_name: str) -> (str, str):
        """根据bag文件和topic名生成确定性的缓存文件路径"""
        bag_hash = hashlib.md5(self.bag_file_path.encode()).hexdigest()[:16]
        safe_topic_name = re.sub(r'[^a-zA-Z0-9_-]', '_', topic_name)
        base_filename = os.path.splitext(os.path.basename(self.bag_file_path))[0]
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', base_filename)
        base_name = f"{safe_filename}_{bag_hash}_{safe_topic_name}"
        return os.path.join(self.CACHE_DIR, f"{base_name}.index"), \
            os.path.join(self.CACHE_DIR, f"{base_name}.cache")

    def load_topic(self, topic_name: str) -> bool:
        """
        加载指定话题的索引到内存中，准备进行读取。

        :param topic_name: 需要读取的话题名称，例如 '/livox/lidar'
        :return: 如果加载成功返回 True
        :raises CacheNotFoundError: 如果找不到该话题的索引文件
        """
        if self.current_topic == topic_name:
            return True

        index_path, cache_path = self._get_cache_paths(topic_name)

        if not os.path.exists(index_path):
            print(f"index_path: {index_path}")
            raise CacheNotFoundError(f"找不到话题 '{topic_name}' 的索引文件。请先使用 RosBagViewer GUI 程序生成缓存。")

        with open(index_path, 'rb') as f:
            index_data = f.read()

        num_entries = len(index_data) // self.INDEX_ENTRY_SIZE
        self.current_topic_index = [
            struct.unpack(self.INDEX_FORMAT, index_data[i * self.INDEX_ENTRY_SIZE:(i + 1) * self.INDEX_ENTRY_SIZE])
            for i in range(num_entries)
        ]

        self.current_topic = topic_name
        self.current_cache_path = cache_path
        print(f"话题 '{topic_name}' 加载成功，共 {len(self.current_topic_index)} 条消息。")
        return True

    def get_message_count(self) -> int:
        """
        获取当前已加载话题的消息总数。

        :return: 消息数量
        """
        return len(self.current_topic_index)

    def get_message(self, index: int):
        """
        获取指定索引位置的ROS消息对象。

        :param index: 消息的索引 (从0开始)
        :return: 重建后的ROS消息对象
        :raises IndexError: 如果索引越界
        :raises RuntimeError: 如果没有先调用 load_topic
        """
        if self.current_topic is None:
            raise RuntimeError("请在使用 get_message 之前先调用 load_topic()。")

        if not 0 <= index < len(self.current_topic_index):
            raise IndexError(f"索引 {index} 越界，有效范围是 0 到 {len(self.current_topic_index) - 1}。")

        try:
            offset, size = self.current_topic_index[index]
            with open(self.current_cache_path, 'rb') as f:
                f.seek(offset)
                msg_type, raw_data, timestamp = pickle.loads(f.read(size))

            msg_class = roslib.message.get_message_class(msg_type)
            if not msg_class:
                raise ValueError(f"无法找到消息类: {msg_type}")

            reconstructed_msg = msg_class()
            reconstructed_msg.deserialize(raw_data)

            # 返回完整的消息对象和它的时间戳
            return reconstructed_msg, timestamp

        except Exception as e:
            # 包装底层错误，提供更多上下文
            raise IOError(f"从缓存读取或重建消息 {index} 时失败: {e}")

    def get_image(self, index: int, encoding="bgr8"):
        """
        一个专门为图像话题设计的便捷接口。

        :param index: 图像消息的索引
        :param encoding: 期望的OpenCV图像编码 (如 "bgr8", "rgb8", "mono8")
        :return: (OpenCV格式的图像, 时间戳)
        :raises ValueError: 如果当前话题不是可识别的图像类型
        """
        msg, timestamp = self.get_message(index)
        msg_type = msg._type

        if msg_type == "sensor_msgs/Image":
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding), timestamp
        elif msg_type == "sensor_msgs/CompressedImage":
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=encoding), timestamp
        else:
            raise ValueError(f"话题 '{self.current_topic}' 不是一个有效的图像类型 (而是 {msg_type})")


# --- 使用示例 ---
if __name__ == '__main__':
    # 假设这是您之前用GUI程序处理过的bag文件
    # 重要：请确保运行此示例的终端已经 source 了您的ROS工作空间
    BAG_FILE = '/media/ss/Fan/sync1.bag'  # <--- 修改为您自己的bag文件路径
    LIDAR_TOPIC = '/livox/lidar'  # <--- 修改为您想读取的话题
    IMAGE_TOPIC = '/left_camera/image/compressed'  # <--- 修改为您的图像话题

    if not os.path.exists(BAG_FILE):
        print(f"错误: Bag 文件 '{BAG_FILE}' 不存在。请修改路径并确保您已使用GUI程序生成了缓存。")
    else:
        try:
            # 1. 创建读取器实例
            reader = BagCacheReader(BAG_FILE)

            # --- 示例1: 读取激光雷达数据 ---
            print("\n--- 读取Lidar数据 ---")
            reader.load_topic(LIDAR_TOPIC)

            message_count = reader.get_message_count()
            print(f"消息总数: {message_count}")

            if message_count > 0:
                # 获取第一条消息
                print("\n获取第一条消息 (索引 0):")
                first_msg, first_ts = reader.get_message(0)
                print(f"时间戳: {first_ts.to_sec()}")
                print(f"点云数量: {first_msg.point_num}")

                # 获取最后一条消息
                print("\n获取最后一条消息 (索引 {}):".format(message_count - 1))
                last_msg, last_ts = reader.get_message(message_count - 1)
                print(f"时间戳: {last_ts.to_sec()}")
                print(f"点云数量: {last_msg.point_num}")

            # --- 示例2: 读取并显示图像数据 (需要OpenCV) ---
            print("\n--- 读取图像数据 ---")
            try:
                import cv2

                reader.load_topic(IMAGE_TOPIC)
                img_count = reader.get_message_count()
                if img_count > 0:
                    print(f"读取第10张图片 (索引 9)...")
                    # 使用便捷接口 get_image()
                    cv_image, img_ts = reader.get_image(9)

                    print(f"图像尺寸: {cv_image.shape}")

                    # 显示图像
                    cv2.imshow("Image Viewer", cv_image)
                    print("按任意键关闭图像窗口...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except ImportError:
                print("\n警告: 未安装OpenCV (pip install opencv-python)，无法演示图像读取。")
            except CacheNotFoundError:
                print(f"\n警告: 找不到话题 '{IMAGE_TOPIC}' 的缓存，跳过图像演示。")
            except IndexError:
                print(f"\n警告: 话题 '{IMAGE_TOPIC}' 的消息数量不足10条，无法读取索引9。")


        except (FileNotFoundError, CacheNotFoundError, RuntimeError, IndexError, IOError) as e:
            print(f"\n程序出错: {e}")