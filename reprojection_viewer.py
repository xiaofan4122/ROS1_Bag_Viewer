import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as ttk
from ttkbootstrap.scrolled import ScrolledText
import numpy as np
import cv2
from PIL import Image, ImageTk

# Matplotlib 用于在Tkinter中嵌入3D绘图
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as mtransforms
from matplotlib import cm, colors

from bag_cache_reader import BagCacheReader


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
        self.K = np.eye(3)
        self.dist_coeffs = np.zeros(5)
        self.T_cam_lidar = np.eye(4)

        # UI 组件
        self.image_panel = None
        self.reprojection_panel = None
        self.ax_3d = None
        self.canvas_3d = None

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
        right_pane.add(self.plot_frame, weight=1)

    def configure_data_sources(self, bag_path, image_topic, lidar_topic, K, T_cam_lidar, dist_coeffs):
        """配置数据源和标定参数"""
        try:
            self.image_reader = BagCacheReader(bag_path)
            self.image_reader.load_topic(image_topic)

            self.lidar_reader = BagCacheReader(bag_path)
            self.lidar_reader.load_topic(lidar_topic)

            self.K = K
            self.dist_coeffs = dist_coeffs
            self.T_cam_lidar = T_cam_lidar

            print("数据源配置成功！")
        except Exception as e:
            messagebox.showerror("配置失败", f"配置数据源时出错: {e}")

    def compressed_img_to_cv2(self, msg):
        """将 sensor_msgs/CompressedImage 转换为 cv2 图像"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv_image

    def update_view_by_index(self, index: int):
        """根据给定的消息索引，从已配置的数据源加载数据并刷新所有视图"""
        if not (self.image_reader and self.lidar_reader):
            print("警告: 数据源未配置。")
            return

        try:
            # 获取图像数据
            image_msg, _ = self.image_reader.get_message(index)
            if image_msg._type == 'sensor_msgs/CompressedImage':
                cv_image = self.compressed_img_to_cv2(image_msg)
            else:
                cv_image = self.image_reader.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            # 获取雷达数据
            lidar_msg, _ = self.lidar_reader.get_message(index)

            # 从ROS点云消息中提取Nx3的NumPy数组 (适用于 a Livox CustomMsg)
            points_lidar = np.array([[p.x, p.y, p.z] for p in lidar_msg.points])

            # 更新所有面板
            self._update_original_image(cv_image)
            self._update_reprojection(cv_image, points_lidar)
            self._update_3d_view(points_lidar)
        except IndexError:
            # 当一个话题的消息比另一个少时，会发生此情况
            error_msg = f"索引 {index} 越界，图像和点云话题的消息数量可能不匹配。"
            print(error_msg)
            self.reprojection_panel.set_image(None)
        except Exception as e:
            error_msg = f"更新视图时出错:\n{e}"
            self.reprojection_panel.set_image(None)
            print(error_msg)

    def _update_original_image(self, image):
        self.image_panel.set_image(image)

    def _update_reprojection(self, image, points_lidar):
        reprojection_img = image.copy()
        h, w, _ = image.shape

        R_cam_lidar = self.T_cam_lidar[:3, :3]
        t_cam_lidar = self.T_cam_lidar[:3, 3]
        rvec, _ = cv2.Rodrigues(R_cam_lidar)

        if len(points_lidar) > 0:
            projected_points, _ = cv2.projectPoints(points_lidar, rvec, t_cam_lidar, self.K, self.dist_coeffs)
            points_camera = (R_cam_lidar @ points_lidar.T).T + t_cam_lidar

            max_depth, min_depth = 50.0, 0.5
            norm = colors.Normalize(vmin=min_depth, vmax=max_depth)
            cmap = cm.get_cmap('viridis')

            for i in range(len(projected_points)):
                u, v = projected_points[i][0]
                depth = points_camera[i, 2]
                if depth > min_depth and depth < max_depth and 0 <= u < w and 0 <= v < h:
                    color_rgba = cmap(norm(depth))
                    color_bgr = [int(c * 255) for c in color_rgba[:3]][::-1]
                    cv2.circle(reprojection_img, (int(u), int(v)), 3, color_bgr, -1)

        self.reprojection_panel.set_image(reprojection_img)

    def _update_3d_view(self, points_lidar):
        self.ax_3d.clear()

        # 绘制点云
        if points_lidar is not None and len(points_lidar) > 0:
            self.ax_3d.scatter(points_lidar[:, 0], points_lidar[:, 1], points_lidar[:, 2], s=1, c=points_lidar[:, 2],
                               cmap='viridis', alpha=0.5)

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
        if points_lidar is not None and len(points_lidar) > 0:
            all_points_for_limits = np.vstack([all_points_for_limits, points_lidar])

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

        self.canvas_3d.draw()

    def _on_3d_scroll(self, event):
        scale_factor = 0.9 if event.button == 'up' else 1.1
        self.ax_3d.dist *= scale_factor
        self.canvas_3d.draw_idle()


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