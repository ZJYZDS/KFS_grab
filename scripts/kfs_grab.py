#! /home/zjy/anaconda3/envs/yolo3D_py38/bin/python
# coding=utf-8

import rospy

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import torch
import struct
import serial  
from std_msgs.msg import Float32MultiArray, MultiArrayDimension 
from serial2tf import *
from utils.kfs_pointcloud import *

FRANE_START = b'\x0B'
FRAME_END = b'\x1A'


# 全局参数配置
CAMERA_INTRINSICS = {
    "fx": 321.6,
    "fy": 321.6,
    "cx": 320.7,
    "cy": 174.2,
}
DEPTH_SCALE = 0.001



class MaskPointCloudPublisher:
    def __init__(self):
        self.z_buffer = []
        #ros init
        rospy.init_node("center_point_coord_node")
        self.pointcloud_part = kfs_pc()
        self.pub_coord = rospy.Publisher("/center_coord",Float32MultiArray,queue_size=5)

        # YOLO model init
        self.model_config={
            "model_path": r"/home/zjy/roboncon2025_ws/src/yolo_depth_pkg/scripts/best.pt"
        }
        self.predict_config = {
            "img_sz": 640,
            "conf": 0.5,
            "iou": 0.5
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.model_config["model_path"]).to(self.device)
        rospy.loginfo("YOLO Model Loaded Successfully")

        # RealSense 相机初始化
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self._init_camera()
        
        # 串口初始化
        self.ser = None
        try:
            self.ser = serial.Serial(
                port="/dev/ttyUSB0",
                baudrate=115200,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
        except Exception as e:
            rospy.logwarn(f"串口打开失败: {e}")

        # 内参更新
        self._update_intrinsics()



    def serial_pub(self, coord):
        if not self.ser:
            rospy.logwarn("串口未初始化，跳过发送")
            return
        if coord is None or any(v is None for v in coord):
            rospy.logwarn("coord存在空值，跳过串口发送")
            return
        
        x, y, z = coord  # mm级
        try:
            data = [x,y,z]
            rospy.loginfo("data_xyz:%.2f , %.2f , %.2f", x, y, z)
            data = struct.pack('<3f', *data)
            self.ser.write(FRAME_START + data + FRAME_END)
            rospy.loginfo("ser send success")
        except Exception as e:
            rospy.logerr(f"串口发送失败: {e}")

    def _init_camera(self):
        """初始化 RealSense 相机流"""
        try:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)
            
            # 获取深度缩放系数
            depth_sensor = profile.get_device().first_depth_sensor()
            global DEPTH_SCALE
            DEPTH_SCALE = depth_sensor.get_depth_scale()
            rospy.loginfo(f"Depth scale: {DEPTH_SCALE}")
        except Exception as e:
            rospy.logerr(f"相机初始化失败: {e}")
            raise

    def _update_intrinsics(self):
        """从相机获取实际内参"""
        profiles = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profiles.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        
        CAMERA_INTRINSICS["fx"] = intrinsics.fx
        CAMERA_INTRINSICS["fy"] = intrinsics.fy
        CAMERA_INTRINSICS["cx"] = intrinsics.ppx
        CAMERA_INTRINSICS["cy"] = intrinsics.ppy
        rospy.loginfo(f"相机内参更新: {CAMERA_INTRINSICS}")

    def TF(self, coord, alpha_deg):
        """
        修正后的坐标旋转变换函数
        :param coord: (x, y, z) 原始坐标
        :param alpha_deg: 旋转角度（度）
        :return: 变换后的坐标
        """
        if coord is None or any(v is None for v in coord):
            rospy.logwarn("coord存在空值，TF变换失败")
            return None
        
        # 1. 转换角度为弧度
        alpha = np.deg2rad(alpha_deg)
        x, y, z = coord
        
        # 2. 保存原始x/y（核心修正）
        x_original = x
        y_original = y
        
        # 3. 正确的旋转变换
        x_new = x_original * np.cos(alpha) - y_original * np.sin(alpha)
        y_new = x_original * np.sin(alpha) + y_original * np.cos(alpha)
        
        # 4. 平移补偿（根据实际机械结构调整，这里保留原逻辑）
        z_new = z + 155  # 仅保留必要的平移，其他平移建议根据实际标定调整
        y_new -= 20
        # 5. 返回新坐标
        return (x_new, y_new, z_new)

  
    def get_reliable_top_seed(self, mask_list, depth_img, color_img):
        """
        从 mask 中提取顶面大概中心点（相机坐标系）
        返回: (x, y, z) 或 None
        """
        # 1. 选择距离图像中心最近的有效 mask（与原来逻辑一致）
        min_dx = np.inf
        best_mask = None
        for mask_info in mask_list:
            mask = mask_info["mask"]
            if cv2.countNonZero(mask) < 50:
                continue
            ys, xs = np.where(mask > 0)
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            if abs(cx - 320) < min_dx:
                min_dx = abs(cx - 320)
                best_mask = mask
        
        if best_mask is None:
            return None
        
        # 2. 形态学腐蚀，去除边缘（迭代次数根据物体大小调整）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        eroded_mask = cv2.erode(best_mask, kernel, iterations=2)
        if cv2.countNonZero(eroded_mask) < 400:
            rospy.logwarn("eroded defect")
            # 腐蚀后太小，回退到原mask
            eroded_mask = best_mask
        self.MASK = eroded_mask
        
        # 3. 获取内部区域的有效深度点
        # 在 mask 内取上 1/3 区域
        ys, xs = np.where(mask > 0)
        y_min, y_max = ys.min(), ys.max()
        y_top = y_min + (y_max - y_min) / 3   # 上三分之一
        top_region = (ys <= y_top)
        xs_top = xs[top_region]
        ys_top = ys[top_region]
        if len(xs_top) < 500:
            # 回退到整个 mask
            xs_top, ys_top = xs, ys

        depths = depth_img[ys_top, xs_top]
        valid = (depths > 0) & (depths < 2000)
        depths_valid = depths[valid]
        if len(depths_valid) < 200:
            return None

        # 取深度中值附近 ±10% 范围内的点
        median_depth = np.median(depths_valid)
        range_depth = np.ptp(depths_valid)   # 极差
        low = median_depth - 0.1 * range_depth
        high = median_depth + 0.1 * range_depth
        mid_band = (depths_valid >= low) & (depths_valid <= high)
        if np.sum(mid_band) < 100:
            # 如果带内点太少，直接使用全部有效点
            mid_band = np.ones_like(depths_valid, dtype=bool)

        xs_mid = xs_top[valid][mid_band]
        ys_mid = ys_top[valid][mid_band]
        

        """
        """
        
        depths_mid = depths_valid[mid_band]
        # 5. 像素转 3D 坐标（相机坐标系）
        fx, fy, cx, cy = CAMERA_INTRINSICS["fx"], CAMERA_INTRINSICS["fy"], \
                        CAMERA_INTRINSICS["cx"], CAMERA_INTRINSICS["cy"]

        # 转 3D 并求重心
        x_3d = (xs_mid - cx) * depths_mid / fx
        y_3d = (ys_mid - cy) * depths_mid / fy
        z_3d = depths_mid
        seed = (np.mean(x_3d), np.mean(y_3d), np.mean(z_3d))
        return seed
        
     
        
    def dispose_result(self, results):
        mask_list = []
        for result in results:
            if result is None or result.masks is None or len(result.masks.data) == 0:
                rospy.logwarn("No valid masks in current result")
                continue

            for i in range(len(result.masks.data)):
                try:
                    mask = result.masks.data[i].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    if i >= len(result.boxes.cls):
                        rospy.logwarn(f"Index {i} out of range for boxes.cls, skip")
                        continue
                    cls_id = int(result.boxes.cls[i])

                    mask_list.append({
                        'mask': mask,
                        'cls': cls_id
                    })
                except Exception as e:
                    rospy.logerr(f"Failed to process mask {i}: {str(e)}", exc_info=True)
                    continue
        return mask_list if mask_list else None

    def filter_z(self, coord):
        z = coord[2]
        if len(self.z_buffer) < 5:
            self.z_buffer.append(z)
            return z
        mean_z = np.mean(self.z_buffer)
        if np.abs(mean_z - z) > 300:
            rospy.logwarn(f"z值异常: 当前{z}, 均值{mean_z}，使用均值替代")
            return mean_z
        else:
            self.z_buffer.pop(0)
            self.z_buffer.append(z)
            return np.mean(self.z_buffer)

    def run(self):
        """主循环：获取帧→生成Mask→发布数据"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # 获取对齐后的帧
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 转换为numpy数组
            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            
            # YOLO推理
            results = self.model.predict(
               source=color_img,
               imgsz=self.predict_config["img_sz"],
               conf=self.predict_config["conf"],
               iou=self.predict_config["iou"],
               verbose=False,
            )
            
            # 处理mask
            mask_list = self.dispose_result(results)
            if mask_list is None:
                rospy.loginfo("mask_list is None , pass")
                continue
            
            # 获取3D坐标
            temp = self.get_reliable_top_seed(mask_list, depth_img, color_img)
            if temp is None:
                rospy.loginfo("center_3d coord is None,pass")
                continue
            """
            点云同步pub
            """

            if self.pointcloud_part.kfs_pointcloud_pub(depth_frame ,color_frame,self.MASK):
                rospy.loginfo("点云发布success")
            else:
                rospy.logwarn("点云发布失败")
            
            x, y, z = temp
            rospy.loginfo(f"原始3D坐标: {x:.3f}, {y:.3f}, {z:.3f}")
            msg_coord = Float32MultiArray()
            dim = MultiArrayDimension()
            dim.size = 3
            dim.stride = 3
            dim.label = "top_face_coord"
            msg_coord.layout.dim.append(dim)
            msg_coord.data = [x,y,z]
            self.pub_coord.publish(msg_coord)
            rospy.loginfo("msg_coord pub success")
            
            # 坐标轴映
            mapped_x, mapped_y, mapped_z = z, x, -y
            rospy.loginfo(f"轴映射后坐标: {mapped_x:.3f}, {mapped_y:.3f}, {mapped_z:.3f}")
            
            # # TF变换（传入角度为30度）
            # coord = self.TF((mapped_x, mapped_y, mapped_z), alpha_deg=30)
            # if coord is None:
            #     continue
            
            # # z值滤波
            # filtered_z = self.filter_z(coord)
            # coord = (coord[0], coord[1], filtered_z)  # 修正元组不可变问题
            # rospy.loginfo(f"TF变换+滤波后坐标: {coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}")
            
            # 串口发送（可选开启）
            # self.serial_pub(coord)
            
            rate.sleep()
#     # ------------------------------
# # numpy 图像 → 转成 RealSense 原生 frame
# # ------------------------------
# def numpy_to_color_frame(color_image):
#     # color_image: numpy array (H, W, 3), uint8
#     h, w = color_image.shape[:2]
#     frame = rs.video_frame()
#     profile = rs.video_stream_profile()

#     # 创建frame
#     vf = rs.rs2video_frame()
#     vf.width = w
#     vf.height = h
#     vf.bpp = 24
#     vf.fmt = rs.format.rgb8
#     vf.stride = w * 3

#     # 把numpy数据拷贝进frame
#     data = color_image.tobytes()
#     vf.data = data
#     frame = rs.video_frame(vf)
#     return frame

# def numpy_to_depth_frame(depth_image):
#     # depth_image: numpy array (H, W), uint16
#     h, w ,_= depth_image.shape
#     frame = rs.video_frame()
#     vf = rs.rs2video_frame()
#     vf.width = w
#     vf.height = h
#     vf.bpp = 16
#     vf.fmt = rs.format.z16
#     vf.stride = w * 2
#     data = depth_image.tobytes()
#     vf.data = data
#     frame = rs.video_frame(vf)
#     return frame


if __name__ == "__main__":
    node = None
    try:
        node = MaskPointCloudPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if node:
            node.pipeline.stop()
            if node.ser:
                node.ser.close()
        rospy.loginfo("程序退出，资源已释放")
