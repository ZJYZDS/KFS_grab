#! /home/zjy/anaconda3/envs/yolo3D_py38/bin/python
# coding=utf-8
import rospy
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import cv2

class kfs_pc:
    def __init__(self):
        self.kfs_pc_pub = rospy.Publisher("/kfs_pointcloud",PointCloud2,queue_size=10)
        self.kfs = rs.pointcloud()
        self.kfs_msg = PointCloud2()

    def kfs_pointcloud_pub(
            self,
            depth_frame,
            color_frame,
            MASK,
    ):
        """
        接收单帧的 color image 与 depth image ,并将其转化为点云(XYZ)
        """
        # color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.kfs.map_to(color_frame)
        points = self.kfs.calculate(depth_frame)

        # 转为N×3的点云数组（X/Y/Z，float32），1D展平索引
        kfs_pc = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        # 获取相机帧的分辨率（与YOLO掩码的H/W一致）
        h = depth_frame.get_height()
        w = depth_frame.get_width()

        # 2. 校验YOLO掩码的有效性和维度匹配性
        if MASK is None or MASK.ndim != 2 or MASK.shape != (h, w):
            rospy.logwarn("YOLO腐蚀掩码无效/维度与相机帧不匹配（需{}×{}）".format(h, w))
            return False
        # 将2D YOLO掩码展平为1D，与点云1D索引对齐（有效区域为True）
        mask_2d_valid = (MASK > 0).astype(bool)
        mask_1d_yolo = mask_2d_valid.reshape(-1)  # 展平为1D

        # 3. 分步筛选点云：先深度过滤，再YOLO掩码过滤（保证索引对齐）
        # 深度过滤：保留0.35~3.0米的点，1D掩码
        mask_1d_depth = np.logical_and(kfs_pc[:, 2] > 0.35, kfs_pc[:, 2] < 3.0)
        # 合并掩码：同时满足「深度有效」+「YOLO目标区域」
        mask_1d_combined = np.logical_and(mask_1d_depth, mask_1d_yolo)
        # 筛选点云：仅保留目标区域的有效点云，保证数组形状始终为N×3
        kfs_pc = kfs_pc[mask_1d_combined]
        # 校验筛选后的点云有效性
        if len(kfs_pc) == 0:
            rospy.logwarn("YOLO掩码区域内无有效深度点云")
            return False

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y',4,PointField.FLOAT32,1),
            PointField('z',8,PointField.FLOAT32,1),
        ]

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        self.kfs_msg = pc2.create_cloud(header=header, fields=fields, points=kfs_pc)
        self.kfs_pc_pub.publish(self.kfs_msg)
        return True

  # def get_mask_center_3d(self, mask_list, depth_img , color_img):
    #     global Depth_frame , Color_frame
    #     if not mask_list or depth_img is None or depth_img.size == 0:
    #         rospy.logwarn("mask_list或depth_img为空")
    #         return None
        
    #     min_dx = np.inf
    #     center_mask = {}

    #     for i, mask_info in enumerate(mask_list):
    #         mask = mask_info["mask"]
    #         mask_area = cv2.countNonZero(mask) if mask.dtype == np.uint8 else np.count_nonzero(mask)
    #         if mask_area < 50:
    #             rospy.loginfo("mask area too low")
    #             continue
            
    #         ys, xs = np.where(mask > 0)
    #         cx = int(np.mean(xs))
    #         cy = int(np.mean(ys))
            
    #         if abs(cx - 320) < min_dx:
    #             min_dx = abs(cx - 320)
    #             center_mask["xys"] = (ys, xs)
    #             center_mask["cxy"] = (cx, cy)
    #             center_mask["mask"] = mask

    #     if not center_mask:
    #         rospy.logwarn("no valid center mask found")
    #         return None
        
    #     (cx, cy) = center_mask["cxy"]
    #     (ys, xs) = center_mask["xys"]
    #     mask = center_mask["mask"]
        
    #     if isinstance(mask, torch.Tensor):
    #         mask = mask.squeeze().cpu().numpy()
        
    #     mask = (mask > 0.5).astype(np.uint8) * 255
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
    #     # 过滤连通域
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #     if num_labels < 2:
    #         rospy.logwarn("无有效连通域")
    #         return None
        
    #     max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #     filtered_mask = (labels == max_label).astype(np.uint8)
    #     ys_filtered, xs_filtered = np.where(filtered_mask > 0)
        
    #     if len(xs_filtered) < 2000:
    #         rospy.logwarn("valid points num is too low")
    #         return None
        
    #     # 深度值过滤
    #     depth_mm = depth_img[ys_filtered, xs_filtered]
    #     Depth_frame = depth_mm
    #     Color_frame = color_img[ys_filtered,xs_filtered]
    #     valid_depth_mask = (depth_mm > 0) & (depth_mm < 2400) & (np.abs(depth_mm - depth_img[cy, cx]) < 700)
        
    #     if not np.any(valid_depth_mask):
    #         rospy.logwarn("no valid depth in mask area")
    #         return None
        
    #     depth_mm = depth_mm[valid_depth_mask]
    #     valid_X = xs_filtered[valid_depth_mask]
    #     valid_Y = ys_filtered[valid_depth_mask]
        
    #     # 像素转3D坐标
    #     fx, fy, cx, cy = CAMERA_INTRINSICS["fx"], CAMERA_INTRINSICS["fy"], \
    #                      CAMERA_INTRINSICS["cx"], CAMERA_INTRINSICS["cy"]
        
    #     x_3d = (valid_X - cx) * depth_mm / fx
    #     y_3d = (valid_Y - cy) * depth_mm / fy
    #     z_3d = depth_mm

    #     c_x = np.mean(x_3d)
    #     c_y = np.mean(y_3d)
    #     c_z = np.mean(z_3d)

    #     rospy.loginfo(f"滤波后有效像素数: {len(xs_filtered)}, 3D重心: ({c_x:.3f}, {c_y:.3f}, {c_z:.3f})")
    #     return (c_x, c_y, c_z)