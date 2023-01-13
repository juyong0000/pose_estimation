import os
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

class Camera:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.context  = rs.context()
        self.pipeline = rs.pipeline()
        self.config   = rs.config()
        self.align    = rs.align(rs.stream.color)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(self.device_product_line + " is ready")
        self.device_name = device.get_info(rs.camera_info.name).replace(" ", "_")
        self.device_name = self.device_name + "_" + device.get_info(rs.camera_info.serial_number)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile     = self.pipeline.start(self.config)
        depth_sensor     = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.color_intrinsic = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsic = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx = self.color_intrinsic.fx
        self.fy = self.color_intrinsic.fy
        self.ppx = self.color_intrinsic.ppx
        self.ppy = self.color_intrinsic.ppy

        self.camera_mat = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float)
        self.dist_coeffs = np.zeros(4)
        self.colorizer = rs.colorizer(color_scheme = 2)

    def stop(self):
        self.pipeline.stop(self.config)

    def stream(self):
        frames           = self.pipeline.wait_for_frames()
        frames           = self.align.process(frames)
        depth_frame      = frames.get_depth_frame()
        color_frame      = frames.get_color_frame()
        self.color_image = np.asanyarray(color_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())
        return self.depth_image

    def generate(self, depth):
        self.pcd         = o3d.geometry.PointCloud()
        w                = np.shape(depth)[1]
        h                = np.shape(depth)[0]
        z                = depth * self.depth_scale  # raw distance
        u                = np.arange(0, w)
        v                = np.arange(0, h)
        mesh_u, mesh_v   = np.meshgrid(u, v)
        mesh_x           = (mesh_u - self.ppx) * z / self.fx
        mesh_y           = (mesh_v - self.ppy) * z / self.fy
        ## remove zeros and NaN values from x, y, z  this is valid regardless of if depth image is filtered or not
        if np.any(z == 0) or np.isnan(z).any():
            z = z[np.nonzero(z)]
            z = z[~ np.isnan(z)]
            mesh_x = mesh_x[np.nonzero(mesh_x)]
            mesh_x = mesh_x[~ np.isnan(mesh_x)]
            mesh_y = mesh_y[np.nonzero(mesh_y)]
            mesh_y = mesh_y[~ np.isnan(mesh_y)]
        ## raw point cloud in numpy format
        self.xyz         = np.zeros((np.size(mesh_x), 3))
        self.xyz[:, 0]   = np.reshape(mesh_x, -1)
        self.xyz[:, 1]   = np.reshape(mesh_y, -1)
        self.xyz[:, 2]   = np.reshape(z,      -1)
        ## raw point cloud in o3d format
        self.pcd.points  = o3d.utility.Vector3dVector(self.xyz)
        return self.pcd
