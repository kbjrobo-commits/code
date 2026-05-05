import os
import sys
import time
import math
import threading

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyrealsense2 as rs

import cv2

from .camera_interface import CameraInterface

L515_DEFAULT_DEPTH = (1024, 768)
L515_DEFAULT_COLOR = (1280, 720)

D455_DEFAULT_DEPTH = (1280, 720)
D455_DEFAULT_COLOR = (1280, 720)

class RealSense(CameraInterface):

    def __init__(self, serial=None):

        self.__WhiteText = "\033[37m"
        self.__BlackText = "\033[30m"
        self.__RedText = "\033[31m"
        self.__BlueText = "\033[34m"

        self.__DefaultText = "\033[0m"
        self.__BoldText = "\033[1m"

        np.set_printoptions(precision=3)

        self.connected = False

        # camera model info
        self.product_line = None
        self.product_id = serial

        # camera setting info
        self.resolution_color = None
        self.resolution_depth = None
        self.fps_color = 30
        self.fps_depth = 30

        # camera configuration
        self._camera_matrix = None
        self._depth_scale = None
        self._distCoeffs = None

        # realsense member variables
        self._align_frame = None

        self._pipeline = None
        self._config = None
        self._profile = None

    @property
    def camera_matrix(self):
        return self._camera_matrix.copy()

    @property
    def distCoeffs(self):
        return self._distCoeffs[:]

    @property
    def depth_scale(self):
        return self._depth_scale

    def initialize(self, resolution_color: tuple, resolution_depth: tuple):
        print(self.__BoldText + self.__BlueText + 'Start streaming...' + self.__BlackText + self.__DefaultText)

        self.resolution_color = resolution_color
        self.resolution_depth = resolution_depth

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self.product_id)

        # Get device product line
        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = self._config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.product_line = str(device.get_info(rs.camera_info.product_line))
        self.product_id = str(device.get_info(rs.camera_info.serial_number))

        # ctx = rs.context()
        # devices = ctx.query_devices()
        # for dev in devices:
        #     dev.hardware_reset()

        print(self.__BoldText + "Device Line: " + self.__DefaultText + "{}".format(self.product_line))
        print(self.__BoldText + "Device SN: " + self.__DefaultText + "{}".format(self.product_id))

        # Set stream resolution
        self._config.enable_stream(rs.stream.color, self.resolution_color[0], self.resolution_color[1],
                                   rs.format.bgr8, self.fps_color)
        self._config.enable_stream(rs.stream.depth, self.resolution_depth[0], self.resolution_depth[1],
                                   rs.format.z16, self.fps_depth)

        self._profile = self._pipeline.start(self._config)

        # get depth scale
        depth_sensor = self._profile.get_device().first_depth_sensor()
        # self._profile.get_device().hardware_reset()

        self._depth_scale = depth_sensor.get_depth_scale()
        print(self.__BoldText + "Depth scale: " + self.__DefaultText + "{}".format(self._depth_scale))

        # get intrinsic matrix & distortion coefficient
        color_profile = rs.video_stream_profile(self._profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        self._distCoeffs = np.array(color_intrinsics.coeffs)
        print(self.__BoldText + "Lens distortion coefficients: " + self.__DefaultText + "{}".format(self._distCoeffs))

        w, h = color_intrinsics.width, color_intrinsics.height
        fx, fy = color_intrinsics.fx, color_intrinsics.fy
        cx, cy = color_intrinsics.ppx, color_intrinsics.ppy

        self._camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self._fx = fx
        self._fy = fy

        # Get align frame
        align_to = rs.stream.color
        self._align_frame = rs.align(align_to)

        self.connected = True

        return

    def get_config(self) -> [np.ndarray, np.ndarray, float]:

        return self.camera_matrix, self.distCoeffs, self.depth_scale

    def disconnect(self):
        self._pipeline.stop()
        print(self.__BoldText + self.__BlueText + 'Stop streaming...' + self.__BlackText + self.__DefaultText)

        return

    def get_color(self, order: str = 'cv') -> np.ndarray:

        frame = self._pipeline.wait_for_frames()
        frame = self._align_frame.process(frame)  ## get aligned frame

        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        if order == 'plt':
            return color_image[:, :, ::-1]
        else:
            return color_image

    def get_depth(self, order: str = 'cv') -> [np.ndarray, np.ndarray]:

        frame = self._pipeline.wait_for_frames()
        frame = self._align_frame.process(frame)  ## get aligned frame

        depth_frame = frame.get_depth_frame()
        depth_map = np.asanyarray(depth_frame.get_data())*self.depth_scale

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.2/self.depth_scale), cv2.COLORMAP_JET)

        if order == 'plt':
            return depth_map, depth_image[:, :, ::-1]
        else:
            return depth_map, depth_image


    def get_color_depth(self, order: str = 'cv', clipping_depth: float = None) -> [np.ndarray, np.ndarray, np.ndarray]:

        frame = self._pipeline.wait_for_frames()
        frame = self._align_frame.process(frame)  ## get aligned frame

        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        depth_frame = frame.get_depth_frame()
        # depth_frame = self._decimation_filter.process(depth_frame)
        # depth_frame = self._spatial_filter.process(depth_frame)
        # depth_frame = self._hole_filling_filter.process(depth_frame)

        depth_map = np.asanyarray(depth_frame.get_data())*self.depth_scale

        if clipping_depth is not None:
            depth_threshold = clipping_depth
            depth_image_3d = np.dstack((depth_map, depth_map, depth_map))  # depth image is 1 channel, color is 3 channels
            # color_image = np.where((depth_image_3d > depth_threshold) | (depth_image_3d <= 0), 153, color_image)
            color_image = np.where((depth_image_3d > depth_threshold), 153, color_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.05/self.depth_scale), cv2.COLORMAP_JET)

        if order == 'plt':
            return color_image[:, :, ::-1], depth_map, depth_image[:, :, ::-1]
        else:
            return color_image, depth_map, depth_image
