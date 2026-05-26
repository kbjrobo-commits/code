import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import pyrealsense2 as rs
from project.config import *
from preject.environment.maze_env import MazeEnvironment

TABLE_WIDTH_MM = int((MAZE_TABLE_LENGTH + 0.06) * 1000)
TABLE_HEIGHT_MM = int((MAZE_TABLE_WIDTH + 0.06) * 1000)

DISPLAY_WIDTH = int((MAZE_TABLE_LENGTH + 0.06) * 1000)
DISPLAY_HEIGHT = int((MAZE_TABLE_WIDTH + 0.06) * 1000)

aruco_size = 30

# aruco
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)

# HSV
WHITE_LOWER = np.array([0, 0, 180])
WHITE_UPPER = np.array([180, 60, 255])

RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])

RED_LOWER2 = np.array([170, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])

# BLACK_LOWER = np.array([35, 80, 50])
# BLACK_UPPER = np.array([85, 255, 255])


def detect_balls() :
    # Calibration Load
    calib = np.load("calibration_result.npz")

    K = calib["K"]
    dist = calib["dist"]

    # 추가한 부분 1

    # Realsense 초기화
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        1280,
        720,
        rs.format.bgr8,
        30
    )

    config.enable_stream(
        rs.stream.depth,
        1280,
        720,
        rs.format.z16,
        30
    )

    profile = pipeline.start(config)

    # calibration 적용
    newK, roi = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        (1280, 720),
        1,
        (1280, 720)
    )

    mapx, mapy = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        newK,
        (1280, 720),
        cv2.CV_32FC1
    )
    # 추가한 부분 2

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale:", depth_scale)


    def get_homography_from_aruco(frame):
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is None:
            return None

        ids = ids.flatten()
        marker_dict = {}

        for marker_corner, marker_id in zip(corners, ids):
            pts = marker_corner.reshape((4, 2))
            center = np.mean(pts, axis=0)

            marker_dict[marker_id] = center

        required_ids = [0, 1, 2, 3]

        for rid in required_ids:
            if rid not in marker_dict:
                print(rid, marker_dict)
                return None

        src_pts = np.array([
            marker_dict[0],  # TL
            marker_dict[1],  # TR
            marker_dict[2],  # BR
            marker_dict[3]   # BL
        ], dtype=np.float32)

        dst_pts = np.array([
            [aruco_size/2, aruco_size/2],
            [DISPLAY_WIDTH - aruco_size/2, aruco_size/2],
            [DISPLAY_WIDTH - aruco_size/2, DISPLAY_HEIGHT - aruco_size/2],
            [aruco_size/2, DISPLAY_HEIGHT - aruco_size/2]
        ], dtype=np.float32) # 수정 필요

        H = cv2.getPerspectiveTransform(
            src_pts,
            dst_pts
        )

        return H

    def remove_corner_regions(mask):
        cleaned = mask.copy()

        pad = 60

        h, w = cleaned.shape

        # TL
        cleaned[0:pad, 0:pad] = 0

        # TR
        cleaned[0:pad, w-pad:w] = 0

        # BR
        cleaned[h-pad:h, w-pad:w] = 0

        # BL
        cleaned[h-pad:h, 0:pad] = 0

        return cleaned

    def pixel_to_table(cx, cy):
        table_x = (
            cx / DISPLAY_WIDTH
        ) * TABLE_WIDTH_MM

        table_y = (
            (DISPLAY_HEIGHT - cy)
            / DISPLAY_HEIGHT
        ) * TABLE_HEIGHT_MM

        return table_x, table_y

    def detect_ball(mask, frame, color):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_ball = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 너무 작은 건 제거
            # 숫자 흰색 원 제거용
            if area < 300:
                continue

            perimeter = cv2.arcLength(cnt, True)

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * area /
                (perimeter * perimeter)
            )

            if circularity < 0.7:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            bbox_area = w * h

            # 가장 큰 객체만 선택
            if bbox_area > max_area:
                max_area = bbox_area

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                # cx = int(cx)
                # cy = int(cy)

                table_x, table_y = pixel_to_table(cx, cy)

                best_ball = {
                    "position": (
                        float(table_x),
                        float(table_y)
                    ),
                    "center": (cx, cy),
                    "radius": radius
                }

        if best_ball is not None:
            cx, cy = best_ball["center"]
            radius = best_ball["radius"]

            draw_cx = int(cx)
            draw_cy = int(cy)   

            cv2.circle(
                frame,
                (draw_cx, draw_cy),
                int(radius),
                color,
                3
            )

            return best_ball["position"]

        return None

    # 초기 프레임 잡기(더미 촬영)
    for _ in range(30):
        pipeline.wait_for_frames()

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        # calibration 적용
        undistorted = cv2.remap(
            color_image,
            mapx,
            mapy,
            cv2.INTER_LINEAR
        )
        # 추가한 부분 3

        # cv2.imshow("raw", color_image)
        # cv2.imshow("undistorted", undistorted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )

        depth_mm = (
            depth_image *
            depth_scale *
            1000
        )
        H = get_homography_from_aruco(
            # color_image
            undistorted
        )

        if H is None:
            print("Need ArUco markers 0,1,2,3")
            continue

        warped_color = cv2.warpPerspective(
            # color_image,
            undistorted,
            H,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        warped_depth = cv2.warpPerspective(
            depth_mm,
            H,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        hsv = cv2.cvtColor(
            warped_color,
            cv2.COLOR_BGR2HSV
        )

        white_mask = cv2.inRange(
            hsv,
            WHITE_LOWER,
            WHITE_UPPER
        )

        red_mask1 = cv2.inRange(
            hsv,
            RED_LOWER1,
            RED_UPPER1
        )

        red_mask2 = cv2.inRange(
            hsv,
            RED_LOWER2,
            RED_UPPER2
        )

        red_mask = cv2.bitwise_or(
            red_mask1,
            red_mask2
        )

        yellow_mask = cv2.inRange(
            hsv,
            YELLOW_LOWER,
            YELLOW_UPPER
        )

        blue_mask = cv2.inRange(
            hsv,
            BLUE_LOWER,
            BLUE_UPPER
        )

        # black_mask = cv2.inRange(
        #     hsv,
        #     BLACK_LOWER,
        #     BLACK_UPPER
        # )

        white_mask = remove_corner_regions(
            white_mask
        )

        red_mask = remove_corner_regions(
            red_mask
        )

        yellow_mask = remove_corner_regions(
            yellow_mask
        )

        # blue_mask = remove_corner_regions(
        #     blue_mask
        # )
        
        # black_mask = remove_corner_regions(
        #     black_mask
        # )

        white_ball = detect_ball(
            white_mask,
            warped_color,
            (255, 255, 255)
        )

        red_ball = detect_ball(
            red_mask,
            warped_color,
            (0, 0, 255)
        )

        yellow_ball = detect_ball(
            yellow_mask,
            warped_color,
            (0, 255, 255)
        )
        cv2.imshow("Warped Result", warped_color)
        cv2.imshow("red", red_mask)
        cv2.imshow("yellow", yellow_mask)
        cv2.imshow("white", white_mask)

        # 세 공 모두 검출 성공
        if (
            white_ball is not None and
            red_ball is not None and
            yellow_ball is not None
        ):
            print("All balls detected")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    white_ball = [
        white_ball[0] / 1000.0,
        white_ball[1] / 1000.0
    ]

    red_ball = [
        red_ball[0] / 1000.0,
        red_ball[1] / 1000.0
    ]

    yellow_ball = [
        yellow_ball[0] / 1000.0,
        yellow_ball[1] / 1000.0
    ]

    # result = {
    #     "White balls": white_ball,
    #     "Red balls": red_ball,
    #     "Yellow balls": yellow_ball
    # }

    L = MAZE_TABLE_LENGTH
    W = MAZE_TABLE_WIDTH
    H = MAZE_TABLE_SURFACE_HEIGHT
    CX = MAZE_TABLE_CENTER_X
    CY = MAZE_TABLE_CENTER_Y
    CH = MAZE_CUSHION_HEIGHT
    TH = MAZE_TABLE_HEIGHT
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
    thickness = 0.003

    center = np.array([CX, CY, H])

    # x_offset = center[0] - L/2
    y_offset = - center[1] - L/2 - thickness
    z_offset = MAZE_BALL_RADIUS + H

    cue_pos = [
        white_ball[0],
        white_ball[1] + y_offset,
        z_offset
    ]

    target_pos = [
        yellow_ball[0],
        yellow_ball[1] + y_offset,
        z_offset
    ]

    ball2_pos = [
        red_ball[0],
        red_ball[1] + y_offset,
        z_offset
    ]
    return cue_pos, target_pos, ball2_pos

# print("\n===== DETECTION RESULT =====")
# print(result)

# cv2.imshow(
#     "Warped Result",
#     warped_color
# )

# cv2.waitKey(0)

# pipeline.stop()
# cv2.destroyAllWindows()


"""
아래 부분은 pybullet 환경 구현 하드코딩한 부분
"""

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# p.resetDebugVisualizerCamera(
#     cameraDistance=0.7,
#     cameraYaw=0,
#     cameraPitch=-89.9,
#     cameraTargetPosition=[
#         MAZE_TABLE_CENTER_X,
#         MAZE_TABLE_CENTER_Y,
#         MAZE_TABLE_SURFACE_HEIGHT
#     ]
# ) # 카메라 탑뷰로 보여줌

# L = MAZE_TABLE_LENGTH
# W = MAZE_TABLE_WIDTH
# H = MAZE_TABLE_SURFACE_HEIGHT
# CX = MAZE_TABLE_CENTER_X
# CY = MAZE_TABLE_CENTER_Y
# CH = MAZE_CUSHION_HEIGHT
# TH = MAZE_TABLE_HEIGHT
# ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
# mass = MAZE_BALL_MASS

# center = np.array([CX, CY, H])

# col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2])
# vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], rgbaColor=COLOR_FELT_GREEN)
# table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
#                                           baseVisualShapeIndex=vis,
#                                           basePosition=[center[0], center[1], center[2]])
# p.changeDynamics(table_id, -1, lateralFriction=MAZE_BALL_FRICTION, restitution=0.5)

# top_z = center[2] + TH / 2 + CH / 2
# thickness = 0.03

# configs = [
#     ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
#     ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
#     ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
#     ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
# ]
# for pos, half_ext in configs:
#     col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
#     vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=COLOR_BROWN)
#     cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
#     p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION)

# balls = {
#     "white": {
#         "rgba": COLOR_WHITE,
#         "positions": white_ball
#     },

#     "red": {
#         "rgba": COLOR_RED,
#         "positions": red_ball
#     },

#     "yellow": {
#         "rgba": COLOR_YELLOW,
#         "positions": yellow_ball
#     }
# }

# x_offset = center[0] - L/2
# y_offset = center[1] - W/2

# for color_type, info in balls.items():
#     color=info["rgba"]
#     pos=info["positions"]
#     position = [pos[0]+x_offset, pos[1]+y_offset, ball_h]
#     col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS)
#     vis = p.createVisualShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, rgbaColor=color)
#     bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=position)
#     p.changeDynamics(
#         bid, 
#         -1,
#         lateralFriction=MAZE_BALL_FRICTION,
#         restitution=MAZE_BALL_RESTITUTION,
#         rollingFriction=MAZE_BALL_ROLLING_FRICTION,
#         spinningFriction=0.02,
#         ccdSweptSphereRadius=MAZE_BALL_RADIUS * 0.5,
#         contactProcessingThreshold=0
#     )

# while True:
#     p.stepSimulation()
#     time.sleep(1/240)

"""
윗 부분은 pybullet 환경 구현 하드코딩한 부분
"""

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# p.resetDebugVisualizerCamera(
#     cameraDistance=0.7,
#     cameraYaw=0,
#     cameraPitch=-89.9,
#     cameraTargetPosition=[
#         MAZE_TABLE_CENTER_X,
#         MAZE_TABLE_CENTER_Y,
#         MAZE_TABLE_SURFACE_HEIGHT
#     ]
# )
# # 카메라는 그냥 탑뷰로 설정함

# client_id = 0

# env = MazeEnvironment(client_id)

# TH = MAZE_TABLE_HEIGHT
# H = MAZE_TABLE_SURFACE_HEIGHT

# ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

# L = MAZE_TABLE_LENGTH
# W = MAZE_TABLE_WIDTH
# CX = MAZE_TABLE_CENTER_X
# CY = MAZE_TABLE_CENTER_Y

# x_offset = CX - L / 2 - 0.03
# y_offset = CY - W / 2 - 0.03

# cue_pos = [
#     white_ball[0] + x_offset,
#     white_ball[1] + y_offset,
#     ball_h
# ]

# target_pos = [
#     yellow_ball[0] + x_offset,
#     yellow_ball[1] + y_offset,
#     ball_h
# ]

# ball2_pos = [
#     red_ball[0] + x_offset,
#     red_ball[1] + y_offset,
#     ball_h
# ]

# env.setup(
#     cue_pos=cue_pos,
#     target_pos=target_pos,
#     ball2_pos=ball2_pos,
#     num_obstacles=0
# )

# while True:
#     p.stepSimulation()
#     time.sleep(1 / 240)

if __name__ == "__main__" :
    result = detect_balls()
    print(result)