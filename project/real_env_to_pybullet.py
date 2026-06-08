import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import json
import pyrealsense2 as rs
from project.config import *
from project.environment.maze_env import MazeEnvironment

# 캘리브레이션 오프셋 파일 경로
_POSITION_CALIB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'calibration_position_offset.json')
_PHYSICS_CALIB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'calibration_result_physics.npz')


def load_position_offset():
    """위치 캘리브레이션 오프셋 로드. 없으면 (0, 0)."""
    if os.path.exists(_POSITION_CALIB_FILE):
        with open(_POSITION_CALIB_FILE, 'r') as f:
            offset = json.load(f)
        print(f"  [CALIB] 위치 오프셋 적용: x={offset.get('x',0):.4f}m, y={offset.get('y',0):.4f}m")
        return offset
    return {'x': 0.0, 'y': 0.0}


def load_physics_calibration():
    """물리 캘리브레이션 파라미터 로드. 없으면 None."""
    if os.path.exists(_PHYSICS_CALIB_FILE):
        calib = np.load(_PHYSICS_CALIB_FILE)
        params = {}
        for k in calib.files:
            v = calib[k]
            params[k] = float(v.item()) if hasattr(v, 'item') else float(v)
        print(f"  [CALIB] 물리 파라미터 적용: {params}")
        return params
    return None

TABLE_WIDTH_MM = int((MAZE_TABLE_WIDTH + 0.06) * 1000)
TABLE_HEIGHT_MM = int((MAZE_TABLE_LENGTH + 0.06) * 1000)

DISPLAY_WIDTH = int((MAZE_TABLE_WIDTH + 0.06) * 1000)
DISPLAY_HEIGHT = int((MAZE_TABLE_LENGTH + 0.06) * 1000)

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

BLACK_LOWER = np.array([100, 100, 50])
BLACK_UPPER = np.array([130, 255, 255])

# BLACK_LOWER = np.array([35, 80, 50])
# BLACK_UPPER = np.array([85, 255, 255])

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

def detect_ball_fixed(mask, frame, color):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_ball = None
    max_score = 0

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)

        if area < 100:
            continue

        perimeter = cv2.arcLength(hull, True)

        if perimeter == 0:
            continue

        (circle_cx, circle_cy), radius = \
            cv2.minEnclosingCircle(hull)

        circle_area = np.pi * radius * radius

        fill_ratio = area / circle_area

        if fill_ratio < 0.35:
            continue

        M = cv2.moments(hull)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        table_x, table_y = pixel_to_table(cx, cy)

        score = area
        if score > max_score:
            max_score = score
            best_ball = {
                "position": (
                    float(table_x),
                    float(table_y)
                ),
                "center": (cx, cy),
                "radius": radius,
                "hull": hull
            }

    if best_ball is not None:
        cx, cy = best_ball["center"]
        radius = best_ball["radius"]

        draw_cx = int(round(cx))
        draw_cy = int(round(cy))

        cv2.circle(
            frame,
            (draw_cx, draw_cy),
            int(round(radius)),
            color,
            3
        )

        cv2.circle(
            frame,
            (draw_cx, draw_cy),
            4,
            (255, 0, 0),
            -1
        )

        cv2.drawContours(
            frame,
            [best_ball["hull"]],
            -1,
            (0, 255, 0),
            2
        )

        return best_ball["position"]

    return None


def detect_balls(ball_pocketed=[False, False, False]) : # ball_pocketed = [노, 빨, 검] = {False if not pocketed else True}
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

    # 초기 프레임 잡기(더미 촬영)
    for _ in range(30):
        pipeline.wait_for_frames()

    import time

    timeout_sec = 10.0
    start_time = time.time()

    check_detected = [False, False, False] # 노, 빨, 검
    while True:
        if time.time() - start_time > timeout_sec :
            print("Detection timeout")
            pipeline.stop()
            cv2.destroyAllWindows()
            res = [not x for x in check_detected]
            return None, res[0], res[1], res[2]

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
            #print("Need ArUco markers 0,1,2,3")
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

        # blue_mask = cv2.inRange(
        #     hsv,
        #     BLUE_LOWER,
        #     BLUE_UPPER
        # )

        black_mask = cv2.inRange(
            hsv,
            BLACK_LOWER,
            BLACK_UPPER
        )

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
        
        black_mask = remove_corner_regions(
            black_mask
        )

        white_ball = detect_ball_fixed(
            white_mask,
            warped_color,
            (255, 255, 255)
        )

        red_ball = detect_ball_fixed(
            red_mask,
            warped_color,
            (0, 0, 255)
        ) if ball_pocketed[1] is False else None
        check_detected[1] = True if (ball_pocketed[1] is True or red_ball is not None or check_detected[1] is True) else False

        yellow_ball = detect_ball_fixed(
            yellow_mask,
            warped_color,
            (0, 255, 255)
        ) if ball_pocketed[0] is False else None
        check_detected[0] = True if (ball_pocketed[0] is True or yellow_ball is not None or check_detected[0] is True) else False

        black_ball = detect_ball_fixed(
            black_mask,
            warped_color,
            (0, 0, 0)
        ) if ball_pocketed[2] is False else None
        check_detected[2] = True if (ball_pocketed[2] is True or black_ball is not None or check_detected[2] is True) else False

        cv2.imshow("Warped Result", warped_color)
        cv2.imshow("red", red_mask)
        cv2.imshow("yellow", yellow_mask)
        cv2.imshow("white", white_mask)
        cv2.imshow("black", black_mask)
        print(white_ball, red_ball, yellow_ball, black_ball)
        # 세 공 모두 검출 성공
        if (
            white_ball is not None and
            (red_ball is not None or ball_pocketed[1] is True) and
            (yellow_ball is not None or ball_pocketed[0] is True) and
            (black_ball is not None or ball_pocketed[2] is True)

        ):
            print("All balls detected")
            cv2.waitKey(0)
            pipeline.stop()
            cv2.destroyAllWindows()
            break

    white_ball = [
        white_ball[0] / 1000.0,
        white_ball[1] / 1000.0
    ]

    red_ball = [
        red_ball[0] / 1000.0,
        red_ball[1] / 1000.0
    ] if ball_pocketed[1] is False else None

    yellow_ball = [
        yellow_ball[0] / 1000.0,
        yellow_ball[1] / 1000.0
    ] if ball_pocketed[0] is False else None

    black_ball = [
        black_ball[0] / 1000.0,
        black_ball[1] / 1000.0
    ] if ball_pocketed[2] is False else None

    # 좌표 변환: 카메라(pixel_to_table) → PyBullet 좌표계
    # pixel_to_table: (0,0)=좌상단 → (TABLE_WIDTH_MM, TABLE_HEIGHT_MM)=우하단
    # PyBullet: 테이블 중심=(CX, CY)
    # ※ x_offset 미적용 (카메라 x좌표 ≈ PyBullet x좌표 가정)
    # ※ y_offset 부호/값은 실측 검증 필요 — calibration_loop.py에서 자동 보정 예정
    L = MAZE_TABLE_LENGTH
    W = MAZE_TABLE_WIDTH
    H = MAZE_TABLE_SURFACE_HEIGHT
    CX = MAZE_TABLE_CENTER_X
    CY = MAZE_TABLE_CENTER_Y
    TH = MAZE_TABLE_HEIGHT
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
    thickness = 0.03

    center = np.array([CX, CY, H])
    x_offset = 0.35 + L + 2 * thickness
    y_offset = (W + 2 * thickness) / 2 - float(center[1])  # 기본 좌표 변환

    cue_pos = [x_offset - float(white_ball[1]), float(white_ball[0]) - y_offset, float(ball_h)]
    target_pos = [x_offset - float(yellow_ball[1]), float(yellow_ball[0]) - y_offset, float(ball_h)] if ball_pocketed[0] is False else None
    ball2_pos = [x_offset - float(red_ball[1]), float(red_ball[0]) - y_offset, float(ball_h)] if ball_pocketed[1] is False else None
    ball3_pos = [x_offset - float(black_ball[1]), float(black_ball[0]) - y_offset, float(ball_h)] if ball_pocketed[2] is False else None

    for pos in [cue_pos, target_pos, ball2_pos, ball3_pos] :
        if pos is not None :
            x = pos[0]
            y = pos[1]
            if abs(x - (CX + L/2 - MAZE_BALL_RADIUS)) < 1e-6 : pos[0] = CX + L/2 - MAZE_BALL_RADIUS
            elif abs(x - (CX - L/2 + MAZE_BALL_RADIUS)) < 1e-6 : pos[0] = CX - L/2 + MAZE_BALL_RADIUS

            if abs(y - (CY + W/2 - MAZE_BALL_RADIUS)) < 1e-6 : pos[1] = CY + W/2 - MAZE_BALL_RADIUS
            elif abs(y - (CY - W/2 + MAZE_BALL_RADIUS)) < 1e-6 : pos[1] = CY - W/2 + MAZE_BALL_RADIUS

    # 캘리브레이션 오프셋 자동 적용
    pos_offset = load_position_offset()
    for pos in [cue_pos, target_pos, ball2_pos, ball3_pos]:
        if pos is not None :
            pos[0] = float(pos[0] + pos_offset.get('x', 0.0))
            pos[1] = float(pos[1] + pos_offset.get('y', 0.0))

    return cue_pos, target_pos, ball2_pos, ball3_pos


def wait_real_balls_stop(interval=0.5, threshold_mm=3.0, max_wait=10.0, verbose=True, ball_pocketed=[False, False, False]):
    """카메라로 공 정지 여부 판단.

    interval초 간격으로 2회 촬영 → 3공 모두 위치 변화 < threshold_mm이면 정지.
    max_wait초 내에 정지 안 하면 타임아웃.

    Returns:
        (cue_pos, target_pos, ball2_pos) — 최종 정지 위치
    """
    import time as _time
    start = _time.time()

    prev = None
    while _time.time() - start < max_wait:
        try:
            current = detect_balls(ball_pocketed)
        except Exception as e:
            if verbose:
                print(f"  [STOP] 검출 실패: {e}, 재시도...")
            _time.sleep(interval)
            continue

        if prev is not None:
            # 3공 모두 변위 계산
            displacements = []
            for i in range(4):
                d = np.linalg.norm(
                    np.array(current[i][:2]) - np.array(prev[i][:2])
                ) * 1000  if current[i] is not None and prev[i] is not None else 0.0 # m → mm
                displacements.append(d)

            max_disp = max(displacements)
            if verbose:
                print(f"  [STOP] 변위: cue={displacements[0]:.1f}mm, "
                      f"t1={displacements[1]:.1f}mm, t2={displacements[2]:.1f}mm, t3={displacements[3]:.1f}mm")

            if max_disp < threshold_mm:
                if verbose:
                    print(f"  [STOP] 공 정지 확인 ({_time.time()-start:.1f}초)")
                return current

        prev = current
        _time.sleep(interval)

    if verbose:
        print(f"  [STOP] 타임아웃 ({max_wait}초), 마지막 위치 반환")
    return prev if prev is not None else detect_balls(ball_pocketed)

if __name__ == "__main__":
    result = detect_balls()
    print(result)
