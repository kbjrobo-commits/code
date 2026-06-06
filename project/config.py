"""
프로젝트 전체 설정
==================
미니골프 퍼팅 + 포켓볼 타격 통합 프로젝트
"""
import numpy as np

# ============================================================
# 모드 설정
# ============================================================
MODE = 'sim'                    # 'sim' (PyBullet) or 'real' (IndyDCP3)
ROBOT_IP = '192.168.0.10'      # 실제 로봇 IP

# ============================================================
# 로봇 파라미터
# ============================================================
HOME_Q_DEG = [0, -15, -75, 0, -90, 0]         # 홈 포지션 (deg)
HOME_Q_RAD = np.array(HOME_Q_DEG) * np.pi / 180  # 홈 포지션 (rad)
MAX_TOOL_SPEED = 0.7           # 최대 툴 속도 (m/s) — 로봇 물리적 한계

# ============================================================
# 궤적 생성 파라미터
# ============================================================
TRAJECTORY_DT = 0.001          # 궤적 시간 스텝 (s) = 1ms
IK_GAIN = 1.0                 # IK 스텝 게인
IK_DAMPING = 1e-3             # Damped Least Squares 감쇠 계수
IK_MAX_ITER = 10              # IK 반복 횟수 per waypoint

# ============================================================
# 타격 파라미터 (공통)
# ============================================================
STRIKE_APPROACH_DIST = 0.10    # 타격 전 접근 거리 (m) — 시뮬/실제 균형 (0.08→0.10)
STRIKE_FOLLOW_DIST = 0.04     # Follow-through 거리 (m) — 테이블 관통 방지용 축소
APPROACH_DURATION = 3.0        # 접근 궤적 시간 (s)
STRIKE_HEIGHT_OFFSET = 0.0     # 타격 높이 오프셋 (m)
RETRACT_HEIGHT = 0.15          # 타격 후 수직 상승 높이 (m)

# ============================================================
# 타격 도구 파라미터 — ㄴ자 큐팁 도구
# ============================================================
# EE에서 아래로 내려온 뒤 수평으로 뻗는 ㄴ자 형태
# 끝단에 실제 큐대 팁(13mm) 부착
#
#   EE (로봇 끝단)
#    |
#    | ← TOOL_VERTICAL_DROP (60mm)
#    |
#    └────● ← TOOL_HORIZONTAL_EXT (30mm), 끝에 큐팁
#
TOOL_VERTICAL_DROP = 0.06       # EE에서 수직 하강 (m)
TOOL_HORIZONTAL_EXT = 0.03      # 꺾인 후 수평 연장 (m)
TOOL_TIP_RADIUS = 0.0065        # 큐팁 반경 (m) — 직경 13mm
TOOL_TIP_LENGTH = 0.015         # 큐팁 두께 (m)
TOOL_HEAD_MASS = 0.15           # 도구 물리 질량 (kg)
HEADLESS_TOOL_MASS = 0.15       # Headless도 동일 질량 (물리 일관성)
TOOL_HEAD_RESTITUTION = 0.9     # 반발 계수
TOOL_CONSTRAINT_FORCE = 5000    # Constraint 최대 힘 (N)
# 실제 도구 장착 z축 회전 오프셋 (rad)
# 도구가 EE z축 기준으로 틀어진 각도. 위에서 봤을 때 반시계=양수.
# 예: -15° → 도구 수평부가 EE x축에서 시계방향으로 15° 틀어짐
TOOL_YAW_OFFSET = np.radians(0.0)
# Pinocchio FK vs PyBullet EE 프레임 Z 오프셋 보정
# Pinocchio가 PyBullet보다 ~62mm 높은 EE 위치를 반환 (URDF 프레임 정의 차이)
# IK 목표를 이만큼 높여서 PyBullet에서 올바른 위치에 도달하도록 보정
PIN_PB_EE_Z_OFFSET = 0.062
# 이전 코드 호환용 (직선 도구 시절의 변수 유지)
TOOL_HEAD_LENGTH = TOOL_VERTICAL_DROP  # attach_compact_tool 호환
TOOL_HEAD_RADIUS = TOOL_TIP_RADIUS     # attach_compact_tool 호환


# ============================================================
# 색상 (PyBullet RGBA)
# ============================================================
COLOR_WHITE = [1, 1, 1, 1]
COLOR_RED = [0.9, 0.1, 0.1, 1]
COLOR_GREEN = [0.1, 0.7, 0.1, 1]
COLOR_BLUE = [0.1, 0.1, 0.9, 1]
COLOR_YELLOW = [1, 0.9, 0.1, 1]
COLOR_DARK_GREEN = [0.05, 0.35, 0.05, 1]
COLOR_BROWN = [0.5, 0.3, 0.1, 1]
COLOR_FELT_GREEN = [0.0, 0.5, 0.15, 1]
COLOR_GOLF_GREEN = [0.2, 0.6, 0.2, 1]
COLOR_HOLE_BLACK = [0.05, 0.05, 0.05, 1]
COLOR_STEEL = [0.7, 0.7, 0.75, 1]
COLOR_WOOD = [0.55, 0.35, 0.15, 1]
COLOR_OBSTACLE = [0.6, 0.2, 0.2, 1]
COLOR_BLACK_BALL = [0.08, 0.08, 0.08, 1]

# ============================================================
# 현재 포켓볼 데모 파라미터
# ============================================================
MAZE_TABLE_LENGTH = 0.305          # 테이블 길이 X (m)
MAZE_TABLE_WIDTH = 0.635           # 테이블 폭 Y (m)
MAZE_TABLE_HEIGHT = 0.02           # 테이블 두께 (m)
MAZE_TABLE_SURFACE_HEIGHT = 0.106  # 테이블 바닥면 높이 (m)
MAZE_TABLE_CENTER_X = 0.485        # 테이블 중심 X (원래 위치 복원)
MAZE_TABLE_CENTER_Y = 0.165         # 테이블 중심 Y (원래 위치 복원)
MAZE_GRID_SPACING = 0.05           # 자석 그리드 간격 (m)
MAZE_OBSTACLE_RADIUS = 0.015       # 장애물 원기둥 반지름 (m)
MAZE_OBSTACLE_HEIGHT = 0.05        # 장애물 높이 (m)
MAZE_CUSHION_HEIGHT = 0.015         # 쿠션 높이 (m)
MAZE_CUSHION_RESTITUTION = 0.50   # 쿠션 반발계수
MAZE_BALL_RADIUS = 0.012           # 큐볼 반지름 (m) — 지름 24mm
MAZE_BALL_MASS = 0.01              # 큐볼 질량 (kg) — 가벼운 공
MAZE_BALL_RESTITUTION = 0.8       # 큐볼 반발계수
MAZE_BALL_FRICTION = 0.3        # 실제 당구공 수준 (0.3은 과도)
MAZE_BALL_ROLLING_FRICTION = 0.026  # 올림: 실측 기반 보정 (0.008→0.012)a
MAZE_STRIKE_ANGLE_DEG = 0          # 수평 타격 (ㄴ자 도구로 수평으로 침)

# 사이드 포켓(홀) — 긴 변(y±, 폭 0.63m) 레일 정중앙 2곳 (짧은 변 x± 모서리 아님)!!
# 플래너만 사용: PyBullet에 구멍 없음, 첫 쿠션 후 홀 쪽 경로 탐색 제외
MAZE_SIDE_POCKET_AVOID = True
MAZE_SIDE_POCKET_HALF_LENGTH = 0.025   # 레일 방향(X) 홀 개구부 반길이 → 총 ~5cm
MAZE_SIDE_POCKET_INWARD_DEPTH = 0.07   # 레일에서 테이블 안쪽 회피 깊이 (m)
MAZE_SIDE_POCKET_MARGIN = 0.012        # 홀 구간 추가 여유 (m)

# ============================================================
# 어닐링 탐색 파라미터
# ============================================================
ANNEAL_N_INITIAL = 300             # 초기 광역 샘플 수 (빠른 공 시뮬)
ANNEAL_N_REFINE_ROUNDS = 2         # 정밀화 라운드 수
ANNEAL_TOP_RATIO = 0.10            # 상위 선택 비율
ANNEAL_SIGMA_ANGLE = [30, 10, 3]   # 각도 분산 축소 (degrees)
ANNEAL_SIGMA_SPEED = [0.15, 0.05, 0.02]  # 속도 분산 축소 (m/s)
ANNEAL_SPEED_RANGE = (0.5, 1.8)    # 공 속도 범위 (m/s) — 로봇 달성 가능 범위
ANNEAL_MAX_CUSHIONS = 6            # 최대 쿠션 반사 횟수
ANNEAL_ROLLING_FRICTION = MAZE_BALL_ROLLING_FRICTION

# Tool-speed -> cue-ball-speed model. The default gain is the 1D elastic
# collision estimate using the effective tool head mass and restitution.
BALL_SPEED_GAIN_SCALE = 1.0
BALL_SPEED_GAIN = (
    (1.0 + np.sqrt(TOOL_HEAD_RESTITUTION * MAZE_BALL_RESTITUTION))
    * TOOL_HEAD_MASS / (TOOL_HEAD_MASS + MAZE_BALL_MASS)
    * BALL_SPEED_GAIN_SCALE
)

# ============================================================
# 포켓볼 데모 파라미터
# ============================================================
POCKET_DEMO_CUSHION_RESTITUTION = 0.5  # 쿠션 반발계수 (낮음, 붙는 현상)

# 포켓 6개 (코너 4 + 사이드 2)
POCKET_RADIUS = 0.0225                # 포켓 반경 (m) — 직경 45mm

# ============================================================
# 공유 접촉 물리 모델 — GUI/Headless 단일 소스 (Goal 2)
# ============================================================
# GUI(maze_env)와 Headless(pocket_planner) 양쪽이 이 값들만 사용하도록 통일한다.
# 적용은 project/physics/contact_model.py 의 헬퍼를 통해 한 곳에서 수행.
#
# 주의(롤링마찰): PyBullet(Bullet)은 두 바디의 롤링마찰을 조합한다
#   combinedRolling = ballRolling*tableLateral + tableRolling*ballLateral
# 따라서 파라미터값 0.026을 두 바디에 넣어도 "유효 접촉 롤링마찰"은 0.026이 아니다.
# 본 설정은 측정값을 rollingFriction "파라미터"로 0.026 지정하는 방식(사용자 확정)이며,
# 핵심은 GUI와 Headless가 동일 값을 쓰도록 하는 것이다.
LATERAL_FRICTION = MAZE_BALL_FRICTION       # 0.3  측면 마찰 (공·테이블 공통)
ROLLING_FRICTION = 0.026                    # 실측 구름마찰계수 (공·테이블 공통)
SPINNING_FRICTION = 0.02                    # 스핀 마찰 (공)
BALL_RESTITUTION = MAZE_BALL_RESTITUTION    # 0.8   공 반발계수
TABLE_RESTITUTION = 0.5                     # 테이블 반발계수
CUSHION_RESTITUTION = POCKET_DEMO_CUSHION_RESTITUTION  # 0.5

# 레거시 이름 호환(단일 소스는 위 LATERAL/ROLLING_FRICTION)
POCKET_DEMO_FRICTION = LATERAL_FRICTION
POCKET_DEMO_ROLLING_FRICTION = ROLLING_FRICTION
POCKET_DEMO_BALL_RESTITUTION = BALL_RESTITUTION

# ============================================================
# 포켓 성공 판정 임계값 (Goal 5)
# ============================================================
# 테이블은 실제 구멍이 없는 솔리드 박스라, 포켓 판정은 "공 중심 ↔ 포켓점" 거리로 내린다.
# 기존 임계값(POCKET_RADIUS=0.0225)은 공 중심이 홀 가장자리만 스쳐도 성공 → 과도하게 관대.
# 적용값(strict): 공이 홀 위에 완전히 들어와야 성공.
#   strict       = POCKET_RADIUS - MAZE_BALL_RADIUS        = 0.0105 m  (적용)
#   practical    = POCKET_RADIUS - MAZE_BALL_RADIUS*0.5    ≈ 0.0165 m
#   conservative = POCKET_RADIUS                           = 0.0225 m  (기존)
POCKET_CAPTURE_RADIUS = POCKET_RADIUS - MAZE_BALL_RADIUS   # 0.0105 m (strict)

# Phase 2: 정밀 정지
PRECISION_STOP_TOLERANCE = 0.01      # 허용 오차 1cm
PRECISION_SPEED_RANGE = (0.5, 2.5)   # 속도 탐색 범위 (m/s) — 미니 테이블+마찰 대비
PRECISION_SPEED_STEPS = 25           # 속도 탐색 해상도

# Phase 2: 초기 배치 (y축 중심선 위 일렬)
LINEUP_SPACING = 0.08                # 목적구 간 간격 (m)
LINEUP_CUE_OFFSET = 0.12             # 큐볼-첫 목적구 간격 (m)

# ============================================================
# Escape shot parameters
# ============================================================
ESCAPE_WALL_GAP_THRESHOLD = 0.03 + MAZE_BALL_RADIUS
ESCAPE_STRIKE_HEIGHT_OFFSET = 0.016
ESCAPE_BALL_SPEED = 0.45
ESCAPE_SAFE_APPROACH_DIST = 0.035
ESCAPE_FOLLOW_DIST = 0.025

