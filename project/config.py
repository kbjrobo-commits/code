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
MAX_TOOL_SPEED = 1.0           # 최대 툴 속도 (m/s)

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
STRIKE_APPROACH_DIST = 0.08    # 타격 전 접근 거리 (m)
STRIKE_FOLLOW_DIST = 0.10     # Follow-through 거리 (m)
APPROACH_DURATION = 3.0        # 접근 궤적 시간 (s)
STRIKE_HEIGHT_OFFSET = 0.0     # 타격 높이 오프셋 (m)

# ============================================================
# 미니골프 파라미터
# ============================================================
MINIGOLF_BALL_RADIUS = 0.0214    # 골프공 반지름 (m)
MINIGOLF_BALL_MASS = 0.046      # 골프공 질량 (kg)
MINIGOLF_HOLE_RADIUS = 0.054    # 홀 컵 반지름 (m)
MINIGOLF_TERRAIN_SIZE = [1.5, 0.8]  # 지형 크기 (m)
MINIGOLF_TERRAIN_RESOLUTION = 150    # 지형 메쉬 해상도 (1cm 단위 정밀도)
MINIGOLF_STRIKE_SPEED = 0.5    # 미니골프 타격 속도 (m/s)

# 미니골프 물리
MINIGOLF_BALL_FRICTION = 0.3
MINIGOLF_BALL_RESTITUTION = 0.5
MINIGOLF_GROUND_FRICTION = 0.4
MINIGOLF_GROUND_RESTITUTION = 0.3

# ============================================================
# 포켓볼 파라미터
# ============================================================
BILLIARD_BALL_RADIUS = 0.02625   # 당구공 반지름 (m)
BILLIARD_BALL_MASS = 0.17       # 당구공 질량 (kg)
BILLIARD_TABLE_LENGTH = 0.6     # 테이블 길이 X (m) — 로봇 도달 범위 내
BILLIARD_TABLE_WIDTH = 0.4      # 테이블 너비 Y (m) — 축소하여 전체 도달 가능
BILLIARD_TABLE_HEIGHT = 0.02    # 테이블 두께 (m)
BILLIARD_TABLE_SURFACE_HEIGHT = 0.3   # 테이블 바닥면 높이 (m)
BILLIARD_TABLE_CENTER_Y = 0.45  # 테이블 중심 Y — 로봇이 테이블 바깥(Y=0)에 위치
BILLIARD_CUSHION_HEIGHT = 0.05  # 쿠션 높이 (m) — 공 지름과 유사
BILLIARD_POCKET_RADIUS = 0.04   # 포켓 반지름 (m)
BILLIARD_STRIKE_SPEED = 0.8    # 포켓볼 타격 속도 (m/s)
BILLIARD_STRIKE_ANGLE_DEG = 15  # 빌리아드 타격 각도 (도) — 얕은 대각선 (수평 에너지 96% 보존)

# 포켓볼 물리
BILLIARD_BALL_FRICTION = 0.3
BILLIARD_BALL_RESTITUTION = 0.85      # 탄성 충돌로 임팩트 효과 강화
BILLIARD_TABLE_FRICTION = 0.4         # 높여서 마찰로 빠르게 감속
BILLIARD_BALL_ROLLING_FRICTION = 0.02
BILLIARD_BALL_SPINNING_FRICTION = 0.02

# ============================================================
# 타격 도구 파라미터 — 컴팩트 헤드 (EE 끝단 직결)
# ============================================================
# 자루 없이 EE 끝에 직접 짧은 헤드만 부착
# → 짧아서 안정적, 무게가 집중되어 임팩트 효과적

TOOL_HEAD_LENGTH = 0.03         # 헤드 길이 (m) — 짧고 컴팩트
TOOL_HEAD_RADIUS = 0.018        # 헤드 반지름 (m)
TOOL_HEAD_MASS = 0.2            # 헤드 질량 (kg) — 적절한 무게 (과하면 불안정)
TOOL_HEAD_RESTITUTION = 0.9     # 반발 계수 — 높여서 임팩트 효과 강화
TOOL_CONSTRAINT_FORCE = 5000    # Constraint 최대 힘 (N) — 높을수록 강성↑

# ============================================================
# 시뮬레이션 Grid Search 파라미터 (미니골프)
# ============================================================
GRID_ANGLE_RANGE = (-30, 30)   # 탐색 각도 범위 (deg)
GRID_ANGLE_STEP = 2.0          # 각도 탐색 간격 (deg)
GRID_SPEED_RANGE = (0.3, 1.5)  # 탐색 속도 범위 (m/s) — 괴굴 지형을 넘기 위해 높은 속도까지 탐색
GRID_SPEED_STEP = 0.05         # 속도 탐색 간격 (m/s)
GRID_SIM_DURATION = 5.0        # 가상 타격 시뮬레이션 시간 (s)
GRID_SIM_STEPS = 2000          # 가상 시뮬레이션 스텝 수 (공이 충분히 굴러가게)

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
