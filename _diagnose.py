"""
진단: 워크스페이스 + 궤적 일치 + 3쿠션 검증
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project.config import *

print("=" * 60)
print("  [1] 워크스페이스 도달성 검증")
print("=" * 60)

# Indy7 reach ≈ 0.80m (6축), 실용 범위 ≈ 0.70m (특이점 회피)
INDY7_REACH = 0.80
INDY7_SAFE = 0.70

CY = MAZE_TABLE_CENTER_Y
CX = MAZE_TABLE_CENTER_X
L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
H = MAZE_TABLE_SURFACE_HEIGHT + MAZE_TABLE_HEIGHT / 2 + MAZE_BALL_RADIUS
ball_z = H + 0.001

# 테이블 4 코너
corners = [
    ("Near-Left",  CX - L/2, CY - W/2, ball_z),
    ("Near-Right", CX + L/2, CY - W/2, ball_z),
    ("Far-Left",   CX - L/2, CY + W/2, ball_z),
    ("Far-Right",  CX + L/2, CY + W/2, ball_z),
]

# 실제 공 위치 (run_maze 기본)
ball_positions = [
    ("Cue",     CX, CY - W/4, ball_z),
    ("Target1", CX, CY + W/8, ball_z),
    ("Target2", CX + L/6, CY, ball_z),
]

# EE가 타격 시 도달해야 할 위치: 공 뒤 approach_dist + tool_offset
# strike_dir은 수평이면 EE는 공 뒤 (approach_dist + tool_offset) 거리에 위치
tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS

print(f"  Robot base: [0, 0, 0]")
print(f"  Indy7 reach: {INDY7_REACH:.2f}m, safe: {INDY7_SAFE:.2f}m")
print(f"  Table center: [0.5, {CY:.2f}, {H:.3f}]")
print(f"  Table bounds: X=[{0.5-L/2:.2f}, {0.5+L/2:.2f}], Y=[{CY-W/2:.2f}, {CY+W/2:.2f}]")
print()

print("  코너 도달성:")
for name, x, y, z in corners:
    d = np.sqrt(x**2 + y**2 + z**2)
    status = "OK" if d < INDY7_SAFE else ("WARN" if d < INDY7_REACH else "FAIL")
    print(f"    {name:12s}: d={d:.3f}m [{status}]")

print()
print("  공 위치 도달성:")
for name, x, y, z in ball_positions:
    d = np.sqrt(x**2 + y**2 + z**2)
    status = "OK" if d < INDY7_SAFE else ("WARN" if d < INDY7_REACH else "FAIL")
    print(f"    {name:12s}: [{x:.3f}, {y:.3f}] d={d:.3f}m [{status}]")

# 가장 먼 타격 위치 (EE ready = 공 + approach_dist 뒤)
print()
print("  타격 시 EE 최대 거리 (approach 시작점):")
for name, x, y, z in ball_positions:
    # 최악 케이스: 타격 방향이 로봇에서 멀어지는 경우
    dir_away = np.array([x, y, 0])
    dir_away /= np.linalg.norm(dir_away)
    ee_pos = np.array([x, y, z]) + dir_away * (STRIKE_APPROACH_DIST + tool_offset)
    d = np.linalg.norm(ee_pos)
    status = "OK" if d < INDY7_SAFE else ("WARN" if d < INDY7_REACH else "FAIL")
    print(f"    {name:12s}: d={d:.3f}m [{status}]")

# IK 실제 검증
print()
print("=" * 60)
print("  [2] IK 도달성 실제 검증 (Pinocchio)")
print("=" * 60)

from src.utils.pinocchio_utils import PinocchioModel
from project.ik_solver import IKSolver

pin = PinocchioModel('src/assets/urdf/indy7_v2/indy7_v2')
ik = IKSolver(pin, gain=IK_GAIN, damping=IK_DAMPING)

q = np.array(HOME_Q_RAD).reshape(-1,1)

angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
for name, bx, by, bz in ball_positions:
    # 직접 해당 공을 향해 타격 시 EE 위치 계산
    # 로봇→공 방향으로 타격
    horiz = np.array([bx, by]) / np.linalg.norm([bx, by])
    strike_dir = np.array([
        horiz[0] * np.cos(angle_rad),
        horiz[1] * np.cos(angle_rad),
        -np.sin(angle_rad)
    ])
    strike_dir /= np.linalg.norm(strike_dir)

    # Ready EE position
    offset = strike_dir * tool_offset
    impact_pos = np.array([bx, by, bz]) - offset
    ready_pos = impact_pos - strike_dir * STRIKE_APPROACH_DIST

    # Orientation (z along strike_dir)
    z_ax = strike_dir
    up = np.array([0,0,1.0])
    x_ax = np.cross(up, z_ax)
    if np.linalg.norm(x_ax) < 1e-6:
        x_ax = np.array([1,0,0.0])
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)

    T_ready = np.eye(4)
    T_ready[:3,:3] = np.column_stack([x_ax, y_ax, z_ax])
    T_ready[:3,3] = ready_pos

    T_impact = np.eye(4)
    T_impact[:3,:3] = T_ready[:3,:3]
    T_impact[:3,3] = impact_pos

    # IK solve
    q_r = q.copy()
    for _ in range(50):
        q_r = ik.solve_step(q_r, T_ready)
    T_fk_ready = pin.FK(q_r)
    err_ready = np.linalg.norm(T_fk_ready[:3,3] - ready_pos) * 1000

    q_i = q_r.copy()
    for _ in range(50):
        q_i = ik.solve_step(q_i, T_impact)
    T_fk_impact = pin.FK(q_i)
    err_impact = np.linalg.norm(T_fk_impact[:3,3] - impact_pos) * 1000

    w_ready = ik.manipulability(q_r)
    w_impact = ik.manipulability(q_i)

    print(f"  {name}:")
    print(f"    Ready:  err={err_ready:.1f}mm, w={w_ready:.4f}")
    print(f"    Impact: err={err_impact:.1f}mm, w={w_impact:.4f}")
    valid_r, viols_r = ik.check_joint_limits(q_r)
    valid_i, viols_i = ik.check_joint_limits(q_i)
    if not valid_r:
        print(f"    !! Ready joint limits exceeded: {viols_r}")
    if not valid_i:
        print(f"    !! Impact joint limits exceeded: {viols_i}")

# 3쿠션 검증
print()
print("=" * 60)
print("  [3] Headless 로봇 PD 1회 타격 검증")
print("=" * 60)

import pybullet as pb
from project.physics.cushion_planner import CushionShotPlanner
import time

cue_pos = np.array([CX, CY - W/4, ball_z])
tgt1_pos = np.array([CX, CY + W/8, ball_z])
tgt2_pos = np.array([CX + L/6, CY, ball_z])

bounds = {
    'x_min': CX - L/2, 'x_max': CX + L/2,
    'y_min': CY - W/2, 'y_max': CY + W/2,
}

planner = CushionShotPlanner(table_bounds=bounds)

# 타겟1 방향으로 직접 타격 (90도)
dir_2d = tgt1_pos[:2] - cue_pos[:2]
test_angle = np.arctan2(dir_2d[1], dir_2d[0])
test_speed = 0.7

print(f"  Test: angle={np.degrees(test_angle):.1f}deg, speed={test_speed} m/s")
print(f"  Cue: {cue_pos[:2]}, Target1: {tgt1_pos[:2]}")

# Fast (자유 도구)
fast_env = planner._create_fast_env(cue_pos, tgt1_pos, tgt2_pos, [])
f_sim, f_cue, f_t1, f_t2, f_cushions, f_obs, f_tool = fast_env
t0 = time.time()
score_f, info_f = planner._simulate_one_fast(
    f_sim, f_cue, f_t1, f_t2, f_cushions, f_tool,
    cue_pos, tgt1_pos, tgt2_pos, test_angle, test_speed)
dt_fast = time.time() - t0
cue_f, _ = pb.getBasePositionAndOrientation(f_cue, physicsClientId=f_sim)
pb.disconnect(f_sim)

# Robot PD
robot_env = planner._create_robot_env(cue_pos, tgt1_pos, tgt2_pos, [])
(r_sim, r_cue, r_t1, r_t2, r_cushions, r_obs,
 r_tool, r_robot, r_joints, r_ee, r_cid, r_ik, r_pin) = robot_env
t0 = time.time()
score_r, info_r = planner._simulate_one_robot(
    r_sim, r_cue, r_t1, r_t2, r_cushions, r_tool,
    r_robot, r_joints, r_ik, r_pin,
    cue_pos, tgt1_pos, tgt2_pos, test_angle, test_speed)
dt_robot = time.time() - t0
cue_r, _ = pb.getBasePositionAndOrientation(r_cue, physicsClientId=r_sim)
pb.disconnect(r_sim)

print(f"\n  Fast (free tool): cue=[{cue_f[0]:.4f}, {cue_f[1]:.4f}]")
print(f"    cushions={info_f['cushion_count']}, hit_t1={info_f['hit_t1']}, hit_t2={info_f['hit_t2']}")
print(f"    score={score_f}, time={dt_fast:.3f}s")
print(f"\n  Robot PD: cue=[{cue_r[0]:.4f}, {cue_r[1]:.4f}]")
print(f"    cushions={info_r['cushion_count']}, hit_t1={info_r['hit_t1']}, hit_t2={info_r['hit_t2']}")
print(f"    score={score_r}, time={dt_robot:.3f}s")

diff = np.linalg.norm(np.array(cue_f[:2]) - np.array(cue_r[:2]))
print(f"\n  Fast vs Robot diff: {diff*1000:.1f}mm")
