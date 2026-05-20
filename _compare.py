"""
Headless vs GUI 1:1 비교 테스트
================================
동일한 angle/speed로 headless 예측과 GUI 실행을 비교하여
근본 원인을 확인합니다.
"""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.config import *
from project.physics.cushion_planner import CushionShotPlanner
import pybullet as pb

CX = MAZE_TABLE_CENTER_X
CY = MAZE_TABLE_CENTER_Y
L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
H = MAZE_TABLE_SURFACE_HEIGHT
TH = MAZE_TABLE_HEIGHT
ball_z = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

# 고정 공 위치
cue_pos = np.array([CX, CY - W/4, ball_z])
tgt1_pos = np.array([CX, CY + W/8, ball_z])
tgt2_pos = np.array([CX + L/6, CY, ball_z])

# 고정 각도/속도 (90도 = Y+ 방향)
test_angle = np.radians(90.0)
test_speed = 0.7

print("=" * 60)
print("  Headless vs GUI 1:1 비교")
print("=" * 60)
print(f"  CX={CX}, CY={CY}")
print(f"  Cue: {cue_pos[:2]}")
print(f"  Target1: {tgt1_pos[:2]}")
print(f"  Target2: {tgt2_pos[:2]}")
print(f"  Angle: 90deg (Y+ 방향)")
print(f"  Speed: {test_speed} m/s")
print(f"  Strike angle: {MAZE_STRIKE_ANGLE_DEG}deg")
print()

bounds = {
    'x_min': CX - L/2, 'x_max': CX + L/2,
    'y_min': CY - W/2, 'y_max': CY + W/2,
}
planner = CushionShotPlanner(table_bounds=bounds)

# ===== HEADLESS 시뮬 =====
print("[1] Headless Robot PD 시뮬")
env = planner._create_robot_env(cue_pos, tgt1_pos, tgt2_pos, [])
(sim, cue_id, t1_id, t2_id, cushions, obs, tool_id,
 robot_id, joints, ee, cid, ik, pin) = env

# 공 초기 위치 확인
cue_before, _ = pb.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
print(f"  Cue before: [{cue_before[0]:.4f}, {cue_before[1]:.4f}, {cue_before[2]:.4f}]")

score, info = planner._simulate_one_robot(
    sim, cue_id, t1_id, t2_id, cushions, tool_id,
    robot_id, joints, ik, pin,
    cue_pos, tgt1_pos, tgt2_pos, test_angle, test_speed)

cue_after, _ = pb.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
cue_vel, _ = pb.getBaseVelocity(cue_id, physicsClientId=sim)
print(f"  Cue after:  [{cue_after[0]:.4f}, {cue_after[1]:.4f}]")
print(f"  Cue velocity: [{cue_vel[0]:.4f}, {cue_vel[1]:.4f}] speed={np.linalg.norm(cue_vel[:2]):.4f}")
print(f"  Cushions: {info['cushion_count']}")
print(f"  Hit T1: {info['hit_t1']}, Hit T2: {info['hit_t2']}")
print(f"  Score: {score}")

# 궤적 첫/마지막 점
path = info.get('cue_path', [])
if len(path) > 0:
    print(f"  Path start: [{path[0][0]:.4f}, {path[0][1]:.4f}]")
    print(f"  Path end:   [{path[-1][0]:.4f}, {path[-1][1]:.4f}]")
    print(f"  Path length: {len(path)} pts")

pb.disconnect(sim)

# ===== GUI 시뮬 =====
print()
print("[2] GUI Robot PD 시뮬 (동일 angle/speed)")

from project.robot_controller import RobotController
from project.trajectory_planner import StrikeTrajectoryPlanner
from project.environment.maze_env import MazeEnvironment

controller = RobotController(mode='sim')
controller.connect()
time.sleep(1)
controller.move_home()
time.sleep(1)

robot_id_gui = controller.pb.my_robot.robotId
ee_link = controller.pb.my_robot.RobotEEJointIdx[-1]
client = controller.pb.ClientId

env_gui = MazeEnvironment(client)
env_gui.setup(
    cue_pos=list(cue_pos),
    target_pos=list(tgt1_pos),
    ball2_pos=list(tgt2_pos),
    num_obstacles=0
)
env_gui.disable_robot_env_collision(robot_id_gui)
env_gui.attach_compact_tool(robot_id_gui, ee_link)
env_gui.disable_tool_env_collision()

controller.boost_pd_gains(kp=800, kd=40)
controller.set_environment(env_gui)
time.sleep(3)

# 큐볼 초기 위치 확인
cue_gui_before, _ = pb.getBasePositionAndOrientation(
    env_gui.cue_ball_id, physicsClientId=client)
print(f"  Cue before: [{cue_gui_before[0]:.4f}, {cue_gui_before[1]:.4f}, {cue_gui_before[2]:.4f}]")

# 3D 방향 계산 (headless와 동일)
angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
dx = np.cos(test_angle) * np.cos(angle_rad)
dy = np.sin(test_angle) * np.cos(angle_rad)
dz = -np.sin(angle_rad)
strike_dir_3d = np.array([dx, dy, dz])
strike_dir_3d /= np.linalg.norm(strike_dir_3d)

print(f"  Strike dir 3D: [{strike_dir_3d[0]:.4f}, {strike_dir_3d[1]:.4f}, {strike_dir_3d[2]:.4f}]")

# 궤적 생성 (phi=0 고정 — headless와 동일)
T_home = controller.get_current_T()
traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)
tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS

trajectory, phases = traj_planner.plan_strike(
    T_current=T_home,
    ball_pos=cue_pos,
    strike_direction=strike_dir_3d,
    strike_speed=test_speed,
    approach_dist=STRIKE_APPROACH_DIST,
    follow_dist=STRIKE_FOLLOW_DIST,
    strike_height=cue_pos[2],
    tool_offset=tool_offset,
    tool_rotation=0.0  # phi=0, headless와 동일
)

# IK 검증
q_current = controller.get_current_q()
result = controller.ik.solve_trajectory_validated(q_current, trajectory)
print(f"  IK valid: {result['valid']}, min_w: {result['min_manipulability']:.4f}")
if not result['valid']:
    print(f"  IK issues: {result['issues'][:3]}")

# 실행
print(f"  Executing trajectory ({len(trajectory)} pts)...")
controller.execute_trajectory(
    trajectory, dt=0.002, phase_indices=phases, strike_speed=test_speed)

# 결과
time.sleep(2)
cue_gui_after, _ = pb.getBasePositionAndOrientation(
    env_gui.cue_ball_id, physicsClientId=client)
cue_gui_vel, _ = pb.getBaseVelocity(
    env_gui.cue_ball_id, physicsClientId=client)
print(f"  Cue after:  [{cue_gui_after[0]:.4f}, {cue_gui_after[1]:.4f}]")
print(f"  Cue velocity: [{cue_gui_vel[0]:.4f}, {cue_gui_vel[1]:.4f}] speed={np.linalg.norm(cue_gui_vel[:2]):.4f}")

# 비교
diff = np.linalg.norm(np.array(cue_after[:2]) - np.array(cue_gui_after[:2]))
print()
print("=" * 60)
print(f"  HEADLESS final: [{cue_after[0]:.4f}, {cue_after[1]:.4f}]")
print(f"  GUI final:      [{cue_gui_after[0]:.4f}, {cue_gui_after[1]:.4f}]")
print(f"  DIFF: {diff*1000:.1f}mm")
print("=" * 60)

time.sleep(5)
controller.disconnect()
