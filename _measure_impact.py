"""GUI 로봇 타격 시 실제 공 속도 측정"""
import numpy as np, sys, time
sys.path.append('.')
from project.config import *
from src.core.pybullet_core import PybulletCore
from project.ik_solver import IKSolver
from project.trajectory_planner import StrikeTrajectoryPlanner
import pybullet as p

pb = PybulletCore()
pb.connect(robot_name='indy7_v2', joint_limit=True, constraint_visualization=False)
ik = IKSolver(pb.my_robot.pinModel, gain=IK_GAIN, damping=IK_DAMPING)
planner = StrikeTrajectoryPlanner(approach_duration=APPROACH_DURATION, dt=TRAJECTORY_DT)

# 테이블 + 공 세팅
from project.environment.maze_env import MazeEnvironment
env = MazeEnvironment(pb.ClientId)
env.setup(num_obstacles=0)
env.disable_robot_env_collision(pb.my_robot.robotId)

# Home
pb.MoveRobot(list(HOME_Q_DEG), degree=True)
time.sleep(1)

# 공 위치
cue_pos = np.array(env.get_cue_ball_position())
print(f"Cue pos: {cue_pos}")

# 단순 타격: 90° (위쪽), EE speed = 0.5 m/s
angle = np.radians(90)
strike_dir_2d = np.array([np.cos(angle), np.sin(angle)])
angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
strike_dir_3d = np.array([
    strike_dir_2d[0] * np.cos(angle_rad),
    strike_dir_2d[1] * np.cos(angle_rad),
    -np.sin(angle_rad)
])
strike_dir_3d /= np.linalg.norm(strike_dir_3d)

ee_speed = 0.5
q_home = np.array(HOME_Q_RAD)
T_home = pb.my_robot.pinModel.FK(q_home)

# 궤적 생성
trajectory, phase = planner.plan_strike(
    T_home, cue_pos, strike_dir_3d,
    strike_speed=ee_speed,
    approach_dist=STRIKE_APPROACH_DIST,
    follow_dist=STRIKE_FOLLOW_DIST,
    tool_offset=TOOL_HEAD_LENGTH,
    tool_rotation=np.radians(90)
)

# 접근
q_i = q_home.copy()
for i in range(phase['approach'][0], phase['approach'][1]):
    q_i = ik.solve_step(q_i, trajectory[i])
    pb.MoveRobot(q_i, degree=False)
    time.sleep(TRAJECTORY_DT)

time.sleep(0.5)

# 타격 전 공 속도
v_before, _ = p.getBaseVelocity(env.cue_ball_id, physicsClientId=pb.ClientId)
print(f"Before strike: v_cue = {np.linalg.norm(v_before[:2]):.4f} m/s")

# 임팩트
strike_end = phase['strike'][1]
T_impact = trajectory[strike_end - 1]
q_impact = ik.solve_step(q_i, T_impact)
pb.MoveRobot(q_impact, degree=False)
time.sleep(0.3)

# 타격 후 공 속도
v_after, _ = p.getBaseVelocity(env.cue_ball_id, physicsClientId=pb.ClientId)
speed_after = np.linalg.norm(v_after[:2])
print(f"After strike:  v_cue = {speed_after:.4f} m/s")
print(f"  v_cue vector: [{v_after[0]:.4f}, {v_after[1]:.4f}, {v_after[2]:.4f}]")
print(f"  EE speed:     {ee_speed} m/s")
print(f"  공속도/EE속도 = {speed_after/ee_speed:.2f}x")

# 이론값 비교
e_eff = np.sqrt(TOOL_HEAD_RESTITUTION * MAZE_BALL_RESTITUTION)
ratio_05 = (1+e_eff) * TOOL_HEAD_MASS / (TOOL_HEAD_MASS + MAZE_BALL_MASS)
print(f"\n이론(0.5kg 도구): {ratio_05*ee_speed:.4f} m/s ({ratio_05:.2f}x)")
print(f"이론(무한 도구): {(1+e_eff)*ee_speed:.4f} m/s ({(1+e_eff):.2f}x)")
print(f"실측 비율 {speed_after/ee_speed:.2f}x → 유효질량 추정: {MAZE_BALL_MASS * speed_after / ((1+e_eff)*ee_speed - speed_after):.2f} kg")

pb.disconnect()
