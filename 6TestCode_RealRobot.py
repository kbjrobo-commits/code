# %% [markdown]
# # 6. Real Robot Test — 시뮬 궤적을 실제 로봇이 따라하는지 확인
#
# **흐름:**
# 1. 시뮬(GUI)에서 전체 라운드 실행 → 타격 결과 관찰
# 2. 저장된 궤적을 실제 로봇이 라운드별로 재생 (타격 포함)
#
# **미니골프: 1라운드 / 빌리아드: 3라운드**

# %% Step 1: Import
import os, sys, time
import numpy as np
sys.path.append('.')

from src.utils import *
from src.core.pybullet_core import PybulletCore
from project.config import *
from project.ik_solver import IKSolver
from project.trajectory_planner import StrikeTrajectoryPlanner
print("Import 완료")

# %% Step 2: 로봇 IP (배정된 번호로 수정)
ROBOT_IP = '192.168.0.13'

# %% Step 3: PyBullet 시뮬레이터 연결
pb = PybulletCore()
pb.connect(robot_name="indy7_v2", joint_limit=True, constraint_visualization=False)
ik = IKSolver(pb.my_robot.pinModel, gain=IK_GAIN, damping=IK_DAMPING)
print(f"시뮬 연결 완료")

# %% Step 4: 실제 로봇 연결
from neuromeka import IndyDCP3
indy = IndyDCP3(robot_ip=ROBOT_IP, index=0)
print(f"실제 로봇 연결: {ROBOT_IP}")
print(f"현재 q(deg): {[round(x,1) for x in indy.get_control_data()['q']]}")

# %% Step 5: 헬퍼 함수
def sync_indy():
    q = indy.get_control_data()['q']
    pb.MoveRobot(q, degree=True)

def wait_indy():
    while True:
        sync_indy()
        if not indy.get_motion_data()["is_in_motion"]:
            break
        time.sleep(0.01)
    print("  모션 완료")

def movej_both(q_deg, wait=True):
    indy.movej(list(np.asarray(q_deg).flatten()))
    pb.MoveRobot(q_deg, degree=True)
    if wait:
        wait_indy()

def replay_trajectory_on_real(traj_SE3, phases=None, label="", strike_speed=1.0):
    """SE3 궤적을 실제 로봇에서 재생
    
    Phase 1 (Approach): Teleop으로 천천히 준비 위치까지 이동
    Phase 2 (Strike):   Teleop 종료 → 단일 MoveL로 Follow-through 끝점까지 풀스윙
    Phase 3 (Retract):  수직 상승 후 Home 복귀
    
    이유: Teleop은 프레임 단위 스트리밍이라 로봇 내부 안전 제어기가
    가속을 억제하여 '밀어치기'가 됨. MoveL은 출발~도착을 한 번에 주므로
    로봇이 자체 가속 프로파일을 그려 원하는 속도까지 도달 가능.
    """
    if phases is None:
        phases = {'approach': (0, len(traj_SE3)),
                  'strike': (len(traj_SE3), len(traj_SE3)),
                  'follow': (len(traj_SE3), len(traj_SE3))}

    approach_end = phases['approach'][1]
    follow_end = phases['follow'][1] if phases['follow'][1] > 0 else len(traj_SE3)
    
    # ======== Phase 1: Approach (Teleop — 천천히) ========
    print(f"  [{label}] Phase 1: Teleop Approach ({approach_end} pts)...")
    dT = 0.002
    indy.start_teleop(0)
    time.sleep(1)
    
    for idx in range(0, approach_end):
        if idx % 50 != 0 and idx != approach_end - 1:
            continue
        T_des = traj_SE3[idx]
        p_des = np.zeros(6)
        p_des[0:3] = 1000 * T_des[0:3, 3]
        p_des[3:6] = Rot2eul(T_des[0:3, 0:3], seq='XYZ', degree=True)
        indy.movetelel_abs(p_des, vel_ratio=0.3, acc_ratio=1)
        time.sleep(dT * 50)
    
    wait_indy()
    indy.stop_teleop()
    print(f"  [{label}] Approach 완료 — 준비 위치 도달")
    time.sleep(0.3)  # 안정화 대기
    
    # ======== Phase 2: Strike (단일 MoveL — 풀스윙) ========
    # Follow-through 끝점을 타겟으로 단일 명령 전송
    T_follow_end = traj_SE3[min(follow_end - 1, len(traj_SE3) - 1)]
    p_target = np.zeros(6)
    p_target[0:3] = 1000 * T_follow_end[0:3, 3]
    p_target[3:6] = Rot2eul(T_follow_end[0:3, 0:3], seq='XYZ', degree=True)
    
    # vel_ratio: strike_speed를 로봇 최대 속도(~1m/s)에 대한 비율로 변환
    # movel의 vel_ratio 범위는 0~100
    vel_pct = np.clip(strike_speed / MAX_TOOL_SPEED * 100, 10, 100)
    print(f"  [{label}] Phase 2: MoveL Strike! vel={vel_pct:.0f}%, acc=100%")
    print(f"    Target: [{p_target[0]:.1f}, {p_target[1]:.1f}, {p_target[2]:.1f}] mm")
    
    indy.movel(p_target, vel_ratio=vel_pct, acc_ratio=100)
    wait_indy()
    print(f"  [{label}] Strike 완료!")
    
    # ======== Phase 3: 수직 상승 후퇴 ========
    T_lift = T_follow_end.copy()
    T_lift[2, 3] += 0.15  # 15cm 상승
    p_lift = np.zeros(6)
    p_lift[0:3] = 1000 * T_lift[0:3, 3]
    p_lift[3:6] = Rot2eul(T_lift[0:3, 0:3], seq='XYZ', degree=True)
    
    print(f"  [{label}] Phase 3: Vertical lift + Home")
    indy.movel(p_lift, vel_ratio=30, acc_ratio=100)
    wait_indy()
    print(f"  [{label}] 전체 완료")

print("헬퍼 함수 정의 완료")

# %% Step 6: 홈 이동 (양쪽 동기화) + FK 비교
movej_both(HOME_Q_DEG, wait=True)

q_rad = np.array(HOME_Q_DEG) * np.pi / 180
T_pin = pb.my_robot.pinModel.FK(q_rad)
p_real = indy.get_control_data()['p']
print(f"Pinocchio EE: {T_pin[:3,3]*1000} mm")
print(f"실제 로봇 EE: {p_real[:3]} mm")
print(f"오차: {np.linalg.norm(T_pin[:3,3]*1000 - np.array(p_real[:3])):.1f} mm")

# %% Step 7: 데모 선택 + 라운드 수
DEMO_TYPE = 'maze'   # 'minigolf', 'billiards', 또는 'maze' (3-cushion)
NUM_ROUNDS = 3

print(f"데모: {DEMO_TYPE}, 라운드: {NUM_ROUNDS}")

# %% Step 8: 시뮬 환경 세팅 + PD 게인 강화
import pybullet as p
robot_id = pb.my_robot.robotId
ee_link = pb.my_robot.RobotEEJointIdx[-1]

if DEMO_TYPE == 'minigolf':
    from project.environment.minigolf_env import MiniGolfEnvironment
    from project.physics.shot_planner import MinigolfShotPlanner
    env = MiniGolfEnvironment(pb.ClientId)
    env.setup(ball_pos=[0.45, -0.10, 0.035], hole_pos=[0.55, 0.12, 0.005])
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    tool_offset = TOOL_HEAD_LENGTH + MINIGOLF_BALL_RADIUS
    shot_planner = MinigolfShotPlanner()
    perception = None
elif DEMO_TYPE == 'maze':
    from project.environment.maze_env import MazeEnvironment
    from project.physics.cushion_planner import CushionShotPlanner
    from project.perception import SimPerception
    env = MazeEnvironment(pb.ClientId)
    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH/2 + MAZE_BALL_RADIUS + 0.001
    env.setup(
        cue_pos=[0.5, CY-W/4, ball_h],
        target_pos=[0.5, CY+W/8, ball_h],
        num_obstacles=0  # 순수 쓰리쿠션 (장애물 없음)
    )
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS
    shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
    perception = SimPerception(env)
else:  # billiards
    from project.environment.billiards_env import BilliardsEnvironment
    from project.physics.shot_planner import BilliardsShotPlanner
    env = BilliardsEnvironment(pb.ClientId)
    CY, W = BILLIARD_TABLE_CENTER_Y, BILLIARD_TABLE_WIDTH
    H, TH = BILLIARD_TABLE_SURFACE_HEIGHT, BILLIARD_TABLE_HEIGHT
    ball_h = H + TH/2 + BILLIARD_BALL_RADIUS + 0.001
    env.setup(cue_pos=[0.5, CY-W/4, ball_h], target_pos=[0.5, CY+W/8, ball_h])
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    tool_offset = TOOL_HEAD_LENGTH + BILLIARD_BALL_RADIUS
    shot_planner = BilliardsShotPlanner()
    perception = None

# PD 게인 강화
robot = pb.my_robot
def _boosted():
    qddot = robot._qddot_des + 800*(robot._q_des-robot._q) + 40*(robot._qdot_des-robot._qdot)
    robot._tau = robot._M @ qddot + robot._c + robot._g
robot._compute_torque_input = _boosted
time.sleep(3)
print(f"환경 세팅 완료 (도구 안정화 3초 대기)")

# %% Step 9: ★★★ 시뮬에서 전체 라운드 실행 (GUI로 관찰) ★★★
traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)
saved_trajectories = []

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*50}")
    print(f"  SIM Round {rnd}/{NUM_ROUNDS}")
    print(f"{'='*50}")

    # --- SCAN ---
    if DEMO_TYPE == 'minigolf':
        ball_pos = env.get_ball_position()
        print(f"  공: {ball_pos}, 홀: {env.hole_pos}")
    elif DEMO_TYPE == 'maze':
        scan = perception.scan_environment()
        ball_pos = scan['cue_pos']
        target_pos = scan['target_pos']
        ball2_pos = scan.get('ball2_pos')
        obstacles = scan.get('obstacles', [])
        print(f"  큐볼: {ball_pos[:2]}, 목표1: {target_pos[:2]}, 목표2: {ball2_pos[:2] if ball2_pos is not None else 'N/A'}")
        if np.linalg.norm(ball_pos[:2]) > 0.70:
            print(f"  큐볼 범위 밖 -> 리셋")
            env.reset_balls(cue_pos=env.cue_start_pos)
            time.sleep(0.5)
            ball_pos = env.get_cue_ball_position()
    else:
        ball_pos = env.get_cue_ball_position()
        target_pos = env.get_target_ball_position()
        print(f"  큐볼: {ball_pos[:2]}, 목표공: {target_pos[:2]}")
        if np.linalg.norm(ball_pos[:2]) > 0.70:
            print(f"  큐볼 범위 밖 -> 리셋")
            env.reset_balls(cue_pos=env.cue_start_pos)
            time.sleep(0.5)
            ball_pos = env.get_cue_ball_position()

    # --- THINK ---
    if DEMO_TYPE == 'minigolf':
        terrain_path = getattr(env, 'terrain_obj_path', None)
        if terrain_path:
            print(f"  Grid Search 실행 중...")
            strike_dir, speed = shot_planner.plan_shot_physics_search(
                ball_pos, env.hole_pos, terrain_path, env.terrain_offset)
        else:
            strike_dir, speed = shot_planner.plan_shot(ball_pos, env.hole_pos)
        strike_dir_3d = np.array(strike_dir).flatten()
    elif DEMO_TYPE == 'maze':
        print(f"  Headless 3D 탐색 중...")
        candidates = shot_planner.plan_shot(ball_pos, target_pos, obstacles, ball2_pos=ball2_pos)
        best = candidates[0]
        strike_dir_2d = best['strike_dir']
        speed = best['strike_speed']
        angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
        horiz = np.array(strike_dir_2d[:2]).flatten()
        horiz = horiz / np.linalg.norm(horiz)
        strike_dir_3d = np.array([
            horiz[0]*np.cos(angle_rad), horiz[1]*np.cos(angle_rad), -np.sin(angle_rad)])
        strike_dir_3d /= np.linalg.norm(strike_dir_3d)
        print(f"  최적: score={best['score']:.0f}, cushions={best['cushion_count']}, "
              f"hit_t1={best['hit_t1']}, hit_t2={best['hit_t2']}")
    else:
        result = shot_planner.find_best_pocket_shot(
            ball_pos, target_pos, env.pocket_positions)
        strike_dir, speed = result['strike_dir'], result['strike_speed']
        angle_rad = np.radians(BILLIARD_STRIKE_ANGLE_DEG)
        horiz = np.array(strike_dir[:2]).flatten()
        horiz = horiz / np.linalg.norm(horiz)
        strike_dir_3d = np.array([
            horiz[0]*np.cos(angle_rad), horiz[1]*np.cos(angle_rad), -np.sin(angle_rad)])
        strike_dir_3d /= np.linalg.norm(strike_dir_3d)

    print(f"  방향: {strike_dir_3d}, 속도: {speed:.3f} m/s")

    # --- 궤적 생성 ---
    T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
    q_now = pb.my_robot.q.copy()

    trajectory, phases = traj_planner.plan_strike(
        T_current=T_now, ball_pos=ball_pos, strike_direction=strike_dir_3d,
        strike_speed=speed, approach_dist=STRIKE_APPROACH_DIST,
        follow_dist=STRIKE_FOLLOW_DIST,
        strike_height=ball_pos[2], tool_offset=tool_offset)

    q_traj = ik.solve_trajectory(q_now, trajectory)
    print(f"  궤적: {len(trajectory)} pts (A:{phases['approach'][1]-phases['approach'][0]}"
          f" S:{phases['strike'][1]-phases['strike'][0]}"
          f" F:{phases['follow'][1]-phases['follow'][0]})")

    # --- SIM 실행: Approach ---
    print(f"  [SIM] Approach...")
    for i in range(phases['approach'][0], phases['approach'][1]):
        pb.MoveRobot(q_traj[i], degree=False)
        time.sleep(0.002)
    q_ready = q_traj[phases['approach'][1] - 1].copy()
    time.sleep(0.5)

    # --- SIM 실행: Strike (swing-through) ---
    print(f"  [SIM] Strike!")
    q_follow = ik.solve_step(q_ready, trajectory[-1])
    swing_t = np.linalg.norm(trajectory[-1][:3,3] - trajectory[phases['approach'][1]-1][:3,3]) / (speed * 0.7)
    swing_t = np.clip(swing_t, 0.05, 0.8)
    avg_qdot = (q_follow - q_ready) / swing_t
    if hasattr(pb.my_robot, '_qdot_des'):
        pb.my_robot._qdot_des = avg_qdot
    pb.MoveRobot(q_follow, degree=False)
    time.sleep(swing_t)
    if hasattr(pb.my_robot, '_qdot_des'):
        pb.my_robot._qdot_des = np.zeros([6, 1])
    # 수직 상승 후퇴 (공/장애물 회피)
    T_lift = trajectory[-1].copy()
    T_lift[2, 3] += 0.15  # 15cm 상승
    q_lift = ik.solve_step(q_follow, T_lift)
    pb.MoveRobot(q_lift, degree=False)
    time.sleep(0.8)

    # --- OBSERVE ---
    if DEMO_TYPE == 'minigolf':
        env.wait_ball_stop(timeout=5.0)
        dist = env.get_distance_to_hole()
        success = env.is_hole_in()
        print(f"  결과: 거리={dist:.4f}m {'HOLE-IN-ONE!' if success else ''}")
    elif DEMO_TYPE == 'maze':
        env.wait_balls_stop(timeout=5.0)
        cue_pos_f = env.get_cue_ball_position()
        t1_pos = env.get_target_ball_position()
        d1 = np.linalg.norm(cue_pos_f[:2] - t1_pos[:2])
        d2 = 0
        if hasattr(env, 'ball2_id'):
            t2_pos = env.get_ball2_position()
            d2 = np.linalg.norm(cue_pos_f[:2] - t2_pos[:2])
        print(f"  결과: d(tgt1)={d1:.3f}m, d(tgt2)={d2:.3f}m")
        success = False
    else:
        env.wait_balls_stop(timeout=5.0)
        success = env.is_pocketed()
        print(f"  결과: {'POCKETED!' if success else 'miss'}")

    # 궤적 + phases 저장, 홈 복귀
    saved_trajectories.append((trajectory, phases))
    pb.MoveRobot(HOME_Q_DEG, degree=True)
    time.sleep(1)

print(f"\n{'='*50}")
print(f"  SIM 완료! 궤적 {len(saved_trajectories)}개 저장됨")
print(f"  다음 셀에서 실제 로봇 재생")
print(f"{'='*50}")

# %% Step 10: ★★★ 실제 로봇 — Round 1 재생 ★★★
# ⚠️ E-Stop 버튼에 손 올리고 실행!
print("=" * 50)
print("  REAL Round 1 재생")
print("=" * 50)
movej_both(HOME_Q_DEG, wait=True)
time.sleep(1)
traj_0, phases_0 = saved_trajectories[0]
replay_trajectory_on_real(traj_0, phases=phases_0, label="Round 1")
movej_both(HOME_Q_DEG, wait=True)
print("Round 1 완료!")

# %% Step 11: 실제 로봇 — Round 2 재생 (billiards)
if len(saved_trajectories) >= 2:
    print("=" * 50)
    print("  REAL Round 2 재생")
    print("=" * 50)
    movej_both(HOME_Q_DEG, wait=True)
    time.sleep(1)
    traj_1, phases_1 = saved_trajectories[1]
    replay_trajectory_on_real(traj_1, phases=phases_1, label="Round 2")
    movej_both(HOME_Q_DEG, wait=True)
    print("Round 2 완료!")
else:
    print("Round 2 없음 (미니골프는 1라운드)")

# %% Step 12: 실제 로봇 — Round 3 재생 (billiards)
if len(saved_trajectories) >= 3:
    print("=" * 50)
    print("  REAL Round 3 재생")
    print("=" * 50)
    movej_both(HOME_Q_DEG, wait=True)
    time.sleep(1)
    traj_2, phases_2 = saved_trajectories[2]
    replay_trajectory_on_real(traj_2, phases=phases_2, label="Round 3")
    movej_both(HOME_Q_DEG, wait=True)
    print("Round 3 완료!")
else:
    print("Round 3 없음")

# %% Step 13: 정리
if hasattr(env, 'cleanup'):
    env.cleanup()
pb.disconnect()
print("완료! 시뮬 연결 해제됨")

# %% [markdown]
# ## 트러블슈팅
# | 문제 | 해결 |
# |------|------|
# | Standstill Failed | `indy.recover()` |
# | 로봇 안 움직임 | Conty에서 서보 ON 확인 |
# | neuromeka import 실패 | `pip install neuromeka` |
