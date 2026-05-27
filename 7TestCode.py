# -*- coding: utf-8 -*-
# %% [markdown]
# # 6. Real Robot Test -- 시뮬 궤적을 실제 로봇이 따라하는지 확인
#
# **흐름:**
# 1. 시뮬(GUI)에서 전체 라운드 실행 -> 타격 결과 관찰
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
# -- 타이밍/전환 상수 --
MOVEL_MIN_DIST_MM      = 3.0  # movel 최소 이동 거리 (mm)


def sync_indy():
    q = indy.get_control_data()['q']
    pb.MoveRobot(q, degree=True)

def wait_indy(timeout=30):
    """실제 로봇 동작 완료 대기 (timeout 포함)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
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


def verify_movel_reached(p_target, tolerance_mm=5.0):
    """movel 목표 도달 여부 검증 -- 안 됐으면 재시도"""
    p_now = indy.get_control_data()['p']
    err_mm = np.linalg.norm(np.array(p_now[:3]) - np.array(p_target[:3]))
    if err_mm > tolerance_mm:
        print(f"  [WARNING] movel 미도달: err={err_mm:.1f}mm > {tolerance_mm}mm, 재시도...")
        indy.movel(list(p_target), vel_ratio=50, acc_ratio=100)
        wait_indy()
        return False
    return True


def SE3_to_p6(T):
    """SE3(4x4) -> [x_mm, y_mm, z_mm, rx, ry, rz] (Euler XYZ deg)"""
    p = np.zeros(6)
    p[0:3] = 1000 * T[0:3, 3]
    p[3:6] = Rot2eul(T[0:3, 0:3], seq='XYZ', degree=True)
    return p


def replay_trajectory_on_real(q_traj_deg, q_follow_deg, phases, label="", strike_speed=1.0):
    """전구간 MoveJ -- 시뮬 IK 관절각도를 그대로 전송

    Euler 변환 없음, FK 오프셋 없음, 좌표계 불일치 없음.
    Phase 1 (Approach):  waypoint별 MoveJ
    Phase 1.5 (Align):   Ready 위치 정밀 정렬 MoveJ
    Phase 2 (Strike):    Follow-through MoveJ
    Phase 3 (Retract):   Home MoveJ
    """
    approach_start = phases['approach'][0]
    approach_end = phases['approach'][1]

    # ======== Phase 1: Approach (MoveJ) ========
    APPROACH_STEP = 100
    APPROACH_VEL = 20
    APPROACH_ACC = 50

    waypoint_indices = list(range(approach_start, approach_end, APPROACH_STEP))
    if waypoint_indices[-1] != approach_end - 1:
        waypoint_indices.append(approach_end - 1)

    print(f"  [{label}] Phase 1: MoveJ Approach ({len(waypoint_indices)} waypoints)...")
    for wi, idx in enumerate(waypoint_indices):
        indy.movej([float(x) for x in q_traj_deg[idx]], vel_ratio=APPROACH_VEL, acc_ratio=APPROACH_ACC)
        wait_indy()
        if wi % 5 == 0:
            print(f"    waypoint {wi+1}/{len(waypoint_indices)}")
    print(f"  [{label}] Approach 완료")

    # ======== Phase 1.5: Align ========
    q_ready = q_traj_deg[approach_end - 1]
    print(f"  [{label}] Phase 1.5: Align")
    time.sleep(0.5)
    indy.movej([float(x) for x in q_ready], vel_ratio=10, acc_ratio=30)
    wait_indy()
    time.sleep(0.5)

    # 진단
    p_now = indy.get_control_data()['p']
    print(f"    Ready pos: [{p_now[0]:.1f}, {p_now[1]:.1f}, {p_now[2]:.1f}] mm")
    print(f"    Ready eul: [{p_now[3]:.1f}, {p_now[4]:.1f}, {p_now[5]:.1f}] deg")
    print(f"  [{label}] Ready 정렬 완료")

    # ======== Phase 2: Strike (MoveJ to follow-through) ========
    print(f"  [{label}] Phase 2: MoveJ Strike!")
    indy.movej([float(x) for x in q_follow_deg], vel_ratio=100, acc_ratio=300)
    wait_indy()
    p_after = indy.get_control_data()['p']
    print(f"    After strike: [{p_after[0]:.1f}, {p_after[1]:.1f}, {p_after[2]:.1f}] mm")
    print(f"  [{label}] Strike 완료!")

    # ======== Phase 3: Home ========
    print(f"  [{label}] Phase 3: Home")
    indy.movej(HOME_Q_DEG, vel_ratio=30, acc_ratio=100)
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
    from project.real_env_to_pybullet import detect_balls
    env = MazeEnvironment(pb.ClientId)
    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH/2 + MAZE_BALL_RADIUS + 0.001
    cue_pos, target_pos, ball2_pos = detect_balls()
    env.setup(
        cue_pos=cue_pos,
        target_pos=target_pos,
        ball2_pos=ball2_pos,
        num_obstacles=0  # 순수 쓰리쿠션 (장애물 없음)
    )
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    tool_offset = MAZE_BALL_RADIUS  # 큐팁이 공 표면에 닿도록 (ㄴ자 오프셋은 planner 내부 처리)
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

# %% Step 9: *** 시뮬에서 전체 라운드 실행 (GUI로 관찰) ***
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
        if np.linalg.norm(ball_pos[:2]) > 0.80:
            print(f"  큐볼 범위 밖 -> 리셋")
            env.reset_balls(cue_pos=env.cue_start_pos)
            time.sleep(0.5)
            ball_pos = env.get_cue_ball_position()
    else:
        ball_pos = env.get_cue_ball_position()
        target_pos = env.get_target_ball_position()
        print(f"  큐볼: {ball_pos[:2]}, 목표공: {target_pos[:2]}")
        if np.linalg.norm(ball_pos[:2]) > 0.80:
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
    # 이전 라운드에서 비활성화된 도구-큐볼 충돌 재활성화
    if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
        p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                 enableCollision=1, physicsClientId=pb.ClientId)
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

    # * 임팩트 후 도구-큐볼 충돌 즉시 비활성화 *
    # Headless planner는 충돌 후 도구를 제거하므로, GUI에서도 동일하게
    # 도구가 공 궤적에 간섭하지 않도록 비활성화
    if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
        p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                 enableCollision=0, physicsClientId=pb.ClientId)

    # 수직 상승 후퇴 (공/장애물 회피)
    T_lift = trajectory[-1].copy()
    T_lift[2, 3] += RETRACT_HEIGHT
    q_lift = ik.solve_step(q_follow, T_lift)
    pb.MoveRobot(q_lift, degree=False)
    time.sleep(0.3)

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

    # 관절각도 저장 (접근 + follow-through), 홈 복귀
    q_traj_deg = np.degrees(np.array(q_traj).reshape(-1, 6))
    q_follow_deg = np.degrees(np.array(q_follow).flatten())
    saved_trajectories.append((q_traj_deg, q_follow_deg, phases))
    pb.MoveRobot(HOME_Q_DEG, degree=True)
    time.sleep(1)

print(f"\n{'='*50}")
print(f"  SIM 완료! 궤적 {len(saved_trajectories)}개 저장됨")
print(f"  다음 셀에서 실제 로봇 재생")
print(f"{'='*50}")

# %% Step 10: *** 실제 로봇 -- 라운드 재생 (루프) ***
# [!] E-Stop 버튼에 손 올리고 실행!
for rnd_idx, (q_traj_d, q_follow_d, phs) in enumerate(saved_trajectories):
    rnd_num = rnd_idx + 1
    print("=" * 50)
    print(f"  REAL Round {rnd_num}/{len(saved_trajectories)} 재생")
    print("=" * 50)
    movej_both(HOME_Q_DEG, wait=True)
    time.sleep(1)
    try:
        replay_trajectory_on_real(q_traj_d, q_follow_d, phs, label=f"Round {rnd_num}")
    except Exception as e:
        print(f"  [!] Round {rnd_num} 오류: {e}")
        try:
            indy.recover()
        except:
            pass
        time.sleep(1)
    movej_both(HOME_Q_DEG, wait=True)
    print(f"Round {rnd_num} 완료!")

# %% Step 11: 정리
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
# | movel 타격이 안됨 | teleop->task 전환 대기 1.5초 확인 |
# | movel 속도 부족 | STRIKE_APPROACH_DIST 증가 (가속 거리 확보) |
