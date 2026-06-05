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
    """실제 로봇 재생: Approach/Align=MoveJ, Strike=MoveL 직선.

    Phase 1 (Approach):  waypoint별 MoveJ
    Phase 1.5 (Align):   Ready 위치 정밀 정렬 MoveJ
    --- [Enter] 대기 (movej↔movel 분리) ---
    Phase 2 (Strike):    MoveL 직선 타격 (acc=600), 실패 시 movej fallback
    --- [Enter] 대기 (movel↔movej 분리) ---
    Phase 3 (Home):      MoveJ 홈 복귀
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

    # ======== Phase 1.5: Align (MoveL) ========
    # MoveL로 TCP 위치+방향 정밀 정렬 → 캐리브레이션 오프셋 유지 + 특이점 회피
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

    # ======== Phase 2: Strike (MoveL 직선) ========

    # 시뮬 FK: ready→follow 변위 계산
    pin = pb.my_robot.pinModel
    T_ready = pin.FK(np.radians(q_ready))
    T_follow = pin.FK(np.radians(q_follow_deg))
    delta_mm = (T_follow[:3, 3] - T_ready[:3, 3]) * 1000.0
    dist_mm = float(np.linalg.norm(delta_mm))

    print(f"\n  {'='*56}")
    print(f"  [{label}] APPROACH 완료 — 로봇 정지")
    print(f"  delta: [{delta_mm[0]:.1f}, {delta_mm[1]:.1f}, {delta_mm[2]:.1f}] mm ({dist_mm:.1f}mm)")
    print(f"  >>> [Enter] = START → MoveL 직선 STRIKE")
    print(f"  {'='*56}")
    input()

    p_ready = indy.get_control_data()['p']  # [x,y,z,rx,ry,rz] mm/deg
    print(f"    Ready TCP: [{p_ready[0]:.1f}, {p_ready[1]:.1f}, {p_ready[2]:.1f}] mm")

    # movel 목표 = 현재 TCP + delta (자세 유지)
    p_target = list(p_ready)
    p_target[0] += delta_mm[0]
    p_target[1] += delta_mm[1]
    p_target[2] += delta_mm[2]

    print(f"  [{label}] Phase 2: MoveL Strike!")
    print(f"    target: [{p_target[0]:.1f}, {p_target[1]:.1f}, {p_target[2]:.1f}] mm")

    if dist_mm < 3.0:
        print(f"    [WARN] 거리 {dist_mm:.1f}mm < 3mm → movej fallback")
        indy.movej([float(x) for x in q_follow_deg], vel_ratio=100, acc_ratio=600)
        wait_indy()
    else:
        indy.movel([float(x) for x in p_target],
                    vel_ratio=100, acc_ratio=600)
        time.sleep(0.2)
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if indy.get_motion_data()['is_in_motion']:
                break
            time.sleep(0.05)
        while time.time() - t0 < 30.0:
            if not indy.get_motion_data()['is_in_motion']:
                break
            time.sleep(0.05)
        p_after = indy.get_control_data()['p']
        moved = float(np.linalg.norm(np.array(p_after[:3]) - np.array(p_ready[:3])))
        print(f"    이동량: {moved:.1f} mm")
        if moved < 3.0:
            print(f"    [WARN] movel 미동작 → movej fallback")
            indy.movej([float(x) for x in q_follow_deg], vel_ratio=100, acc_ratio=600)
            wait_indy()
    print(f"  [{label}] Strike 완료!")

    # ======== Phase 3: Home ========
    input(f"\n  >>> [Enter] → Home 복귀\n")
    print(f"  [{label}] Phase 3: Home")
    indy.movej(HOME_Q_DEG, vel_ratio=30, acc_ratio=100)
    wait_indy()
    print(f"  [{label}] 전체 완료")

def cue_ball_is_near_cushion(cue_pos) :
    CX = MAZE_TABLE_CENTER_X
    CY = MAZE_TABLE_CENTER_Y
    L = MAZE_TABLE_LENGTH
    W = MAZE_TABLE_WIDTH
    x = cue_pos[0]
    y = cue_pos[1]

    if abs(x - (CX + L/2)) < ESCAPE_WALL_GAP_THRESHOLD:
        return True
    elif abs(x - (CX - L/2)) < ESCAPE_WALL_GAP_THRESHOLD:
        return True
    
    if abs(y - (CY + W/2)) < ESCAPE_WALL_GAP_THRESHOLD:
        return True
    elif abs(y - (CY - W/2)) < ESCAPE_WALL_GAP_THRESHOLD:
        return True

    return False

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
# 'minigolf', 'billiards', 'maze' (3-cushion), 'pocket_phase1', 'pocket_phase2'
DEMO_TYPE = 'pocket_phase1'
NUM_ROUNDS = 1  # pocket_phase1=3(공 3개), pocket_phase2=1(트릭샷 1회)

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
    from project.real_env_to_pybullet import detect_balls, load_position_offset
    _pos_offset = load_position_offset()
    env = MazeEnvironment(pb.ClientId)
    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH/2 + MAZE_BALL_RADIUS + 0.001
    cue_pos, target_pos, ball2_pos = detect_balls()
    env.setup(
        cue_pos=cue_pos,
        target_pos=target_pos,
        ball2_pos=ball2_pos,
        num_obstacles=0,  # 순수 쓰리쿠션 (장애물 없음)
        position_offset=_pos_offset,
    )
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    tool_offset = MAZE_BALL_RADIUS  # 큐팁이 공 표면에 닿도록 (ㄴ자 오프셋은 planner 내부 처리)
    shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
    perception = SimPerception(env)
elif DEMO_TYPE in ('pocket_phase1', 'pocket_phase2'):
    from project.environment.maze_env import MazeEnvironment
    from project.physics.pocket_planner import PocketShotPlanner
    from project.real_env_to_pybullet import detect_balls, load_position_offset
    _pos_offset = load_position_offset()
    env = MazeEnvironment(pb.ClientId)
    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH/2 + MAZE_BALL_RADIUS + 0.001

    if DEMO_TYPE == 'pocket_phase2':
        # Phase 2: 비전으로 흰 큐볼 + trick ball 2개 위치 받기
        # detect_balls()는 (white, red, yellow) 반환
        print("  [VISION] 공 위치 감지 중... (큐볼=흰, trick1=노랑, trick2=빨강)")
        cue_pos, red_pos, yellow_pos, black_pos = detect_balls()
        print(f"    큐볼: {cue_pos[:2]}")
        print(f"    Trick1(노랑): {yellow_pos[:2]}")
        print(f"    Trick2(빨강): {red_pos[:2]}")
        print(f"    목적구3(검정): {black_pos[:2]}")

        # 초기 배치: 비전으로 받은 위치 사용
        env.setup(
            cue_pos=cue_pos,
            target_pos=yellow_pos,  # trick ball 1 = 노란공
            ball2_pos=red_pos,      # trick ball 2 = 빨간공
            ball3_pos=black_pos,    # trick ball 3 = 검은공
            num_obstacles=0,
            setup_pockets=True,
            position_offset=_pos_offset,
        )
        NUM_ROUNDS = 1
    else:
        # Phase 1: 비전으로 모든 공 위치 받기
        print("  [VISION] 공 위치 감지 중...")
        cue_pos, yellow_pos, red_pos, black_pos = detect_balls()
        print(f"    큐볼: {cue_pos[:2]}")
        print(f"    목적구1(노랑): {yellow_pos[:2]}")
        print(f"    목적구2(빨강): {red_pos[:2]}")
        print(f"    목적구3(검정): {black_pos[:2]}")

        env.setup(
            cue_pos=cue_pos,
            target_pos=yellow_pos,
            ball2_pos=red_pos,
            ball3_pos=black_pos,  # 3번째 공: 비전으로 감지된 위치
            num_obstacles=0,
            setup_pockets=True,
            position_offset=_pos_offset,
        )
        NUM_ROUNDS = 3

    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS
    shot_planner = PocketShotPlanner(table_bounds=env.table_bounds)
    perception = None
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
    qddot = robot._qddot_des + 5000*(robot._q_des-robot._q) + 200*(robot._qdot_des-robot._qdot)
    robot._tau = robot._M @ qddot + robot._c + robot._g
robot._compute_torque_input = _boosted
time.sleep(3)
print(f"환경 세팅 완료 (도구 안정화 3초 대기)")

# %% Step 9: *** 시뮬에서 전체 라운드 실행 (GUI로 관찰) ***
traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)
saved_trajectories = []

if DEMO_TYPE in ('pocket_phase1', 'pocket_phase2'):
    # ========================================================
    # Pocket 데모: Closed-Loop (감지→계획→실행→관측→반복)
    #
    # Phase 1: 공을 순서대로 포켓에 넣음
    #   매 타격마다: 비전 감지 → 시뮬 계획 → 실제 로봇 타격 → 비전 확인
    #   → 들어갔으면 다음 공, 안 들어갔으면 재감지 후 재계획
    #
    # Phase 2: 트릭샷 1회
    #   비전 감지 → 시뮬 계획 → 실제 로봇 타격 → 비전 결과 확인
    # ========================================================
    MAX_ATTEMPTS_PER_BALL = 3

    def _pocket_plan_and_traj(cue_p, target_p, strike_dir_2d, strike_speed, execute_sim=True):
        """시뮬 계획 → IK 궤적 → (q_traj_deg, q_follow_deg, phases) 반환.

        Args:
            execute_sim: True면 IK 통과 후 시뮬에서 실행, False면 IK 검증만
        """
        T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
        q_now = pb.my_robot.q.copy()
        angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
        horiz = np.array(strike_dir_2d[:2]).flatten()
        hn = np.linalg.norm(horiz)
        if hn > 1e-6:
            horiz = horiz / hn
        sd3d = np.array([horiz[0]*np.cos(angle_rad),
                         horiz[1]*np.cos(angle_rad),
                         -np.sin(angle_rad)])
        sd3d = sd3d / np.linalg.norm(sd3d)

        traj_c, ph_c = traj_planner.plan_strike(
            T_current=T_now, ball_pos=cue_p, strike_direction=sd3d,
            strike_speed=strike_speed,
            approach_dist=STRIKE_APPROACH_DIST,
            follow_dist=STRIKE_FOLLOW_DIST, strike_height=cue_p[2],
            tool_offset=tool_offset,
            table_bounds=env.table_bounds)

        approach_end = ph_c.get('approach', (0, 0))[1]
        val_from = int(approach_end * 0.65)
        ik_res = ik.solve_trajectory_validated(q_now, traj_c, validate_from=val_from,
                                                table_bounds=env.table_bounds)

        # IK 실패 + 특이점이 원인이면 3도 틸트로 재시도
        if not ik_res['valid']:
            has_singularity = any('특이점' in issue for issue in ik_res['issues'])
            if has_singularity:
                print(f"    [RETRY] 특이점 감지 -> 3deg 틸트로 재시도")
                traj_c, ph_c = traj_planner.plan_strike(
                    T_current=T_now, ball_pos=cue_p, strike_direction=sd3d,
                    strike_speed=strike_speed,
                    approach_dist=STRIKE_APPROACH_DIST,
                    follow_dist=STRIKE_FOLLOW_DIST, strike_height=cue_p[2],
                    tool_offset=tool_offset,
                    table_bounds=env.table_bounds,
                    singularity_tilt=np.radians(3.0))
                approach_end = ph_c.get('approach', (0, 0))[1]
                val_from = int(approach_end * 0.65)
                ik_res = ik.solve_trajectory_validated(q_now, traj_c, validate_from=val_from,
                                                        table_bounds=env.table_bounds)
                if ik_res['valid']:
                    print(f"    [RETRY] 틸트 재시도 성공!")

        if not ik_res['valid']:
            print(f"    [IK-DIAG] 실패 원인 ({len(ik_res['issues'])}건):")
            for issue in ik_res['issues'][:5]:
                print(f"      - {issue}")
            if len(ik_res['issues']) > 5:
                print(f"      ... +{len(ik_res['issues'])-5}건")
            print(f"    [IK-DIAG] min_manipulability={ik_res['min_manipulability']:.6f}, "
                  f"min_singularity_margin={ik_res.get('min_singularity_margin', -1):.4f}")
            return None

        q_traj_full = ik_res['q_trajectory']
        q_ready = q_traj_full[approach_end - 1].copy()
        q_follow = ik.solve_step(q_ready, traj_c[-1])

        # 시뮬에서 실행 (GUI 확인용) — execute_sim=False이면 건너뜀
        if execute_sim:
            for i in range(ph_c['approach'][0], ph_c['approach'][1]):
                pb.MoveRobot(q_traj_full[i], degree=False)
                time.sleep(0.002)
            time.sleep(0.3)
            sw_t = np.linalg.norm(traj_c[-1][:3,3] - traj_c[ph_c['approach'][1]-1][:3,3]) / (strike_speed * 0.7)
            sw_t = np.clip(sw_t, 0.05, 0.8)
            avg_qd = (q_follow - q_ready) / sw_t
            if hasattr(pb.my_robot, '_qdot_des'):
                pb.my_robot._qdot_des = avg_qd
            pb.MoveRobot(q_follow, degree=False)
            time.sleep(sw_t)
            if hasattr(pb.my_robot, '_qdot_des'):
                pb.my_robot._qdot_des = np.zeros([6, 1])

        q_traj_deg = np.degrees(np.array(q_traj_full).reshape(-1, 6))
        q_follow_deg = np.degrees(np.array(q_follow).flatten())
        return (q_traj_deg, q_follow_deg, ph_c)

    def _sim_execute_and_real_replay(traj_data, label):
        """시뮬 궤적 → 실제 로봇 재생 (Enter 확인 포함)."""
        q_traj_d, q_follow_d, ph = traj_data
        pb.MoveRobot(HOME_Q_DEG, degree=True)
        time.sleep(1)
        movej_both(HOME_Q_DEG, wait=True)
        time.sleep(1)
        replay_trajectory_on_real(q_traj_d, q_follow_d, ph, label=label)

    if DEMO_TYPE == 'pocket_phase1':
        # === Phase 1: 포켓볼 Closed-Loop ===
        print(f"\n{'='*50}")
        print(f"  POCKET PHASE 1: 포켓볼 (Closed-Loop)")
        print(f"{'='*50}")

        from project.real_env_to_pybullet import detect_balls
        ball_names = ['노란공', '빨간공', '검은공']
        balls_pocketed = [False, False, False]

        for ball_idx in range(3):
            if balls_pocketed[ball_idx]:
                continue

            for attempt in range(1, MAX_ATTEMPTS_PER_BALL + 1):
                print(f"\n  --- {ball_names[ball_idx]} (시도 {attempt}/{MAX_ATTEMPTS_PER_BALL}) ---")

                # 1) 비전으로 현재 공 위치 감지
                print(f"  [VISION] 공 위치 감지...")
                cue_pos, yellow_pos, red_pos, black_pos = detect_balls(balls_pocketed)
                print(f"    큐: {cue_pos[:2]}, 노: {yellow_pos[:2] if yellow_pos is not None else None}, 빨: {red_pos[:2] if red_pos is not None else None}, 검: {black_pos[:2] if black_pos is not None else None}")

                # 시뮬 환경에 비전 위치 반영
                env.reset_balls(cue_pos=cue_pos, target_pos=yellow_pos, ball2_pos=red_pos, ball3_pos=black_pos)
                # if yellow_pos is not None :
                #     p.resetBasePositionAndOrientation(
                #         env.target_ball_id, yellow_pos, [0,0,0,1], physicsClientId=pb.ClientId)
                # if red_pos is not None :
                #     p.resetBasePositionAndOrientation(
                #         env.ball2_id, red_pos, [0,0,0,1], physicsClientId=pb.ClientId)
                # if black_pos is not None :    
                #     p.resetBasePositionAndOrientation(
                #         env.ball3_id, black_pos, [0,0,0,1], physicsClientId=pb.ClientId)
                time.sleep(0.5)
                
                if cue_ball_is_near_cushion(cue_pos) is True :
                    CX = MAZE_TABLE_CENTER_X
                    CY = MAZE_TABLE_CENTER_Y
                    # escape니까 테이블 offset은 굳이 미적용
                    center = [CX, CY, cue_pos[2]]
                    strike_dir = center - cue_pos
                    strike_dir[2] = 0.0
                    strike_dir = strike_dir / (np.norm(strike_dir) + 1e-6)
                    cue_pos[2] += ESCAPE_STRIKE_HEIGHT_OFFSET
                    trajectory, phases = traj_planner.plan_strike(
                        T_current=pb.my_robot.pinModel.FK(pb.my_robot.q),
                        ball_pos=cue_pos,
                        strike_direction=strike_dir,
                        strike_speed=ESCAPE_BALL_SPEED,  # 매우 느리게
                        approach_dist=0.10,  # 5cm만 접근
                        follow_dist=0.04,
                    )
                    q_now = pb.my_robot.q.copy()
                    q_traj = ik.solve_trajectory(q_now, trajectory)
                    q_traj_deg = np.degrees(np.array(q_traj).reshape(-1, 6))

                    print("  [SIM] 시뮬 실행...")
                    # Approach
                    for i in range(phases['approach'][0], phases['approach'][1]):
                        pb.MoveRobot(q_traj[i], degree=False)
                        time.sleep(0.002)
                    q_ready = q_traj[phases['approach'][1] - 1].copy()
                    time.sleep(0.3)

                    # Strike (sim)
                    q_follow_idx = min(phases['follow'][1] - 1, len(q_traj) - 1)
                    q_follow = q_traj[q_follow_idx]
                    pb.MoveRobot(q_follow, degree=False)
                    time.sleep(0.5)

                    try:
                        # 실제 로봇도 진행 (strike 후 복귀까지)
                        print("  [REAL] 실제 로봇 재생...")
                        q_follow_deg = np.degrees(q_follow)
                        print("  실제 로봇 접근 중...")
                        replay_trajectory_on_real(q_traj_deg, q_follow_deg, phases, speed = 1.0)

                    except Exception as e:
                        print(f"  [ERROR] 실제 로봇 실행 실패: {e}")
                        try:
                            indy.recover()
                        except:
                            pass
                        movej_both(HOME_Q_DEG, wait=True)
                    
                    time.sleep(2)
                    continue

                # 타격 대상
                if ball_idx == 0:
                    target_pos = yellow_pos
                elif ball_idx == 2:
                    target_pos = black_pos
                else :
                    target_pos = red_pos


                # 2) 포켓 경로 계획
                print(f"  [PLAN] 포켓 경로 계획 중...")
                other_balls = []
                for oi in range(3):
                    if oi == ball_idx or balls_pocketed[oi]:
                        continue
                    ob = yellow_pos if oi == 0 else (red_pos if oi == 1 else black_pos)
                    other_balls.append(ob)

                candidates = shot_planner.plan_pocket_shot(
                    cue_pos, target_pos, other_balls
                )

                if not candidates:
                    print(f"  [FAIL] 포켓 경로 없음")
                    continue

                # 후보 순서대로 IK + 시뮬 시도
                traj_data = None
                for ci, cand in enumerate(candidates[:5]):
                    result = _pocket_plan_and_traj(
                        cue_pos, target_pos, cand['strike_dir'], cand['strike_speed'],
                        execute_sim=True)
                    if result:
                        print(f"  [IK-OK] 후보 #{ci+1}")
                        traj_data = result
                        break
                    else:
                        print(f"  [IK-FAIL] 후보 #{ci+1}")
                        pb.MoveRobot(HOME_Q_DEG, degree=True)
                        time.sleep(0.3)

                if traj_data is None:
                    print(f"  [FAIL] 모든 후보 IK 실패")
                    pb.MoveRobot(HOME_Q_DEG, degree=True)
                    time.sleep(1)
                    continue

                # 3) 실제 로봇 타격
                _sim_execute_and_real_replay(traj_data, f"{ball_names[ball_idx]} #{attempt}")

                # 4) 비전으로 결과 확인
                # detect_balls()는 3공 모두 감지될 때까지 무한 루프이므로,
                # 포켓된 공이 있으면 타임아웃됨 → 포켓 성공으로 판단
                print(f"  [OBSERVE] 결과 확인 중...")
                time.sleep(3)

                import threading
                detect_result = [None]  # [cue, red, yellow] or None


                # def _detect_with_timeout():
                #     try:
                #         detect_result[0] = detect_balls(balls_pocketed)
                #     except:
                #         pass

                # t = threading.Thread(target=_detect_with_timeout, daemon=True)
                # t.start()
                # t.join(timeout=10.0)  # 10초 타임아웃

                detect_result[0] = detect_balls(balls_pocketed)
                # detect_ball() : 10초이상 걸리면 타임 아웃 => None 반환
                # detect가 안되면 자동종료, detect가 되면 키입력 받은 후 종료

                if detect_result[0] is None:
                    print(f"  ★ {ball_names[ball_idx]} 포켓 성공! (카메라에서 미감지)")
                    balls_pocketed[ball_idx] = True
                    break
                else:
                    cue_f, yellow_f, red_f, black_f = detect_result[0]
                    print(f"    큐: {cue_f[:2]}, 노: {yellow_f[:2] if yellow_f is not None else None}, 빨: {red_f[:2] if red_f is not None else None}, 검: {black_f[:2] if black_f is not None else None}")
                    print(f"  ✗ {ball_names[ball_idx]} 아직 테이블 위 — 재시도")

                movej_both(HOME_Q_DEG, wait=True)
                time.sleep(1)

        pocketed_count = sum(balls_pocketed)
        print(f"\n  === PHASE 1 결과: {pocketed_count}/{len(ball_names)} 포켓 성공 ===")

    else:
        # === Phase 2: 트릭샷 ===
        print(f"\n{'='*50}")
        print(f"  POCKET PHASE 2: POSTECH 트릭샷 (Closed-Loop)")
        print(f"{'='*50}")

        # 1) 비전으로 공 위치 감지
        print(f"\n  [VISION] 공 위치 감지 (큐=흰, trick1=노랑, trick2=빨강)...")
        from project.real_env_to_pybullet import detect_balls
        cue_pos, red_pos, yellow_pos, _ = detect_balls()
        print(f"    큐볼: {cue_pos[:2]}")
        print(f"    Trick1(노랑): {yellow_pos[:2]}")
        print(f"    Trick2(빨강): {red_pos[:2]}")

        # 2) 시뮬 환경에 비전 위치 반영
        env.reset_balls(cue_pos=cue_pos, target_pos=yellow_pos, ball2_pos=red_pos)
        # p.resetBasePositionAndOrientation(
        #     env.target_ball_id, yellow_pos, [0,0,0,1], physicsClientId=pb.ClientId)
        # p.resetBasePositionAndOrientation(
        #     env.ball2_id, red_pos, [0,0,0,1], physicsClientId=pb.ClientId)
        time.sleep(0.5)

        # 3) POSTECH O 배치 (C형 4공은 시뮬에서 — 실제에선 이미 배치됨)
        meta = env.setup_postech_o()
        time.sleep(2)

        # 4) 트릭샷 계획
        print(f"\n  [PLAN] 트릭샷 탐색 중...")
        trick1_pos = np.array(meta['trick1_pos'])
        trick2_pos = np.array(meta['trick2_pos'])
        target1_goal = np.array(meta['target1_goal'])
        target2_goal = np.array(meta['target2_goal'])
        c_positions = [
            np.array(p.getBasePositionAndOrientation(
                bid, physicsClientId=pb.ClientId)[0])
            for bid in meta['c_ball_ids']
        ]
        candidates = shot_planner.plan_trick_shot(
            cue_pos, trick1_pos, trick2_pos,
            target1_goal, target2_goal, c_positions
        )

        if candidates:
            top = candidates[0]
            print(f"  [PLAN] Best: angle={top['angle_deg']:.1f}°, "
                  f"speed={top['ball_speed']:.2f}m/s, dist={top['total_dist']*100:.1f}cm")

            result = _pocket_plan_and_traj(
                cue_pos, trick1_pos, top['strike_dir'], top['strike_speed'])

            if result:
                print(f"  [SIM] 시뮬 확인 완료! 궤적 {len(result[0])} pts")
                saved_trajectories.append(result)

                # 5) 실제 로봇 타격
                _sim_execute_and_real_replay(result, "TrickShot")

                # 6) 비전으로 결과 확인
                print(f"\n  [OBSERVE] 트릭샷 결과 확인...")
                time.sleep(3)
                try:
                    cue_f, red_f, yellow_f, _ = detect_balls()
                    o_cx = MAZE_TABLE_CENTER_X
                    o_cy = MAZE_TABLE_CENTER_Y - 0.12
                    rx, ry = 0.035, 0.045
                    g1 = np.array([o_cx + rx*np.cos(np.radians(150)),
                                   o_cy + ry*np.sin(np.radians(150))])
                    g2 = np.array([o_cx + rx*np.cos(np.radians(210)),
                                   o_cy + ry*np.sin(np.radians(210))])
                    da = np.linalg.norm(yellow_f[:2]-g1) + np.linalg.norm(red_f[:2]-g2)
                    db = np.linalg.norm(yellow_f[:2]-g2) + np.linalg.norm(red_f[:2]-g1)
                    total = min(da, db)
                    print(f"  ★ 트릭샷 결과: 합산 거리 {total*100:.1f}cm")
                except Exception as e:
                    print(f"  [OBSERVE] 결과 확인 실패: {e}")
            else:
                print(f"  [FAIL] IK 실패")
        else:
            print(f"  [FAIL] 트릭샷 후보 없음")


    pb.MoveRobot(HOME_Q_DEG, degree=True)
    time.sleep(2)

if DEMO_TYPE in ('pocket_phase1', 'pocket_phase2'):
    NUM_ROUNDS = 0  # pocket은 state_machine으로 이미 처리됨

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
        # saveState 검증에서 사용할 변수 (아래 실행 섹션에서 처리)
        speed = candidates[0]['strike_speed'] if candidates else 1.0
        strike_dir_3d = None  # maze는 아래에서 후보별 처리
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

    # --- 궤적 생성 + SIM 실행 ---
    T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
    q_now = pb.my_robot.q.copy()

    if DEMO_TYPE == 'maze':
        # === saveState 기반 검증 (state_machine.py와 동일) ===
        MAX_VERIFY = 5
        ik_valid_list = []
        angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)

        for ci, cand in enumerate(candidates):
            if len(ik_valid_list) >= MAX_VERIFY:
                break
            sd2d = cand['strike_dir']
            horiz = np.array(sd2d[:2]).flatten()
            hn = np.linalg.norm(horiz)
            if hn > 1e-6:
                horiz = horiz / hn
            sd3d = np.array([horiz[0]*np.cos(angle_rad), horiz[1]*np.cos(angle_rad), -np.sin(angle_rad)])
            sd3d = sd3d / np.linalg.norm(sd3d)

            traj_c, ph_c = traj_planner.plan_strike(
                T_current=T_now, ball_pos=ball_pos, strike_direction=sd3d,
                strike_speed=cand['strike_speed'],
                approach_dist=cand.get('safe_approach_dist', STRIKE_APPROACH_DIST),
                follow_dist=STRIKE_FOLLOW_DIST, strike_height=ball_pos[2],
                tool_offset=tool_offset,
                table_bounds=env.table_bounds if hasattr(env, 'table_bounds') else None)

            approach_end = ph_c.get('approach', (0, 0))[1]
            val_from = int(approach_end * 0.65)
            ik_result = ik.solve_trajectory_validated(q_now, traj_c, validate_from=val_from)
            if ik_result['valid']:
                print(f"  [IK-OK] #{ci+1}/{len(candidates)} "
                      f"(angle={cand['angle_deg']:.1f}, score={cand['score']:.0f})")
                ik_valid_list.append((cand, ik_result, traj_c, ph_c, sd3d.copy()))
            else:
                print(f"  [SKIP] #{ci+1} (angle={cand['angle_deg']:.1f}): IK failed")

        if not ik_valid_list:
            print(f"  [FAIL] All candidates failed IK. Skipping round.")
            pb.MoveRobot(HOME_Q_DEG, degree=True)
            time.sleep(1)
            continue

        use_verify = len(ik_valid_list) > 1
        if use_verify:
            print(f"  [VERIFY] Testing {len(ik_valid_list)} candidates via saveState...")

        # 접촉 추적 리셋 함수
        def _reset_tracking():
            if hasattr(env, 'reset_contact_tracking'):
                env.reset_contact_tracking()
            if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
                p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                         enableCollision=1, physicsClientId=pb.ClientId)

        verified_idx = None
        for vi, (cand, ik_res, traj_v, ph_v, sd3d_v) in enumerate(ik_valid_list):
            is_last = (vi == len(ik_valid_list) - 1)
            if use_verify and not is_last:
                state_id = p.saveState(physicsClientId=pb.ClientId)
                _reset_tracking()

                # Approach
                for i in range(ph_v['approach'][0], ph_v['approach'][1]):
                    pb.MoveRobot(ik_res['q_trajectory'][i], degree=False)
                    time.sleep(0.002)
                time.sleep(0.3)

                # Strike swing
                q_ready_v = ik_res['q_trajectory'][ph_v['approach'][1] - 1].copy()
                q_follow_v = ik.solve_step(q_ready_v, traj_v[-1])
                sw_t = np.linalg.norm(traj_v[-1][:3,3] - traj_v[ph_v['approach'][1]-1][:3,3]) / (cand['strike_speed'] * 0.7)
                sw_t = np.clip(sw_t, 0.05, 0.8)
                avg_qd = (q_follow_v - q_ready_v) / sw_t
                if hasattr(pb.my_robot, '_qdot_des'):
                    pb.my_robot._qdot_des = avg_qd
                pb.MoveRobot(q_follow_v, degree=False)
                time.sleep(sw_t)
                if hasattr(pb.my_robot, '_qdot_des'):
                    pb.my_robot._qdot_des = np.zeros([6, 1])

                # 도구-큐볼 충돌 비활성화 (공이 자유롭게 이동)
                if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
                    p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                             enableCollision=0, physicsClientId=pb.ClientId)

                # 수직 후퇴
                T_lift_v = traj_v[-1].copy()
                T_lift_v[2, 3] += RETRACT_HEIGHT
                q_lift_v = ik.solve_step(q_follow_v, T_lift_v)
                pb.MoveRobot(q_lift_v, degree=False)
                time.sleep(0.3)

                # 공 정지 대기 + 결과 확인
                env.wait_balls_stop(timeout=8.0)
                events = getattr(env, '_contact_events', [])
                from project.physics.cushion_rules import valid_cushion_sequence
                valid_shot = valid_cushion_sequence(events, 2)

                if valid_shot:
                    print(f"    [V#{vi+1}] OK angle={cand['angle_deg']:.1f} events={events}")
                    p.removeState(state_id, physicsClientId=pb.ClientId)
                    verified_idx = vi
                    # 궤적 저장 (이미 실행 완료된 상태)
                    q_traj_full = ik_res['q_trajectory']
                    q_traj_deg = np.degrees(np.array(q_traj_full).reshape(-1, 6))
                    q_follow_deg = np.degrees(np.array(q_follow_v).flatten())
                    saved_trajectories.append((q_traj_deg, q_follow_deg, ph_v))
                    break
                else:
                    print(f"    [V#{vi+1}] MISS angle={cand['angle_deg']:.1f} events={events}")
                    p.restoreState(stateId=state_id, physicsClientId=pb.ClientId)
                    p.removeState(state_id, physicsClientId=pb.ClientId)
                    pb.MoveRobot(HOME_Q_DEG, degree=True)
                    time.sleep(0.3)
            else:
                # 마지막 후보 또는 단일 후보 -> fallback 실행
                verified_idx = vi
                break

        if verified_idx is not None and (not use_verify or verified_idx == len(ik_valid_list) - 1):
            # 최종 후보를 직접 실행 (saveState 없이)
            cand_f, ik_res_f, traj_f, ph_f, sd3d_f = ik_valid_list[verified_idx]
            _reset_tracking()

            print(f"  [SELECTED] #{verified_idx+1}/{len(ik_valid_list)} "
                  f"angle={cand_f['angle_deg']:.1f}, score={cand_f['score']:.0f}")
            print(f"  방향: {sd3d_f}, 속도: {cand_f['strike_speed']:.3f} m/s")

            # Approach
            print(f"  [SIM] Approach...")
            for i in range(ph_f['approach'][0], ph_f['approach'][1]):
                pb.MoveRobot(ik_res_f['q_trajectory'][i], degree=False)
                time.sleep(0.002)
            q_ready_f = ik_res_f['q_trajectory'][ph_f['approach'][1] - 1].copy()
            time.sleep(0.5)

            # Strike
            print(f"  [SIM] Strike!")
            q_follow_f = ik.solve_step(q_ready_f, traj_f[-1])
            sw_t = np.linalg.norm(traj_f[-1][:3,3] - traj_f[ph_f['approach'][1]-1][:3,3]) / (cand_f['strike_speed'] * 0.7)
            sw_t = np.clip(sw_t, 0.05, 0.8)
            avg_qd = (q_follow_f - q_ready_f) / sw_t
            if hasattr(pb.my_robot, '_qdot_des'):
                pb.my_robot._qdot_des = avg_qd
            pb.MoveRobot(q_follow_f, degree=False)
            time.sleep(sw_t)
            if hasattr(pb.my_robot, '_qdot_des'):
                pb.my_robot._qdot_des = np.zeros([6, 1])

            # 도구-큐볼 충돌 비활성화
            if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
                p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                         enableCollision=0, physicsClientId=pb.ClientId)

            # 수직 후퇴
            T_lift_f = traj_f[-1].copy()
            T_lift_f[2, 3] += RETRACT_HEIGHT
            q_lift_f = ik.solve_step(q_follow_f, T_lift_f)
            pb.MoveRobot(q_lift_f, degree=False)
            time.sleep(0.3)

            # 궤적 저장
            q_traj_full = ik_res_f['q_trajectory']
            q_traj_deg = np.degrees(np.array(q_traj_full).reshape(-1, 6))
            q_follow_deg = np.degrees(np.array(q_follow_f).flatten())
            saved_trajectories.append((q_traj_deg, q_follow_deg, ph_f))
        elif verified_idx is not None:
            # saveState 검증에서 이미 성공+저장 완료됨
            cand_f = ik_valid_list[verified_idx][0]
            print(f"  [SELECTED] #{verified_idx+1}/{len(ik_valid_list)} "
                  f"angle={cand_f['angle_deg']:.1f}, score={cand_f['score']:.0f} (verified)")

        # OBSERVE (maze)
        env.wait_balls_stop(timeout=5.0)
        cue_pos_f = env.get_cue_ball_position()
        t1_pos = env.get_target_ball_position()
        d1 = np.linalg.norm(cue_pos_f[:2] - t1_pos[:2])
        d2 = 0
        if hasattr(env, 'ball2_id'):
            t2_pos = env.get_ball2_position()
            d2 = np.linalg.norm(cue_pos_f[:2] - t2_pos[:2])
        events = getattr(env, '_contact_events', [])
        from project.physics.cushion_rules import valid_cushion_sequence
        valid_shot = valid_cushion_sequence(events, 2)
        print(f"  결과: d(tgt1)={d1:.3f}m, d(tgt2)={d2:.3f}m, "
              f"events={events}, valid={'YES' if valid_shot else 'NO'}")
    else:
        # minigolf / billiards: 기존 로직
        print(f"  방향: {strike_dir_3d}, 속도: {speed:.3f} m/s")

        trajectory, phases = traj_planner.plan_strike(
            T_current=T_now, ball_pos=ball_pos, strike_direction=strike_dir_3d,
            strike_speed=speed, approach_dist=STRIKE_APPROACH_DIST,
            follow_dist=STRIKE_FOLLOW_DIST,
            strike_height=ball_pos[2], tool_offset=tool_offset,
            table_bounds=env.table_bounds if hasattr(env, 'table_bounds') else None)

        q_traj = ik.solve_trajectory(q_now, trajectory)
        print(f"  궤적: {len(trajectory)} pts (A:{phases['approach'][1]-phases['approach'][0]}"
              f" S:{phases['strike'][1]-phases['strike'][0]}"
              f" F:{phases['follow'][1]-phases['follow'][0]})")

        # SIM: Approach
        if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
            p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                     enableCollision=1, physicsClientId=pb.ClientId)
        print(f"  [SIM] Approach...")
        for i in range(phases['approach'][0], phases['approach'][1]):
            pb.MoveRobot(q_traj[i], degree=False)
            time.sleep(0.002)
        q_ready = q_traj[phases['approach'][1] - 1].copy()
        time.sleep(0.5)

        # SIM: Strike
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

        if hasattr(env, 'tool_id') and hasattr(env, 'cue_ball_id'):
            p.setCollisionFilterPair(env.tool_id, env.cue_ball_id, -1, -1,
                                     enableCollision=0, physicsClientId=pb.ClientId)

        T_lift = trajectory[-1].copy()
        T_lift[2, 3] += RETRACT_HEIGHT
        q_lift = ik.solve_step(q_follow, T_lift)
        pb.MoveRobot(q_lift, degree=False)
        time.sleep(0.3)

        # OBSERVE
        if DEMO_TYPE == 'minigolf':
            env.wait_ball_stop(timeout=5.0)
            dist = env.get_distance_to_hole()
            success = env.is_hole_in()
            print(f"  결과: 거리={dist:.4f}m {'HOLE-IN-ONE!' if success else ''}")
        else:
            env.wait_balls_stop(timeout=5.0)
            success = env.is_pocketed()
            print(f"  결과: {'POCKETED!' if success else 'miss'}")

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

    # 카메라로 공 정지 확인 + 최종 위치 관측
    if DEMO_TYPE in ('maze', 'pocket_phase1', 'pocket_phase2'):
        try:
            from project.real_env_to_pybullet import detect_balls
            print(f"  [OBSERVE] 카메라로 최종 위치 감지 중...")
            time.sleep(3)  # 공 정지 대기
            cue_f, tgt1_f, tgt2_f, _ = detect_balls()
            print(f"  최종 위치: cue={cue_f[:2]}, t1={tgt1_f[:2]}, t2={tgt2_f[:2]}")

            if DEMO_TYPE == 'pocket_phase2':
                # 트릭샷 결과: trick balls와 O 목표 위치 거리
                o_cx, o_cy = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y - 0.12
                rx, ry = 0.035, 0.045
                # 목표: 150° / 210° 위치
                g1 = np.array([o_cx + rx * np.cos(np.radians(150)),
                               o_cy + ry * np.sin(np.radians(150))])
                g2 = np.array([o_cx + rx * np.cos(np.radians(210)),
                               o_cy + ry * np.sin(np.radians(210))])
                # 두 가지 매칭
                da = np.linalg.norm(tgt1_f[:2] - g1) + np.linalg.norm(tgt2_f[:2] - g2)
                db = np.linalg.norm(tgt1_f[:2] - g2) + np.linalg.norm(tgt2_f[:2] - g1)
                total = min(da, db)
                print(f"  [TRICK] 목표 합산 거리: {total*100:.1f}cm")
        except Exception as e:
            print(f"  [OBSERVE] 카메라 관측 실패: {e}")

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
