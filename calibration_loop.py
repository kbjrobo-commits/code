# -*- coding: utf-8 -*-
"""
완전 자동 물리 캘리브레이션 루프
=================================
Phase 1: 좌표 캘리브레이션 — 카메라→시뮬 좌표 매핑 검증/보정
Phase 2: 물리 캘리브레이션 — 마찰/반발/속도전달비 최적화

사용법:
  # Phase 1만 (좌표 보정)
  python calibration_loop.py --phase position

  # Phase 2만 (물리 최적화, 좌표가 맞는 상태에서)
  python calibration_loop.py --phase physics --num-trials 5

  # 전체 (좌표 → 물리 순서)
  python calibration_loop.py --phase all --num-trials 5

  # 시뮬 테스트 (실제 로봇 없이)
  python calibration_loop.py --test

원리:
  Phase 1: 카메라로 공 검출 → 로봇이 공 위치로 이동 → 실제 접촉 여부로 좌표 오프셋 보정
  Phase 2: 로봇이 자동으로 간단한 타격 수행 → 카메라로 전/후 관측 → Nelder-Mead 최적화
"""
import numpy as np
import pybullet as p
import pybullet_data
import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project.config import *


# ============================================================
# 파라미터 정의
# ============================================================
PARAM_NAMES = [
    'ball_restitution',      # 공 반발계수
    'cushion_restitution',   # 쿠션 반발계수
    'rolling_friction',      # 구름 마찰
    'lateral_friction',      # 횡 마찰
    'speed_gain_scale',      # 도구→공 속도전달비 보정
    'table_y_offset',        # 테이블 Y 위치 보정 (m)
]

PARAM_DEFAULTS = np.array([
    MAZE_BALL_RESTITUTION,       # 0.85
    MAZE_CUSHION_RESTITUTION,    # 0.8
    MAZE_BALL_ROLLING_FRICTION,  # 0.012
    MAZE_BALL_FRICTION,          # 0.15
    BALL_SPEED_GAIN_SCALE,       # 1.0
    0.0,                         # table_y_offset
])

PARAM_BOUNDS = [
    (0.5, 1.0),    # ball_restitution
    (0.5, 1.0),    # cushion_restitution
    (0.001, 0.05), # rolling_friction
    (0.05, 0.5),   # lateral_friction
    (0.7, 1.5),    # speed_gain_scale
    (-0.02, 0.02), # table_y_offset
]

CALIB_FILE = 'calibration_result_physics.npz'
POSITION_CALIB_FILE = 'calibration_position_offset.json'


# ============================================================
# Phase 1: 좌표 캘리브레이션
# ============================================================
def run_position_calibration(indy, pb, env, ik):
    """카메라→시뮬 좌표 매핑을 자동으로 보정.

    절차:
      1. 카메라로 큐볼 위치 검출
      2. 시뮬에서 해당 위치로 도구 이동 궤적 생성
      3. 로봇이 공 위치로 천천히 접근 (치지 않고 접근만)
      4. 접근 후 카메라로 공 위치 재확인:
         - 공이 움직였으면 → 도구가 닿음 → 좌표 대략 맞음
         - 공이 안 움직였으면 → 도구가 빗나감 → 좌표 불일치
      5. 오프셋 조정 후 반복
    """
    from project.real_env_to_pybullet import detect_balls
    from project.trajectory_planner import StrikeTrajectoryPlanner

    traj_planner = StrikeTrajectoryPlanner()

    # 기존 오프셋 로드
    offset = load_position_offset()
    print(f"\n{'='*60}")
    print(f"  Phase 1: 좌표 캘리브레이션")
    print(f"  현재 오프셋: x={offset['x']:.4f}m, y={offset['y']:.4f}m")
    print(f"{'='*60}")

    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration+1}/{max_iterations} ---")

        # 1. 카메라로 공 검출
        print("  카메라로 공 검출 중...")
        try:
            cue_pos, target_pos, ball2_pos = detect_balls()
        except Exception as e:
            print(f"  [ERROR] 공 검출 실패: {e}")
            continue

        # detect_balls()가 캘리브레이션 오프셋을 자동 적용
        print(f"  큐볼 위치: [{cue_pos[0]:.4f}, {cue_pos[1]:.4f}, {cue_pos[2]:.4f}]")

        # 시뮬에서 공 위치 업데이트
        env.setup(cue_pos=cue_pos, target_pos=target_pos, ball2_pos=ball2_pos,
                  num_obstacles=0)

        # 2. 로봇을 공 위치로 천천히 접근 (타격 방향: +x, 매우 느린 속도)
        T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
        q_now = pb.my_robot.q.copy()

        # 간단한 접근: 공 바로 위에서 접근
        approach_dir = np.array([1.0, 0.0, 0.0])  # +x 방향
        trajectory, phases = traj_planner.plan_strike(
            T_current=T_now, ball_pos=cue_pos,
            strike_direction=approach_dir,
            strike_speed=0.1,  # 매우 느리게
            approach_dist=0.05,  # 5cm만 접근
            follow_dist=0.02,
            table_bounds=env.table_bounds
        )

        q_traj = ik.solve_trajectory(q_now, trajectory)

        # 3. 시뮬에서 접근만 (strike 전까지)
        approach_end = phases['approach'][1]
        print(f"  접근 중... ({approach_end} points)")
        for i in range(approach_end):
            pb.MoveRobot(q_traj[i], degree=False)
            time.sleep(0.002)

        # 4. 실제 로봇도 접근 (approach만)
        q_approach_deg = np.degrees(q_traj[:approach_end])
        print("  실제 로봇 접근 중...")
        _replay_approach_only(indy, pb, q_approach_deg)

        # 5. 카메라로 공 위치 재확인
        time.sleep(1.0)  # 안정화 대기
        print("  카메라로 재확인 중...")
        try:
            cue_pos2, _, _ = detect_balls()
            # detect_balls()가 오프셋 자동 적용
        except Exception as e:
            print(f"  [ERROR] 재검출 실패: {e}")
            _return_home(indy, pb)
            continue

        # 6. 공이 움직였는지 확인
        displacement = np.linalg.norm(
            np.array(cue_pos2[:2]) - np.array(cue_pos[:2]))
        print(f"  공 변위: {displacement*1000:.1f}mm")

        if displacement > 0.005:  # 5mm 이상 움직임
            print(f"  ✅ 도구가 공에 닿았습니다! 좌표 캘리브레이션 완료.")
            _return_home(indy, pb)
            break
        else:
            # 공이 안 움직임 → 좌표 불일치
            # 간단한 보정: 도구 현재 위치 vs 목표 위치 차이로 오프셋 추정
            # (실제로는 더 정교한 방법이 필요할 수 있음)
            print(f"  ❌ 도구가 공에 닿지 않았습니다.")
            print(f"     오프셋을 수동으로 조정하거나,")
            print(f"     ArUco 마커 위치를 재확인하세요.")
            # 작은 오프셋 시도: y를 0.005m씩 조정
            offset['y'] += 0.005
            print(f"     y_offset을 +5mm 조정: {offset['y']:.4f}m")
            save_position_offset(offset)

        _return_home(indy, pb)

    save_position_offset(offset)
    print(f"\n  최종 오프셋: x={offset['x']:.4f}m, y={offset['y']:.4f}m")
    return offset


def _replay_approach_only(indy, pb, q_traj_deg):
    """접근 궤적만 실제 로봇에서 재생 (느리게, 안전하게)."""
    from src.utils import Rot2eul
    # movej로 시작점 이동
    indy.movej(list(q_traj_deg[0]))
    _wait_indy(indy)
    time.sleep(0.5)

    # 접근 궤적 재생 (movel, 느린 속도)
    step = max(1, len(q_traj_deg) // 20)  # 20개 포인트로 축소
    for i in range(0, len(q_traj_deg), step):
        T = pb.my_robot.pinModel.FK(np.radians(q_traj_deg[i]))
        p6 = np.zeros(6)
        p6[0:3] = 1000 * T[0:3, 3]
        p6[3:6] = Rot2eul(T[0:3, 0:3], seq='XYZ', degree=True)
        try:
            indy.movel(list(p6), vel_ratio=20, acc_ratio=50)
            _wait_indy(indy, timeout=10)
        except Exception as e:
            print(f"    movel 오류: {e}")
            break


def _wait_indy(indy, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not indy.get_motion_data()["is_in_motion"]:
            break
        time.sleep(0.01)


def _return_home(indy, pb):
    """로봇을 홈 위치로 복귀."""
    indy.movej(list(HOME_Q_DEG))
    _wait_indy(indy)
    pb.MoveRobot(HOME_Q_DEG, degree=True)
    time.sleep(1)


def load_position_offset(path=None):
    if path is None:
        path = POSITION_CALIB_FILE
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'x': 0.0, 'y': 0.0}


def save_position_offset(offset, path=None):
    if path is None:
        path = POSITION_CALIB_FILE
    with open(path, 'w') as f:
        json.dump(offset, f, indent=2)
    print(f"  좌표 오프셋 저장: {path}")


# ============================================================
# Phase 2: 물리 캘리브레이션 (완전 자동)
# ============================================================
def run_physics_calibration(indy, pb, env, ik, num_trials=5):
    """로봇이 자동으로 타격 → 카메라로 관측 → 파라미터 최적화.

    절차:
      1. 카메라로 3공 검출
      2. 플래너가 간단한 직선/1쿠션 타격 계산
      3. 로봇이 자동으로 타격 (7TestCode 흐름)
      4. 공이 멈추면 카메라로 최종 위치 촬영
      5. N회 반복 후 Nelder-Mead 최적화
    """
    from project.real_env_to_pybullet import detect_balls
    from project.physics.cushion_planner import CushionShotPlanner
    from project.trajectory_planner import StrikeTrajectoryPlanner
    from src.utils import Rot2eul

    traj_planner = StrikeTrajectoryPlanner()
    offset = load_position_offset()

    print(f"\n{'='*60}")
    print(f"  Phase 2: 물리 캘리브레이션 ({num_trials}회 자동 타격)")
    print(f"  좌표 오프셋: x={offset['x']:.4f}m, y={offset['y']:.4f}m")
    print(f"{'='*60}")

    trials = []
    for trial_idx in range(num_trials):
        print(f"\n{'='*40}")
        print(f"  Trial {trial_idx+1}/{num_trials}")
        print(f"{'='*40}")

        # 1. 카메라로 초기 위치 검출
        print("  [SCAN] 카메라로 공 검출...")
        try:
            cue_start, tgt1_start, tgt2_start = detect_balls()
            # detect_balls()가 오프셋 자동 적용
        except Exception as e:
            print(f"  [ERROR] 공 검출 실패: {e}")
            continue

        print(f"  cue={cue_start[:2]}, t1={tgt1_start[:2]}, t2={tgt2_start[:2]}")

        # 시뮬 환경 설정
        env.setup(cue_pos=cue_start, target_pos=tgt1_start,
                  ball2_pos=tgt2_start, num_obstacles=0)

        # 2. 플래너로 타격 계획 (자동 — 각도/속도 자동 결정)
        print("  [THINK] 타격 계획 중...")
        shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
        ball_pos = np.array(cue_start)
        target_pos = np.array(tgt1_start)
        ball2_pos = np.array(tgt2_start)

        try:
            candidates = shot_planner.plan_shot(ball_pos, target_pos,
                                                 obstacles=[], ball2_pos=ball2_pos)
        except Exception as e:
            print(f"  [ERROR] 플래너 실패: {e}")
            continue

        if not candidates:
            print("  [ERROR] 유효한 후보 없음")
            continue

        best = candidates[0]
        strike_dir_2d = best['strike_dir']
        speed = best['strike_speed']
        angle_deg = best.get('angle_deg', 0)

        horiz = np.array(strike_dir_2d[:2]).flatten()
        horiz = horiz / np.linalg.norm(horiz)
        strike_dir_3d = np.array([horiz[0], horiz[1], 0.0])

        print(f"  방향: {angle_deg:.1f}°, 속도: {speed:.3f} m/s, "
              f"쿠션: {best.get('cushion_count', '?')}")

        # 3. 궤적 생성
        T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
        q_now = pb.my_robot.q.copy()

        approach_dist = best.get('safe_approach_dist', STRIKE_APPROACH_DIST)
        trajectory, phases = traj_planner.plan_strike(
            T_current=T_now, ball_pos=ball_pos,
            strike_direction=strike_dir_3d,
            strike_speed=speed,
            approach_dist=approach_dist,
            follow_dist=STRIKE_FOLLOW_DIST,
            table_bounds=env.table_bounds
        )

        q_traj = ik.solve_trajectory(q_now, trajectory)
        q_traj_deg = np.degrees(np.array(q_traj).reshape(-1, 6))

        # 4. 시뮬 실행 (approach + strike)
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

        # 5. 실제 로봇 재생
        print("  [REAL] 실제 로봇 재생...")
        q_follow_deg = np.degrees(q_follow)
        try:
            _replay_strike_on_real(indy, pb, q_traj_deg, q_follow_deg,
                                    phases, speed)
        except Exception as e:
            print(f"  [ERROR] 실제 로봇 실행 실패: {e}")
            try:
                indy.recover()
            except:
                pass
            _return_home(indy, pb)
            continue

        # 6. 카메라로 공 정지 대기 + 최종 위치 촬영
        print("  [WAIT] 공 정지 대기 (카메라 기반)...")
        from project.real_env_to_pybullet import wait_real_balls_stop
        try:
            final_positions = wait_real_balls_stop(
                interval=0.5, threshold_mm=3.0, max_wait=10.0)
            cue_final, tgt1_final, tgt2_final = final_positions
            # detect_balls()가 오프셋 자동 적용
        except Exception as e:
            print(f"  [ERROR] 최종 검출 실패: {e}")
            _return_home(indy, pb)
            continue

        # 8. trial 데이터 저장
        trial = {
            'cue_start': np.array(cue_start[:2]),
            'target_start': np.array(tgt1_start[:2]),
            'ball2_start': np.array(tgt2_start[:2]),
            'strike_angle': np.radians(angle_deg),  # 플래너에서 자동
            'strike_speed': speed,                    # 플래너에서 자동
            'cue_final': np.array(cue_final[:2]),
            'target_final': np.array(tgt1_final[:2]),
            'ball2_final': np.array(tgt2_final[:2]),
        }
        trials.append(trial)

        cue_disp = np.linalg.norm(trial['cue_final'] - trial['cue_start'])
        print(f"  큐볼 변위: {cue_disp*100:.1f}cm")
        if cue_disp < 0.01:
            print(f"  ⚠️ 큐볼이 거의 안 움직임 — 헛침 또는 좌표 불일치!")

        # 홈 복귀
        _return_home(indy, pb)
        print(f"  Trial {trial_idx+1} 완료")

    # 9. 최적화
    if len(trials) < 2:
        print(f"\n  [ERROR] 유효한 trial이 {len(trials)}개뿐 — 최소 2개 필요")
        return None

    # 데이터 저장
    np.savez('calibration_trials.npz', trials=trials)
    print(f"\n  {len(trials)}개 trial 저장: calibration_trials.npz")

    # Nelder-Mead 최적화
    optimal = optimize_parameters(trials)
    save_calibration(optimal)
    return optimal


def _replay_strike_on_real(indy, pb, q_traj_deg, q_follow_deg, phases, speed):
    """실제 로봇에서 접근→타격 재생 (7TestCode 흐름 기반)."""
    from src.utils import Rot2eul

    # Approach: movej로 ready 위치까지
    ready_idx = phases['approach'][1] - 1
    q_ready_deg = q_traj_deg[ready_idx]

    # 중간 경유점으로 안전 접근
    # 먼저 상공 위치로 이동
    mid_idx = max(0, ready_idx - 200)
    indy.movej(list(q_traj_deg[mid_idx]))
    _wait_indy(indy)

    # ready 위치로 movel 접근
    T_ready = pb.my_robot.pinModel.FK(np.radians(q_ready_deg))
    p6_ready = np.zeros(6)
    p6_ready[0:3] = 1000 * T_ready[0:3, 3]
    p6_ready[3:6] = Rot2eul(T_ready[0:3, 0:3], seq='XYZ', degree=True)
    indy.movel(list(p6_ready), vel_ratio=30, acc_ratio=60)
    _wait_indy(indy)
    time.sleep(0.3)

    # Strike: movel로 follow 위치까지 (빠르게)
    T_follow = pb.my_robot.pinModel.FK(np.radians(q_follow_deg))
    p6_follow = np.zeros(6)
    p6_follow[0:3] = 1000 * T_follow[0:3, 3]
    p6_follow[3:6] = Rot2eul(T_follow[0:3, 0:3], seq='XYZ', degree=True)

    dist_mm = np.linalg.norm(p6_follow[:3] - p6_ready[:3])
    vel_ratio = min(100, max(50, int(speed * 100)))
    indy.movel(list(p6_follow), vel_ratio=vel_ratio, acc_ratio=100)
    _wait_indy(indy, timeout=5)

    # Retract: 수직 상승
    p6_retract = p6_follow.copy()
    p6_retract[2] += RETRACT_HEIGHT * 1000
    indy.movel(list(p6_retract), vel_ratio=30, acc_ratio=60)
    _wait_indy(indy)


# ============================================================
# 시뮬레이션 평가 함수 (변경 없음)
# ============================================================
def simulate_strike(trial, params):
    """주어진 파라미터로 헤드리스 PyBullet 시뮬레이션 실행."""
    ball_rest, cushion_rest = params[0], params[1]
    roll_fric, lat_fric = params[2], params[3]
    speed_scale, y_offset = params[4], params[5]

    sim_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=sim_id)
    p.setGravity(0, 0, -9.8, physicsClientId=sim_id)
    p.setTimeStep(1/240, physicsClientId=sim_id)

    L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
    H, TH, CH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT, MAZE_CUSHION_HEIGHT
    CX = MAZE_TABLE_CENTER_X
    CY = MAZE_TABLE_CENTER_Y + y_offset
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
    center = np.array([CX, CY, H])

    # 테이블
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2],
                                  physicsClientId=sim_id)
    tid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             basePosition=list(center), physicsClientId=sim_id)
    p.changeDynamics(tid, -1, lateralFriction=lat_fric, restitution=0.5,
                      physicsClientId=sim_id)

    # 쿠션
    top_z = center[2] + TH / 2 + CH / 2
    thickness = 0.03
    for pos, he in [
        ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
        ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
        ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
    ]:
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=sim_id)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                 basePosition=pos, physicsClientId=sim_id)
        p.changeDynamics(cid, -1, restitution=cushion_rest, physicsClientId=sim_id)

    # 공 생성
    def make_ball(pos2d):
        c = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                    physicsClientId=sim_id)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=c,
                                 basePosition=[pos2d[0], pos2d[1], ball_h],
                                 physicsClientId=sim_id)
        p.changeDynamics(bid, -1, lateralFriction=lat_fric, restitution=ball_rest,
                          rollingFriction=roll_fric, spinningFriction=0.02,
                          physicsClientId=sim_id)
        return bid

    cue_id = make_ball(trial['cue_start'])
    tgt1_id = make_ball(trial['target_start'])
    tgt2_id = make_ball(trial['ball2_start'])

    for _ in range(50):
        p.stepSimulation(physicsClientId=sim_id)

    # 타격
    angle = trial['strike_angle']
    ball_speed_gain = ((1.0 + np.sqrt(TOOL_HEAD_RESTITUTION * ball_rest))
                       * TOOL_HEAD_MASS / (TOOL_HEAD_MASS + MAZE_BALL_MASS)
                       * speed_scale)
    ball_speed = trial['strike_speed'] * ball_speed_gain
    p.resetBaseVelocity(cue_id,
                         linearVelocity=[ball_speed*np.cos(angle),
                                         ball_speed*np.sin(angle), 0],
                         physicsClientId=sim_id)

    for step in range(1200):
        p.stepSimulation(physicsClientId=sim_id)
        if step > 200 and step % 50 == 0:
            if all(np.linalg.norm(p.getBaseVelocity(b, physicsClientId=sim_id)[0][:2]) < 0.005
                   for b in [cue_id, tgt1_id, tgt2_id]):
                break

    result = {k: np.array(p.getBasePositionAndOrientation(b, physicsClientId=sim_id)[0][:2])
              for k, b in [('cue_final', cue_id), ('target_final', tgt1_id),
                           ('ball2_final', tgt2_id)]}
    p.disconnect(sim_id)
    return result


def evaluate(params, trials):
    """Loss = Σ ||actual - sim||²"""
    total = 0.0
    for trial in trials:
        sim = simulate_strike(trial, params)
        for key in ['cue_final', 'target_final', 'ball2_final']:
            if key in trial and trial[key] is not None:
                total += np.sum((sim[key] - np.array(trial[key]))**2)
    return total


def optimize_parameters(trials, method='nelder-mead'):
    """Nelder-Mead로 파라미터 최적화."""
    from scipy.optimize import minimize, differential_evolution

    x0 = PARAM_DEFAULTS.copy()
    print(f"\n  초기 파라미터: {dict(zip(PARAM_NAMES, x0))}")
    initial_loss = evaluate(x0, trials)
    print(f"  초기 Loss: {initial_loss:.6f}")

    if method == 'nelder-mead':
        result = minimize(evaluate, x0, args=(trials,), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                   'disp': True})
    else:
        result = differential_evolution(evaluate, PARAM_BOUNDS, args=(trials,),
                                         maxiter=200, tol=1e-6, disp=True, seed=42)

    optimal = result.x
    print(f"\n  최적 파라미터:")
    for name, val, default in zip(PARAM_NAMES, optimal, PARAM_DEFAULTS):
        change = (val - default) / default * 100 if default != 0 else 0
        print(f"    {name}: {default:.4f} → {val:.4f} ({change:+.1f}%)")
    print(f"  최종 Loss: {result.fun:.6f} (초기: {initial_loss:.6f})")
    return optimal


def save_calibration(params, path=None):
    if path is None:
        path = CALIB_FILE
    np.savez(path, **dict(zip(PARAM_NAMES, params)))
    print(f"\n  캘리브레이션 결과 저장: {path}")


def load_calibration(path=None):
    if path is None:
        path = CALIB_FILE
    if not os.path.exists(path):
        return None
    calib = np.load(path)
    return {name: float(calib[name]) for name in PARAM_NAMES if name in calib}


# ============================================================
# 메인
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='완전 자동 물리 캘리브레이션')
    parser.add_argument('--phase', choices=['position', 'physics', 'all'],
                        default='all', help='실행할 캘리브레이션 단계')
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--robot-ip', type=str, default='192.168.0.13')
    parser.add_argument('--test', action='store_true',
                        help='시뮬 테스트 (실제 로봇 없이)')
    parser.add_argument('--optimize-only', action='store_true',
                        help='저장된 데이터로 최적화만')
    parser.add_argument('--data', type=str, default='calibration_trials.npz')
    args = parser.parse_args()

    if args.test:
        # 시뮬 테스트
        print("  테스트 모드: 시뮬 평가 함수 확인")
        test_trial = {
            'cue_start': np.array([0.345, 0.1725]),
            'target_start': np.array([0.345, 0.2888]),
            'ball2_start': np.array([0.45, 0.25]),
            'strike_angle': np.radians(166.0),
            'strike_speed': MAX_TOOL_SPEED,
            'cue_final': np.array([0.40, 0.30]),
            'target_final': np.array([0.35, 0.35]),
            'ball2_final': np.array([0.50, 0.20]),
        }
        result = simulate_strike(test_trial, PARAM_DEFAULTS)
        print(f"  시뮬: cue={result['cue_final']}, t1={result['target_final']}")
        loss = evaluate(PARAM_DEFAULTS, [test_trial])
        print(f"  Loss: {loss:.6f}")
        sys.exit(0)

    if args.optimize_only:
        data = np.load(args.data, allow_pickle=True)
        optimal = optimize_parameters(list(data['trials']))
        save_calibration(optimal)
        sys.exit(0)

    # 실제 로봇 연결
    from src.core.pybullet_core import PybulletCore
    from neuromeka import IndyDCP3
    from project.environment.maze_env import MazeEnvironment
    from project.ik_solver import IKSolver

    pb = PybulletCore()
    pb.connect(robot_name="indy7_v2", joint_limit=True,
               constraint_visualization=False)
    ik = IKSolver(pb.my_robot.pinModel, gain=IK_GAIN, damping=IK_DAMPING)
    indy = IndyDCP3(robot_ip=args.robot_ip, index=0)
    print(f"  로봇 연결: {args.robot_ip}")

    robot_id = pb.my_robot.robotId
    ee_link = pb.my_robot.RobotEEJointIdx[-1]
    env = MazeEnvironment(pb.ClientId)
    env.setup(num_obstacles=0, skip_balls=True)
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()

    # PD 게인 강화
    robot = pb.my_robot
    def _boosted():
        qddot = robot._qddot_des + 5000*(robot._q_des-robot._q) + 200*(robot._qdot_des-robot._qdot)
        robot._tau = robot._M @ qddot + robot._c + robot._g
    robot._compute_torque_input = _boosted

    # 홈 위치
    _return_home(indy, pb)

    if args.phase in ('position', 'all'):
        run_position_calibration(indy, pb, env, ik)

    if args.phase in ('physics', 'all'):
        run_physics_calibration(indy, pb, env, ik, num_trials=args.num_trials)

    pb.disconnect()
    print("\n  캘리브레이션 완료!")
