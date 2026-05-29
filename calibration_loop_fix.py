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

    방식:
      1. +x 방향으로 비교적 약하게 타격
      2. 사용자가 공 진행 방향 입력
      3. y offset 수정

      4. +y 방향으로 비교적 약하게 타격
      5. 사용자가 공 진행 방향 입력
      6. x offset 수정

    공 진행 방향 (+x 타격 기준):
      [l] 왼쪽 : y_offset 감소 (처음 3mm 이후 빗맞은 타격 횟수당 1mm 씩 감소)
      [r] 오른쪽 : y_offset 증가 ([l]와 변화폭 공유)
      [g] 직진 : offset 확정
      [m] miss => 공의 왼쪽, 오른쪽을 빗나가는지 사용자 입력 후 offset 크게 변화 (5mm)

    핵심:
      +x shot -> y 보정
      +y shot -> x 보정
      coarse-to-fine calibration

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
    offset_size = 0.003

    while abs(offset_size) > 1e-6:
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
        q_traj_deg = np.degrees(np.array(q_traj).reshape(-1, 6))

        # 3. 시뮬 진행 (strike 후 복귀까지)
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
            # 4. 실제 로봇도 진행 (strike 후 복귀까지)
            print("  [REAL] 실제 로봇 재생...")
            q_follow_deg = np.degrees(q_follow)
            print("  실제 로봇 접근 중...")
            _replay_strike_on_real(indy, pb, q_traj_deg, q_follow_deg, phases, speed = 0.1)

        except Exception as e:
            print(f"  [ERROR] 실제 로봇 실행 실패: {e}")
            try:
                indy.recover()
            except:
                pass
            _return_home(indy, pb)
            continue

        # 5. 공 움직임 입력
        print("\n공 진행 방향을 입력하세요:")
        print("  [l] 왼쪽으로 감")
        print("  [r] 오른쪽으로 감")
        print("  [g] 직진 (good)")
        print("  [m] 공을 못 맞춤")

        cmd = input("입력 > ").strip().lower()

        if cmd == 'l':
            """
            공이 왼쪽으로 갔음 (+y)
            공의 오른쪽을 침
            공이 더 오른쪽으로 가야함
            y_offset 감소
            """
            
            offset['y'] -= offset_size
            offset_size = max(0.0, offset_size - 0.001)
            print(f"  new y offset = {offset['y']:.4f}")
        
        elif cmd == 'r':
            """
            공이 오른쪽으로 갔음 (-y)
            공의 왼쪽을 침
            공이 더 왼쪽으로 가야함
            y_offset 증가
            """
            
            offset['y'] += offset_size
            offset_size = max(0.0, offset_size - 0.001)
            print(f"  new y offset = {offset['y']:.4f}")

        elif cmd == 'g':
            print("\n  Y offset calibration 완료")
            save_position_offset(offset)
            _return_home(indy, pb)
            break

        elif cmd == 'm':
            print("\n공의 어디를 지나쳤는지를 입력하세요:")
            print("  [l] 왼쪽으로 감")
            print("  [r] 오른쪽으로 감")
            cmd2 = input("입력 > ").strip().lower()

            if cmd2 == 'l':
                """
                툴팁이 왼쪽으로 지나갔기 때문에 공을 +y로 이동해야함
                """
                offset['y'] += 0.005
                print(f"  new y offset = {offset['y']:.4f}")
            
            elif cmd2 == 'r':
                """
                툴팁이 오른쪽으로 지나갔기 때문에 공을 -y로 이동해야함
                """
                offset['y'] -= 0.005
                print(f"  new y offset = {offset['y']:.4f}")

            else:
                print("잘못된 입력")
        
        else:
            print("잘못된 입력")
        
        save_position_offset(offset)
        _return_home(indy, pb)

    offset_size = 0.003
    while abs(offset_size) > 1e-6:
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
        approach_dir = np.array([0.0, 1.0, 0.0])  # +y 방향
        trajectory, phases = traj_planner.plan_strike(
            T_current=T_now, ball_pos=cue_pos,
            strike_direction=approach_dir,
            strike_speed=0.1,  # 매우 느리게
            approach_dist=0.05,  # 5cm만 접근
            follow_dist=0.02,
            table_bounds=env.table_bounds
        )

        q_traj = ik.solve_trajectory(q_now, trajectory)
        q_traj_deg = np.degrees(np.array(q_traj).reshape(-1, 6))

        # 3. 시뮬 진행 (strike 후 복귀까지)
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
            # 4. 실제 로봇도 진행 (strike 후 복귀까지)
            print("  [REAL] 실제 로봇 재생...")
            q_follow_deg = np.degrees(q_follow)
            print("  실제 로봇 접근 중...")
            _replay_strike_on_real(indy, pb, q_traj_deg, q_follow_deg, phases, speed = 0.1)

        except Exception as e:
            print(f"  [ERROR] 실제 로봇 실행 실패: {e}")
            try:
                indy.recover()
            except:
                pass
            _return_home(indy, pb)
            continue

        # 5. 공 움직임 입력
        print("\n공 진행 방향을 입력하세요:")
        print("  [l] 왼쪽으로 감")
        print("  [r] 오른쪽으로 감")
        print("  [g] 직진 (good)")
        print("  [m] 공을 못 맞춤")

        cmd = input("입력 > ").strip().lower()

        if cmd == 'l':
            """
            공이 왼쪽으로 갔음 (-x)
            공의 오른쪽을 침
            공이 더 오른쪽으로 가야함
            x_offset 증가
            """
            
            offset['x'] += offset_size
            offset_size = max(0.0, offset_size - 0.001)
            print(f"  new x offset = {offset['x']:.4f}")
        
        elif cmd == 'r':
            """
            공이 오른쪽으로 갔음 (+x)
            공의 왼쪽을 침
            공이 더 왼쪽으로 가야함
            x_offset 감소
            """
            
            offset['x'] -= offset_size
            offset_size = max(0.0, offset_size - 0.001)
            print(f"  new x offset = {offset['x']:.4f}")

        elif cmd == 'g':
            print("\n  X offset calibration 완료")
            save_position_offset(offset)
            _return_home(indy, pb)
            break

        elif cmd == 'm':
            print("\n공의 어디를 지나쳤는지를 입력하세요:")
            print("  [l] 왼쪽으로 감")
            print("  [r] 오른쪽으로 감")
            cmd2 = input("입력 > ").strip().lower()

            if cmd2 == 'l':
                """
                툴팁이 왼쪽으로 지나갔기 때문에 공을 -x로 이동해야함
                """
                offset['x'] -= 0.005
                print(f"  new x offset = {offset['x']:.4f}")
            
            elif cmd2 == 'r':
                """
                툴팁이 오른쪽으로 지나갔기 때문에 공을 +x로 이동해야함
                """
                offset['x'] += 0.005
                print(f"  new x offset = {offset['x']:.4f}")

            else:
                print("잘못된 입력")
        
        else:
            print("잘못된 입력")
        
        save_position_offset(offset)
        _return_home(indy, pb)

    save_position_offset(offset)
    print(f"\n  최종 오프셋: x={offset['x']:.4f}m, y={offset['y']:.4f}m")
    return offset


def _replay_approach_only(indy, pb, q_traj_deg):
    """접근 궤적만 실제 로봇에서 재생 (느리게, 안전하게)."""
    # movej로 시작점 이동
    indy.movej(list(q_traj_deg[0]))
    _wait_indy(indy)
    time.sleep(0.5)

    # 접근 궤적 재생 (movej, 느린 속도)
    step = max(1, len(q_traj_deg) // 20)  # 20개 포인트로 축소
    for i in range(0, len(q_traj_deg), step):
        try:
            indy.movej(list(q_traj_deg[i]), vel_ratio=20, acc_ratio=50)
            _wait_indy(indy, timeout=10)
        except Exception as e:
            print(f"    movej 오류: {e}")
            break


def _wait_indy(indy, timeout=30, pb=None):
    """실제 로봇 동작 완료 대기 + 시뮬 동기화 (7TestCode wait_indy와 동일)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pb is not None:
            # 실제 로봇 관절각도를 시뮬에 반영
            q = indy.get_control_data()['q']
            pb.MoveRobot(q, degree=True)
        if not indy.get_motion_data()["is_in_motion"]:
            break
        time.sleep(0.01)


def _return_home(indy, pb):
    """로봇을 홈 위치로 복귀 (시뮬 동기화 포함)."""
    indy.movej(list(HOME_Q_DEG))
    _wait_indy(indy, pb=pb)
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
    """실제 로봇에서 접근→타격 재생 (7TestCode replay_trajectory_on_real과 동일).

    Phase 1 (Approach):  waypoint별 MoveJ
    Phase 1.5 (Align):   Ready 위치 정밀 정렬 MoveJ
    Phase 2 (Strike):    Follow-through MoveJ (vel=100, acc=300)
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

    print(f"  [REAL] Phase 1: MoveJ Approach ({len(waypoint_indices)} waypoints)...")
    for wi, idx in enumerate(waypoint_indices):
        indy.movej([float(x) for x in q_traj_deg[idx]], vel_ratio=APPROACH_VEL, acc_ratio=APPROACH_ACC)
        _wait_indy(indy, pb=pb)
    print(f"  [REAL] Approach 완료")

    # ======== Phase 1.5: Align ========
    q_ready = q_traj_deg[approach_end - 1]
    print(f"  [REAL] Phase 1.5: Align")
    time.sleep(0.5)
    indy.movej([float(x) for x in q_ready], vel_ratio=10, acc_ratio=30)
    _wait_indy(indy, pb=pb)
    time.sleep(0.5)

    # ======== Phase 2: Strike (MoveL 직선 타격) ========
    # movej는 곡선이 되므로, 타격만 movel로 직선 보장
    # FK 변환 없이: 실제 로봇 현재 위치 + 시뮬 delta로 계산
    p_ready = indy.get_control_data()['p']  # [x_mm, y_mm, z_mm, rx, ry, rz]

    # 시뮬에서 ready→follow 간 Cartesian 변위 계산
    T_ready_sim = pb.my_robot.pinModel.FK(np.radians(q_ready))
    T_follow_sim = pb.my_robot.pinModel.FK(np.radians(q_follow_deg))
    delta_mm = (T_follow_sim[:3, 3] - T_ready_sim[:3, 3]) * 1000.0  # m → mm

    # 실제 로봇 좌표계에서 follow 위치 = 현재 위치 + delta (자세는 유지)
    p_follow = list(p_ready)
    p_follow[0] += delta_mm[0]
    p_follow[1] += delta_mm[1]
    p_follow[2] += delta_mm[2]
    # 자세(rx, ry, rz)는 그대로 유지

    print(f"  [REAL] Phase 2: MoveL Strike!")
    print(f"    delta: [{delta_mm[0]:.1f}, {delta_mm[1]:.1f}, {delta_mm[2]:.1f}] mm")
    indy.movel(p_follow, vel_ratio=100, acc_ratio=300)
    _wait_indy(indy, pb=pb)
    print(f"  [REAL] Strike 완료!")

    # ======== Phase 3: Home ========
    print(f"  [REAL] Phase 3: Home")
    indy.movej(list(HOME_Q_DEG), vel_ratio=30, acc_ratio=100)
    _wait_indy(indy, pb=pb)


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
    # 실제 로봇 현재 위치를 시뮬에 동기화
    q_real = indy.get_control_data()['q']
    pb.MoveRobot(q_real, degree=True)
    print(f"  현재 q(deg): {[round(x,1) for x in q_real]}")

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
