# -*- coding: utf-8 -*-
"""
자동 물리 캘리브레이션 루프
============================
실제 로봇 타격 결과와 시뮬레이션을 비교하여
물리 파라미터 (반발계수, 마찰, 속도전달비, 테이블 오프셋)를 자동 최적화.

사용법:
  1. 데이터 수집 모드: 실제 로봇+카메라로 N회 타격 후 결과 저장
     python calibration_loop.py --collect --num-trials 5

  2. 최적화 모드: 저장된 데이터로 파라미터 최적화
     python calibration_loop.py --optimize --data calibration_trials.npz

  3. 결과 적용: 최적 파라미터를 config에서 로드
     calibration_result_physics.npz 파일이 존재하면 config.py에서 자동 로드

원리:
  1. 알려진 각도/속도로 수구 타격
  2. 카메라로 3공 최종 위치 관측
  3. PyBullet 헤드리스에서 같은 타격을 다양한 파라미터로 시뮬
  4. Loss = Σ ||actual_final - sim_final||² 최소화
  5. scipy.optimize.minimize (Nelder-Mead) 또는 differential_evolution
"""
import numpy as np
import pybullet as p
import pybullet_data
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project.config import *


# ============================================================
# 파라미터 정의
# ============================================================
PARAM_NAMES = [
    'ball_restitution',      # 공-공/공-쿠션 반발계수
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


# ============================================================
# 시뮬레이션 평가 함수
# ============================================================
def simulate_strike(trial, params):
    """주어진 파라미터로 헤드리스 PyBullet 시뮬레이션 실행.

    Args:
        trial: dict with keys:
            'cue_start': [x, y] 수구 초기 위치
            'target_start': [x, y] 황구 초기 위치
            'ball2_start': [x, y] 적구 초기 위치
            'strike_angle': float (rad) 타격 각도
            'strike_speed': float (m/s) 도구→공 속도
        params: np.array [6] 물리 파라미터

    Returns:
        sim_finals: dict with 'cue_final', 'target_final', 'ball2_final' — 각 [x, y]
    """
    ball_rest = params[0]
    cushion_rest = params[1]
    roll_fric = params[2]
    lat_fric = params[3]
    speed_scale = params[4]
    y_offset = params[5]

    # 헤드리스 PyBullet 환경 생성
    sim_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=sim_id)
    p.setGravity(0, 0, -9.8, physicsClientId=sim_id)
    p.setTimeStep(1/240, physicsClientId=sim_id)

    # 테이블
    L = MAZE_TABLE_LENGTH
    W = MAZE_TABLE_WIDTH
    H = MAZE_TABLE_SURFACE_HEIGHT
    CX = MAZE_TABLE_CENTER_X
    CY = MAZE_TABLE_CENTER_Y + y_offset  # Y 오프셋 보정
    TH = MAZE_TABLE_HEIGHT
    CH = MAZE_CUSHION_HEIGHT

    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
    center = np.array([CX, CY, H])

    # 테이블 바닥
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2],
                                  physicsClientId=sim_id)
    table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                  basePosition=[center[0], center[1], center[2]],
                                  physicsClientId=sim_id)
    p.changeDynamics(table_id, -1, lateralFriction=lat_fric,
                      restitution=0.5, physicsClientId=sim_id)

    # 쿠션 (4벽)
    top_z = center[2] + TH / 2 + CH / 2
    thickness = 0.03
    cushion_configs = [
        ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
        ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
        ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
    ]
    for pos, half_ext in cushion_configs:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                      physicsClientId=sim_id)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 basePosition=pos, physicsClientId=sim_id)
        p.changeDynamics(cid, -1, restitution=cushion_rest, physicsClientId=sim_id)

    # 공 3개
    def create_ball(pos_2d):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                      physicsClientId=sim_id)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=col,
                                 basePosition=[pos_2d[0], pos_2d[1], ball_h],
                                 physicsClientId=sim_id)
        p.changeDynamics(bid, -1,
                          lateralFriction=lat_fric,
                          restitution=ball_rest,
                          rollingFriction=roll_fric,
                          spinningFriction=0.02,
                          physicsClientId=sim_id)
        return bid

    cue_id = create_ball(trial['cue_start'])
    tgt1_id = create_ball(trial['target_start'])
    tgt2_id = create_ball(trial['ball2_start'])

    # 안정화
    for _ in range(50):
        p.stepSimulation(physicsClientId=sim_id)

    # 타격: 공에 직접 속도 부여
    angle = trial['strike_angle']
    # 도구 속도 → 공 속도 (speed_scale 보정 적용)
    tool_speed = trial['strike_speed']
    ball_speed_gain = (
        (1.0 + np.sqrt(TOOL_HEAD_RESTITUTION * ball_rest))
        * TOOL_HEAD_MASS / (TOOL_HEAD_MASS + MAZE_BALL_MASS)
        * speed_scale
    )
    ball_speed = tool_speed * ball_speed_gain
    vx = ball_speed * np.cos(angle)
    vy = ball_speed * np.sin(angle)
    p.resetBaseVelocity(cue_id, linearVelocity=[vx, vy, 0],
                         physicsClientId=sim_id)

    # 시뮬레이션 실행 (최대 5초)
    for step in range(1200):
        p.stepSimulation(physicsClientId=sim_id)
        if step > 200 and step % 50 == 0:
            speeds = [np.linalg.norm(p.getBaseVelocity(bid, physicsClientId=sim_id)[0][:2])
                      for bid in [cue_id, tgt1_id, tgt2_id]]
            if all(s < 0.005 for s in speeds):
                break

    # 최종 위치
    cue_final = np.array(p.getBasePositionAndOrientation(cue_id,
                          physicsClientId=sim_id)[0][:2])
    tgt1_final = np.array(p.getBasePositionAndOrientation(tgt1_id,
                           physicsClientId=sim_id)[0][:2])
    tgt2_final = np.array(p.getBasePositionAndOrientation(tgt2_id,
                           physicsClientId=sim_id)[0][:2])

    p.disconnect(sim_id)

    return {
        'cue_final': cue_final,
        'target_final': tgt1_final,
        'ball2_final': tgt2_final,
    }


def evaluate(params, trials):
    """전체 trial에 대한 총 Loss 계산.

    Loss = Σ_i ( ||cue_sim - cue_actual||² + ||t1_sim - t1_actual||² + ||t2_sim - t2_actual||² )
    """
    total_loss = 0.0
    for trial in trials:
        sim_result = simulate_strike(trial, params)
        for key in ['cue_final', 'target_final', 'ball2_final']:
            if key in trial and trial[key] is not None:
                diff = sim_result[key] - np.array(trial[key])
                total_loss += np.dot(diff, diff)
    return total_loss


# ============================================================
# 데이터 수집 (실제 로봇+카메라)
# ============================================================
def collect_calibration_data(num_trials=5, save_path='calibration_trials.npz'):
    """실제 로봇에서 타격 데이터를 수집.

    각 trial: 초기 공 위치 + 타격 각도/속도 + 최종 공 위치
    카메라로 공 위치를 전/후로 촬영하여 기록.

    ※ 이 함수는 실제 로봇+카메라가 연결되어 있어야 합니다.
    """
    try:
        from project.real_env_to_pybullet import detect_balls
    except ImportError:
        print("[ERROR] RealSense 카메라 또는 real_env_to_pybullet 모듈을 찾을 수 없습니다.")
        return None

    print("=" * 60)
    print(f"  캘리브레이션 데이터 수집 ({num_trials}회 타격)")
    print("=" * 60)
    print()
    print("  절차:")
    print("  1. 3공을 테이블 위에 배치")
    print("  2. 카메라로 초기 위치 촬영")
    print("  3. 수동으로 큐대(혹은 로봇)로 수구 타격")
    print("  4. 공이 멈춘 후 카메라로 최종 위치 촬영")
    print("  5. 타격 각도/속도 입력 (추정치)")
    print()

    trials = []
    for i in range(num_trials):
        print(f"\n--- Trial {i+1}/{num_trials} ---")

        # 초기 위치 촬영
        input("  공을 배치한 후 Enter를 누르세요 (초기 위치 촬영)...")
        try:
            cue_start, target_start, ball2_start = detect_balls()
            print(f"  초기: cue={cue_start[:2]}, t1={target_start[:2]}, t2={ball2_start[:2]}")
        except Exception as e:
            print(f"  [ERROR] 공 검출 실패: {e}")
            continue

        # 타격 정보 입력
        angle_deg = float(input("  타격 각도 (도, 0=+x방향): "))
        speed = float(input("  타격 속도 (m/s, 예: 1.0): "))

        # 최종 위치 촬영
        input("  공이 멈춘 후 Enter를 누르세요 (최종 위치 촬영)...")
        try:
            cue_final, target_final, ball2_final = detect_balls()
            print(f"  최종: cue={cue_final[:2]}, t1={target_final[:2]}, t2={ball2_final[:2]}")
        except Exception as e:
            print(f"  [ERROR] 공 검출 실패: {e}")
            continue

        trials.append({
            'cue_start': np.array(cue_start[:2]),
            'target_start': np.array(target_start[:2]),
            'ball2_start': np.array(ball2_start[:2]),
            'strike_angle': np.radians(angle_deg),
            'strike_speed': speed,
            'cue_final': np.array(cue_final[:2]),
            'target_final': np.array(target_final[:2]),
            'ball2_final': np.array(ball2_final[:2]),
        })
        print(f"  Trial {i+1} 저장 완료")

    if trials:
        np.savez(save_path, trials=trials)
        print(f"\n  {len(trials)}개 trial 저장: {save_path}")
    return trials


# ============================================================
# 최적화
# ============================================================
def optimize_parameters(trials, method='nelder-mead'):
    """Nelder-Mead 또는 differential_evolution으로 파라미터 최적화.

    Args:
        trials: list of trial dicts
        method: 'nelder-mead' 또는 'differential_evolution'

    Returns:
        optimal_params: np.array [6]
    """
    from scipy.optimize import minimize, differential_evolution

    x0 = PARAM_DEFAULTS.copy()
    print(f"\n  초기 파라미터: {dict(zip(PARAM_NAMES, x0))}")

    initial_loss = evaluate(x0, trials)
    print(f"  초기 Loss: {initial_loss:.6f}")

    if method == 'nelder-mead':
        result = minimize(
            evaluate, x0, args=(trials,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                     'disp': True}
        )
    elif method == 'differential_evolution':
        result = differential_evolution(
            evaluate, PARAM_BOUNDS, args=(trials,),
            maxiter=200, tol=1e-6, disp=True,
            seed=42
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    optimal = result.x
    final_loss = result.fun
    print(f"\n  최적 파라미터:")
    for name, val, default in zip(PARAM_NAMES, optimal, PARAM_DEFAULTS):
        change = (val - default) / default * 100 if default != 0 else 0
        print(f"    {name}: {default:.4f} → {val:.4f} ({change:+.1f}%)")
    print(f"  최종 Loss: {final_loss:.6f} (초기: {initial_loss:.6f})")

    return optimal


def save_calibration(params, save_path='calibration_result_physics.npz'):
    """최적 파라미터를 파일로 저장."""
    param_dict = dict(zip(PARAM_NAMES, params))
    np.savez(save_path, **param_dict)
    print(f"\n  캘리브레이션 결과 저장: {save_path}")
    print(f"  config.py에서 자동 로드하려면:")
    print(f"    CALIB_PATH = '{save_path}'")
    print(f"    if os.path.exists(CALIB_PATH):")
    print(f"        calib = np.load(CALIB_PATH)")
    print(f"        MAZE_BALL_RESTITUTION = float(calib['ball_restitution'])")
    print(f"        # ... etc")


def load_calibration(path='calibration_result_physics.npz'):
    """저장된 캘리브레이션 결과 로드.

    Returns:
        dict of parameter name → value, or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None
    calib = np.load(path)
    result = {}
    for name in PARAM_NAMES:
        if name in calib:
            result[name] = float(calib[name])
    return result


# ============================================================
# 메인
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='물리 파라미터 자동 캘리브레이션')
    parser.add_argument('--collect', action='store_true',
                        help='실제 로봇으로 타격 데이터 수집')
    parser.add_argument('--optimize', action='store_true',
                        help='저장된 데이터로 파라미터 최적화')
    parser.add_argument('--num-trials', type=int, default=5,
                        help='수집할 타격 횟수 (기본: 5)')
    parser.add_argument('--data', type=str, default='calibration_trials.npz',
                        help='타격 데이터 파일 경로')
    parser.add_argument('--method', type=str, default='nelder-mead',
                        choices=['nelder-mead', 'differential_evolution'],
                        help='최적화 알고리즘')
    parser.add_argument('--output', type=str, default='calibration_result_physics.npz',
                        help='캘리브레이션 결과 저장 경로')
    args = parser.parse_args()

    if args.collect:
        trials = collect_calibration_data(args.num_trials, args.data)
        if trials and args.optimize:
            optimal = optimize_parameters(trials, args.method)
            save_calibration(optimal, args.output)

    elif args.optimize:
        if not os.path.exists(args.data):
            print(f"[ERROR] 데이터 파일 없음: {args.data}")
            sys.exit(1)
        data = np.load(args.data, allow_pickle=True)
        trials = list(data['trials'])
        optimal = optimize_parameters(trials, args.method)
        save_calibration(optimal, args.output)

    else:
        # 테스트: 기본 파라미터로 시뮬 평가 함수 확인
        print("  테스트 모드: 기본 파라미터로 시뮬 평가")
        test_trial = {
            'cue_start': np.array([0.345, 0.1725]),
            'target_start': np.array([0.345, 0.2888]),
            'ball2_start': np.array([0.45, 0.25]),
            'strike_angle': np.radians(97.0),
            'strike_speed': MAX_TOOL_SPEED,
            'cue_final': np.array([0.40, 0.30]),     # 가상 관측치
            'target_final': np.array([0.35, 0.35]),
            'ball2_final': np.array([0.50, 0.20]),
        }
        result = simulate_strike(test_trial, PARAM_DEFAULTS)
        print(f"  시뮬 결과: cue={result['cue_final']}, "
              f"t1={result['target_final']}, t2={result['ball2_final']}")
        loss = evaluate(PARAM_DEFAULTS, [test_trial])
        print(f"  Loss: {loss:.6f}")
        print("\n  실제 캘리브레이션:")
        print("    python calibration_loop.py --collect --optimize --num-trials 5")
