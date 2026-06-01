# -*- coding: utf-8 -*-
"""
MoveL 기반 캘리브레이션 (Approach / Strike 분리)
==============================================
실기 타격은 movej 곡선이 아니라 movel 직선으로 재생합니다.

  Phase 1: 좌표 — +x / +y 약타격, 키보드로 offset 보정
  Phase 2: 물리 — 3쿠션 플래너 + 카메라 전/후 → Nelder-Mead

사용법:
  # 한 세션 (Approach → Enter → Strike)
  python calibration_loop_movel.py --phase all --real-step full

  # 실행 분리 (권장)
  python calibration_loop_movel.py --phase position --real-step approach --axis x
  python calibration_loop_movel.py --phase position --real-step strike
  python calibration_loop_movel.py --phase physics --real-step approach
  python calibration_loop_movel.py --real-step strike --phase physics

  --allow-auto-strike : full 모드에서 Enter 생략 (비권장)
  자세한 설명: docs/캘리브레이션_MoveL_분리.md
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project.config import *
from project.real_movel_player import (
    DEFAULT_PLAN_FILE,
    compute_fk_offset_mm,
    execute_movel_split,
    load_shot_plan,
    movej_home,
    plan_cushion_shot,
    plan_linear_shot,
    run_real_approach_only,
    run_real_strike_only,
    run_sim_trajectory,
    sync_sim_from_real,
)

PARAM_NAMES = [
    'ball_restitution', 'cushion_restitution', 'rolling_friction',
    'lateral_friction', 'speed_gain_scale', 'table_y_offset',
]
PARAM_DEFAULTS = np.array([
    MAZE_BALL_RESTITUTION, MAZE_CUSHION_RESTITUTION,
    MAZE_BALL_ROLLING_FRICTION, MAZE_BALL_FRICTION,
    BALL_SPEED_GAIN_SCALE, 0.0,
])
PARAM_BOUNDS = [
    (0.5, 1.0), (0.5, 1.0), (0.001, 0.05), (0.05, 0.5), (0.7, 1.5), (-0.02, 0.02),
]
CALIB_FILE = 'calibration_result_physics.npz'
POSITION_CALIB_FILE = 'calibration_position_offset.json'
TRIALS_FILE = 'calibration_trials.npz'
PLAN_FILE = DEFAULT_PLAN_FILE


def _ball_surface_z():
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    return H + TH / 2 + MAZE_BALL_RADIUS + 0.001


def _vision_to_sim_position(pos):
    p = np.asarray(pos, dtype=float).reshape(-1)
    z = float(p[2]) if p.size >= 3 else _ball_surface_z()
    return [float(p[0]), float(p[1]), z]


def _detect_balls_vision():
    from project.real_env_to_pybullet import detect_balls
    cue, tgt, b2 = detect_balls()
    return (
        _vision_to_sim_position(cue),
        _vision_to_sim_position(tgt),
        _vision_to_sim_position(b2),
    )


def _setup_env_from_vision(env, num_obstacles=0):
    print("  [VISION] 공 검출...")
    cue_pos, target_pos, ball2_pos = _detect_balls_vision()
    if getattr(env, 'cue_ball_id', None) is not None:
        env.reset_balls(cue_pos=cue_pos, target_pos=target_pos, ball2_pos=ball2_pos)
    else:
        env.setup(
            cue_pos=cue_pos, target_pos=target_pos, ball2_pos=ball2_pos,
            num_obstacles=num_obstacles, obstacle_positions=[],
        )
    return cue_pos, target_pos, ball2_pos


def load_position_offset():
    if os.path.exists(POSITION_CALIB_FILE):
        with open(POSITION_CALIB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'x': 0.0, 'y': 0.0}


def save_position_offset(offset):
    with open(POSITION_CALIB_FILE, 'w', encoding='utf-8') as f:
        json.dump(offset, f, indent=2)
    print(f"  오프셋 저장: {POSITION_CALIB_FILE}")


def _return_home(indy, pb):
    print("  [REAL] Home (movej)")
    movej_home(indy, pb)


def _execute_real_shot(indy, pb, traj, phases, speed, fk_offset_mm, real_step,
                       plan_file, confirm_strike, meta=None):
    """real_step: full | approach | strike"""
    if real_step == 'approach':
        run_real_approach_only(
            indy, pb, traj, phases, speed, fk_offset_mm, plan_file, meta=meta)
        return None
    if real_step == 'strike':
        run_real_strike_only(indy, pb, plan_file, speed=speed, wait_start=True)
        return True
    execute_movel_split(
        indy, pb, traj, phases, speed,
        confirm_before_strike=confirm_strike,
        fk_offset_mm=fk_offset_mm,
    )
    return True


def _position_keyboard(offset, axis, offset_size):
    print("\n공 진행 방향: [l]왼쪽 [r]오른쪽 [g]직진 [m]miss")
    cmd = input("입력 > ").strip().lower()
    done = False
    if axis == 'x':
        if cmd == 'l':
            offset['y'] -= offset_size
        elif cmd == 'r':
            offset['y'] += offset_size
        elif cmd == 'g':
            done = True
        elif cmd == 'm':
            cmd2 = input("  빗맞음 [l]/[r] > ").strip().lower()
            offset['y'] += 0.005 if cmd2 == 'l' else (-0.005 if cmd2 == 'r' else 0)
    else:
        if cmd == 'l':
            offset['x'] -= offset_size
        elif cmd == 'r':
            offset['x'] += offset_size
        elif cmd == 'g':
            done = True
        elif cmd == 'm':
            cmd2 = input("  빗맞음 [l]/[r] > ").strip().lower()
            offset['x'] += 0.005 if cmd2 == 'l' else (-0.005 if cmd2 == 'r' else 0)
    return offset, done


def run_position_approach_once(indy, pb, env, ik, axis, fk_offset_mm, plan_file):
    """분리 실행 1단계: 비전 → 시뮬 → Approach(movel) → 정지·계획 저장."""
    offset = load_position_offset()
    strike_dir = np.array([1.0, 0.0, 0.0]) if axis == 'x' else np.array([0.0, 1.0, 0.0])
    cue_pos, _, _ = _setup_env_from_vision(env, num_obstacles=0)
    sync_sim_from_real(indy, pb, verbose=True)
    traj, phases, speed = plan_linear_shot(pb, env, ik, cue_pos, strike_dir, strike_speed=0.1)
    print(f"  [SIM] {'+x' if axis == 'x' else '+y'} 약타격 미리보기...")
    run_sim_trajectory(pb, ik, traj, phases)
    time.sleep(0.3)
    run_real_approach_only(
        indy, pb, traj, phases, speed, fk_offset_mm, plan_file,
        meta={'calib': 'position', 'axis': axis, 'offset': offset})
    print(f"\n  다음 명령:\n"
          f"    python calibration_loop_movel.py --phase position "
          f"--real-step strike --plan-file {plan_file}")
    return offset


def run_position_strike_once(indy, pb, plan_file):
    """분리 실행 2단계: Enter(START) → Strike(movel) → 키보드 보정."""
    loaded = load_shot_plan(plan_file)
    meta = json.loads(loaded.get('meta_json', '{}'))
    axis = meta.get('axis', 'x')
    offset = meta.get('offset', load_position_offset())
    offset_size = 0.003

    run_real_strike_only(indy, pb, plan_file, speed=loaded['speed'], wait_start=True)
    offset, done = _position_keyboard(offset, axis, offset_size)
    save_position_offset(offset)
    _return_home(indy, pb)
    if done:
        print(f"  축 '{axis}' 보정 완료 (g)")
    return offset, done


def _run_position_axis(indy, pb, env, ik, offset, axis, fk_offset_mm, confirm_strike,
                       real_step='full', plan_file=PLAN_FILE):
    """axis: 'x' (+x shot → y offset) or 'y' (+y shot → x offset)."""
    offset_size = 0.003
    strike_dir = np.array([1.0, 0.0, 0.0]) if axis == 'x' else np.array([0.0, 1.0, 0.0])
    axis_name = '+x → y보정' if axis == 'x' else '+y → x보정'

    while abs(offset_size) > 1e-6:
        try:
            cue_pos, _, _ = _setup_env_from_vision(env, num_obstacles=0)
        except Exception as e:
            print(f"  [ERROR] 비전: {e}")
            continue

        sync_sim_from_real(indy, pb, verbose=True)
        traj, phases, speed = plan_linear_shot(
            pb, env, ik, cue_pos, strike_dir, strike_speed=0.1)

        print(f"  [SIM] {axis_name} 약타격 미리보기...")
        run_sim_trajectory(pb, ik, traj, phases)
        time.sleep(0.3)

        meta = {'calib': 'position', 'axis': axis, 'offset': offset}
        try:
            if real_step == 'approach':
                run_real_approach_only(
                    indy, pb, traj, phases, speed, fk_offset_mm, plan_file, meta=meta)
                print(f"\n  Ready 정지. Strike:\n"
                      f"    python calibration_loop_movel.py --phase position "
                      f"--real-step strike --plan-file {plan_file}")
                return offset
            _execute_real_shot(
                indy, pb, traj, phases, speed, fk_offset_mm, real_step, plan_file,
                confirm_strike, meta=meta)
        except Exception as e:
            print(f"  [ERROR] 실기: {e}")
            try:
                indy.recover()
            except Exception:
                pass
            _return_home(indy, pb)
            continue

        if real_step == 'strike':
            offset, done = _position_keyboard(offset, axis, offset_size)
            save_position_offset(offset)
            _return_home(indy, pb)
            if done:
                return offset
            continue

        offset, done = _position_keyboard(offset, axis, offset_size)
        if done:
            save_position_offset(offset)
            _return_home(indy, pb)
            return offset
        save_position_offset(offset)
        _return_home(indy, pb)

    return offset


def run_position_calibration(indy, pb, env, ik, fk_offset_mm, confirm_strike=True,
                             real_step='full', plan_file=PLAN_FILE, axis=None):
    offset = load_position_offset()
    print(f"\n{'='*60}")
    print("  Phase 1: 좌표 캘리브레이션 (MoveL)")
    print(f"  오프셋 x={offset['x']:.4f} y={offset['y']:.4f}  real_step={real_step}")
    print(f"{'='*60}")

    if real_step == 'approach':
        if axis not in ('x', 'y'):
            raise SystemExit("--real-step approach 는 --axis x 또는 y 필요")
        return run_position_approach_once(indy, pb, env, ik, axis, fk_offset_mm, plan_file)
    if real_step == 'strike':
        return run_position_strike_once(indy, pb, plan_file)

    offset = _run_position_axis(
        indy, pb, env, ik, offset, 'x', fk_offset_mm, confirm_strike,
        real_step, plan_file)
    offset = _run_position_axis(
        indy, pb, env, ik, offset, 'y', fk_offset_mm, confirm_strike,
        real_step, plan_file)
    print(f"\n  Phase 1 완료: x={offset['x']:.4f}, y={offset['y']:.4f}")
    return offset


def run_physics_strike_once(indy, pb, plan_file):
    """분리 실행: START → Strike → 카메라 관측 → trial 저장."""
    loaded = load_shot_plan(plan_file)
    meta = json.loads(loaded.get('meta_json', '{}'))
    if meta.get('calib') != 'physics':
        print("  [WARN] plan meta가 physics trial이 아닐 수 있음")

    run_real_strike_only(indy, pb, plan_file, speed=loaded['speed'], wait_start=True)
    try:
        cue_fin, tgt_fin, b2_fin = _observe_balls_after_strike()
    except Exception as e:
        print(f"  [ERROR] 타격 후 검출: {e}")
        _return_home(indy, pb)
        return None

    trial = {
        'cue_start': np.array(meta['cue_start']),
        'target_start': np.array(meta['target_start']),
        'ball2_start': np.array(meta['ball2_start']),
        'strike_angle': float(meta['strike_angle']),
        'strike_speed': float(meta['strike_speed']),
        'cue_final': cue_fin,
        'target_final': tgt_fin,
        'ball2_final': b2_fin,
    }
    trials = []
    if os.path.exists(TRIALS_FILE):
        trials = list(np.load(TRIALS_FILE, allow_pickle=True)['trials'])
    trials.append(trial)
    np.savez(TRIALS_FILE, trials=np.array(trials, dtype=object))
    print(f"  Trial 저장 → {TRIALS_FILE} ({len(trials)}개)")
    _return_home(indy, pb)
    return trial


def _observe_balls_after_strike():
    from project.real_env_to_pybullet import detect_balls
    time.sleep(2.0)
    cue, tgt, b2 = detect_balls()
    return (
        np.array(cue[:2], dtype=float),
        np.array(tgt[:2], dtype=float),
        np.array(b2[:2], dtype=float),
    )


def run_physics_calibration(indy, pb, env, ik, num_trials, fk_offset_mm,
                            allow_auto_strike=False, real_step='full',
                            plan_file=PLAN_FILE):
    trials = []
    print(f"\n{'='*60}")
    print(f"  Phase 2: 물리 캘리브레이션 (MoveL) × {num_trials}")
    print(f"{'='*60}")

    for n in range(num_trials):
        print(f"\n--- Trial {n + 1}/{num_trials} ---")
        try:
            cue_pos, target_pos, ball2_pos = _setup_env_from_vision(env, num_obstacles=0)
        except Exception as e:
            print(f"  [ERROR] 비전: {e}")
            continue

        sync_sim_from_real(indy, pb)
        try:
            traj, phases, speed, best = plan_cushion_shot(
                pb, env, ik, cue_pos, target_pos, ball2_pos)
        except Exception as e:
            print(f"  [ERROR] 플래너: {e}")
            continue

        angle_deg = float(np.degrees(np.arctan2(best['strike_dir'][1], best['strike_dir'][0])))
        angle = angle_deg
        cue0 = np.array(cue_pos[:2])
        tgt0 = np.array(target_pos[:2])
        b20 = np.array(ball2_pos[:2])

        print(f"  플랜: angle={angle:.1f}° speed={speed:.3f}")
        run_sim_trajectory(pb, ik, traj, phases)
        time.sleep(0.3)

        meta = {
            'calib': 'physics',
            'trial': n + 1,
            'cue_start': cue0.tolist(),
            'target_start': tgt0.tolist(),
            'ball2_start': b20.tolist(),
            'strike_angle': float(np.radians(angle_deg)),
            'strike_speed': float(speed),
        }
        try:
            if real_step == 'approach':
                run_real_approach_only(
                    indy, pb, traj, phases, speed, fk_offset_mm, plan_file, meta=meta)
                print(f"\n  Trial {n+1} Approach 완료. Strike:\n"
                      f"    python calibration_loop_movel.py --phase physics "
                      f"--real-step strike --plan-file {plan_file}")
                return trials
            _execute_real_shot(
                indy, pb, traj, phases, speed, fk_offset_mm, 'full', plan_file,
                confirm_strike=not allow_auto_strike, meta=meta)
        except Exception as e:
            print(f"  [ERROR] 실기: {e}")
            _return_home(indy, pb)
            continue

        try:
            cue_fin, tgt_fin, b2_fin = _observe_balls_after_strike()
        except Exception as e:
            print(f"  [WARN] 타격 후 검출 실패: {e}")
            continue

        trial = {
            'cue_start': cue0, 'target_start': tgt0, 'ball2_start': b20,
            'strike_angle': np.radians(angle_deg), 'strike_speed': speed,
            'cue_final': cue_fin, 'target_final': tgt_fin, 'ball2_final': b2_fin,
        }
        trials.append(trial)
        np.savez(TRIALS_FILE, trials=np.array(trials, dtype=object))
        print(f"  Trial 저장 ({len(trials)}개)")
        _return_home(indy, pb)

    if not trials:
        print("  수집된 trial 없음.")
        return None

    optimal = optimize_parameters(trials)
    save_calibration(optimal)
    return optimal


def simulate_strike(trial, params):
    ball_rest, cushion_rest = params[0], params[1]
    roll_fric, lat_fric = params[2], params[3]
    speed_scale, y_offset = params[4], params[5]

    sim_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=sim_id)
    p.setGravity(0, 0, -9.8, physicsClientId=sim_id)
    p.setTimeStep(1 / 240, physicsClientId=sim_id)

    L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
    H, TH, CH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT, MAZE_CUSHION_HEIGHT
    CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y + y_offset
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
    center = np.array([CX, CY, H])

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L / 2, W / 2, TH / 2],
                                  physicsClientId=sim_id)
    tid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             basePosition=list(center), physicsClientId=sim_id)
    p.changeDynamics(tid, -1, lateralFriction=lat_fric, restitution=0.5,
                      physicsClientId=sim_id)

    top_z = center[2] + TH / 2 + CH / 2
    thickness = 0.03
    for pos, he in [
        ([center[0], center[1] + W / 2 + thickness / 2, top_z], [L / 2, thickness / 2, CH / 2]),
        ([center[0], center[1] - W / 2 - thickness / 2, top_z], [L / 2, thickness / 2, CH / 2]),
        ([center[0] - L / 2 - thickness / 2, center[1], top_z], [thickness / 2, W / 2, CH / 2]),
        ([center[0] + L / 2 + thickness / 2, center[1], top_z], [thickness / 2, W / 2, CH / 2]),
    ]:
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=sim_id)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                 basePosition=pos, physicsClientId=sim_id)
        p.changeDynamics(cid, -1, restitution=cushion_rest, physicsClientId=sim_id)

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

    angle = trial['strike_angle']
    ball_speed_gain = ((1.0 + np.sqrt(TOOL_HEAD_RESTITUTION * ball_rest))
                       * TOOL_HEAD_MASS / (TOOL_HEAD_MASS + MAZE_BALL_MASS) * speed_scale)
    ball_speed = trial['strike_speed'] * ball_speed_gain
    p.resetBaseVelocity(cue_id,
                         linearVelocity=[ball_speed * np.cos(angle),
                                         ball_speed * np.sin(angle), 0],
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
    total = 0.0
    for trial in trials:
        sim = simulate_strike(trial, params)
        for key in ['cue_final', 'target_final', 'ball2_final']:
            if key in trial and trial[key] is not None:
                total += np.sum((sim[key] - np.array(trial[key])) ** 2)
    return total


def optimize_parameters(trials, method='nelder-mead'):
    from scipy.optimize import minimize, differential_evolution

    x0 = PARAM_DEFAULTS.copy()
    print(f"\n  초기: {dict(zip(PARAM_NAMES, x0))}")
    initial_loss = evaluate(x0, trials)
    print(f"  초기 Loss: {initial_loss:.6f}")

    if method == 'nelder-mead':
        result = minimize(evaluate, x0, args=(trials,), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': True})
    else:
        result = differential_evolution(evaluate, PARAM_BOUNDS, args=(trials,),
                                       maxiter=200, tol=1e-6, disp=True, seed=42)

    optimal = result.x
    print("\n  최적 파라미터:")
    for name, val, default in zip(PARAM_NAMES, optimal, PARAM_DEFAULTS):
        change = (val - default) / default * 100 if default != 0 else 0
        print(f"    {name}: {default:.4f} → {val:.4f} ({change:+.1f}%)")
    print(f"  Loss: {result.fun:.6f} (초기 {initial_loss:.6f})")
    return optimal


def save_calibration(params, path=None):
    path = path or CALIB_FILE
    np.savez(path, **dict(zip(PARAM_NAMES, params)))
    print(f"  저장: {path}")


def _boost_robot_pd(pb):
    robot = pb.my_robot

    def _boosted():
        qddot = (robot._qddot_des + 5000 * (robot._q_des - robot._q)
                 + 200 * (robot._qdot_des - robot._qdot))
        robot._tau = robot._M @ qddot + robot._c + robot._g

    robot._compute_torque_input = _boosted


def main():
    parser = argparse.ArgumentParser(description='MoveL 캘리브레이션')
    parser.add_argument('--phase', choices=['position', 'physics', 'all'], default='all')
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--robot-ip', default='192.168.0.13')
    parser.add_argument('--real-step', choices=['full', 'approach', 'strike'],
                        default='full',
                        help='full=한 세션 | approach=접근만 | strike=START 후 타격만')
    parser.add_argument('--plan-file', default=PLAN_FILE,
                        help=f'분리 실행 시 계획 npz (기본 {PLAN_FILE})')
    parser.add_argument('--axis', choices=['x', 'y'],
                        help='position + approach 일 때 타격 축 (+x 또는 +y)')
    parser.add_argument('--allow-auto-strike', action='store_true',
                        help='full 모드: Enter 없이 Strike (비권장)')
    parser.add_argument('--skip-fk-offset', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--optimize-only', action='store_true')
    parser.add_argument('--data', default=TRIALS_FILE)
    args = parser.parse_args()

    if args.test:
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
        r = simulate_strike(test_trial, PARAM_DEFAULTS)
        print(f"  sim cue={r['cue_final']}")
        print(f"  loss={evaluate(PARAM_DEFAULTS, [test_trial]):.6f}")
        return

    if args.optimize_only:
        data = np.load(args.data, allow_pickle=True)
        save_calibration(optimize_parameters(list(data['trials'])))
        return

    from src.core.pybullet_core import PybulletCore
    from neuromeka import IndyDCP3
    from project.environment.maze_env import MazeEnvironment
    from project.ik_solver import IKSolver

    pb = PybulletCore()
    pb.connect(robot_name='indy7_v2', joint_limit=True, constraint_visualization=False)
    ik = IKSolver(pb.my_robot.pinModel, gain=IK_GAIN, damping=IK_DAMPING)
    indy = IndyDCP3(robot_ip=args.robot_ip, index=0)
    print(f"  로봇: {args.robot_ip}")

    env = MazeEnvironment(pb.ClientId)
    try:
        _setup_env_from_vision(env, num_obstacles=0)
    except Exception as e:
        print(f"  [ERROR] 초기 비전: {e}")
        pb.disconnect()
        sys.exit(1)

    robot_id = pb.my_robot.robotId
    ee_link = pb.my_robot.RobotEEJointIdx[-1]
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    _boost_robot_pd(pb)

    sync_sim_from_real(indy, pb, verbose=True)
    fk_offset = np.zeros(3) if args.skip_fk_offset else compute_fk_offset_mm(indy, pb)
    print(f"  FK offset(mm): {fk_offset}")

    if args.real_step != 'strike':
        movej_home(indy, pb)
    else:
        sync_sim_from_real(indy, pb, verbose=True)
        print("  [strike 모드] 로봇 Ready 유지 — Approach 직후 자세에서 실행")

    print("\n  [안내] Approach(movel) 정지 → [Enter]=START → Strike(movel)")
    print("         분리: --real-step approach 후 --real-step strike\n")

    confirm = not args.allow_auto_strike
    if args.phase in ('position', 'all'):
        run_position_calibration(
            indy, pb, env, ik, fk_offset, confirm_strike=confirm,
            real_step=args.real_step, plan_file=args.plan_file, axis=args.axis)

    if args.phase in ('physics', 'all'):
        if args.real_step == 'strike':
            if args.phase == 'all':
                print("  [WARN] --phase all + strike 는 physics strike만 실행합니다.")
            run_physics_strike_once(indy, pb, args.plan_file)
        else:
            run_physics_calibration(
                indy, pb, env, ik, args.num_trials, fk_offset,
                allow_auto_strike=args.allow_auto_strike,
                real_step=args.real_step, plan_file=args.plan_file)

    pb.disconnect()
    print("\n  완료.")


if __name__ == '__main__':
    main()
