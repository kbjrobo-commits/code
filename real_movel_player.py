"""실기 MoveL: Approach / Strike 분리 재생 (캘리브레이션·테스트 공용)."""
import json
import os
import time

import numpy as np
import pybullet as p

from project.config import *

MOVEL_MIN_DIST_MM = 3.0
APPROACH_MOVEL_STEP = 100
APPROACH_VEL = 20
APPROACH_ACC = 50
ALIGN_VEL = 10
ALIGN_ACC = 30


def SE3_to_p6(T, offset_mm=None):
    from src.utils import Rot2eul
    off = np.zeros(3) if offset_mm is None else np.asarray(offset_mm).reshape(3)
    p6 = np.zeros(6)
    p6[0:3] = 1000.0 * np.asarray(T)[0:3, 3] - off
    p6[3:6] = Rot2eul(np.asarray(T)[0:3, 0:3], seq='XYZ', degree=True)
    return p6


def wait_indy(indy, timeout=60, sync_sim=False, pb=None):
    time.sleep(0.2)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        if indy.get_motion_data()['is_in_motion']:
            break
        time.sleep(0.05)
    while time.time() - t0 < timeout:
        if sync_sim and pb is not None:
            sync_sim_from_real(indy, pb)
        if not indy.get_motion_data()['is_in_motion']:
            break
        time.sleep(0.05)
    if sync_sim and pb is not None:
        sync_sim_from_real(indy, pb)


def sync_sim_from_real(indy, pb, verbose=False):
    q_deg = list(np.asarray(indy.get_control_data()['q']).flatten())
    q_rad = np.deg2rad(q_deg).reshape(-1, 1)
    robot = pb.my_robot
    client = pb.ClientId
    for i, jidx in enumerate(robot.RobotMovableJointIdx):
        p.resetJointState(
            robot.robotId, jidx, float(q_rad[i, 0]), 0.0, physicsClientId=client)
    robot._q = q_rad.copy()
    robot._qdot = np.zeros_like(q_rad)
    robot.set_desired_joint_pos(q_rad)
    robot._get_robot_states()
    pb.MoveRobot(q_deg, degree=True)
    if verbose:
        T = robot.pinModel.FK(robot.q)
        p_mm = 1000 * T[0:3, 3]
        print(f"  [SYNC] EE(mm)=[{p_mm[0]:.1f}, {p_mm[1]:.1f}, {p_mm[2]:.1f}]")
    return q_deg


def movej_home(indy, pb, wait=True):
    indy.movej(list(HOME_Q_DEG), vel_ratio=30, acc_ratio=50)
    if wait:
        wait_indy(indy, sync_sim=True, pb=pb)
    sync_sim_from_real(indy, pb)


def verify_movel(indy, p_target, tol_mm=5.0):
    p_now = indy.get_control_data()['p']
    err = float(np.linalg.norm(np.array(p_now[:3]) - np.array(p_target[:3])))
    if err > tol_mm:
        print(f"  [WARN] movel 미도달 err={err:.1f}mm, 재시도...")
        indy.movel(list(p_target), vel_ratio=50, acc_ratio=100)
        wait_indy(indy)
    return err


def compute_fk_offset_mm(indy, pb):
    movej_home(indy, pb)
    T_pin = pb.my_robot.pinModel.FK(np.deg2rad(HOME_Q_DEG))
    p_real = indy.get_control_data()['p']
    return 1000.0 * T_pin[0:3, 3] - np.array(p_real[:3])


def strike_target_from_ready(p_ready, T_ready, T_follow):
    p_strike = np.array(p_ready, dtype=float).copy()
    delta_m = np.asarray(T_follow)[0:3, 3] - np.asarray(T_ready)[0:3, 3]
    p_strike[0:3] = p_ready[0:3] + 1000.0 * delta_m
    p_strike[3:6] = p_ready[3:6]
    return p_strike, float(np.linalg.norm(1000.0 * delta_m))


def run_approach_movel(indy, pb, trajectory, phases, fk_offset_mm=None):
    """Approach + Align (movel). 반환: plan dict (p_ready_cmd, p_strike_cmd)."""
    traj = np.asarray(trajectory)
    off = fk_offset_mm
    a0, a1 = phases['approach']
    follow_end = phases['follow'][1]

    sync_sim_from_real(indy, pb)
    wps = list(range(a0, a1, APPROACH_MOVEL_STEP))
    if not wps or wps[-1] != a1 - 1:
        wps.append(a1 - 1)

    print(f"  [REAL] Approach movel ({len(wps)} pts)...")
    for idx in wps:
        indy.movel(list(SE3_to_p6(traj[idx], off)), vel_ratio=APPROACH_VEL, acc_ratio=APPROACH_ACC)
        wait_indy(indy, sync_sim=True, pb=pb)

    T_ready = traj[a1 - 1]
    p_ready = SE3_to_p6(T_ready, off)
    print("  [REAL] Align ready...")
    time.sleep(0.3)
    indy.movel(list(p_ready), vel_ratio=ALIGN_VEL, acc_ratio=ALIGN_ACC)
    wait_indy(indy, sync_sim=True, pb=pb)
    verify_movel(indy, p_ready, tol_mm=5.0)

    T_follow = traj[min(follow_end - 1, len(traj) - 1)]
    p_strike, planner_mm = strike_target_from_ready(p_ready, T_ready, T_follow)
    p_now = indy.get_control_data()['p']
    print(f"  [REAL] Ready. TCP=[{p_now[0]:.0f},{p_now[1]:.0f},{p_now[2]:.0f}] "
          f"strike Δ={planner_mm:.0f}mm")
    return {
        'p_ready_cmd': np.array(p_ready),
        'p_strike_cmd': np.array(p_strike),
        'planner_strike_mm': planner_mm,
    }


def run_strike_movel(indy, pb, p_strike_cmd, p_ready_cmd=None,
                    strike_vel_ratio=None, speed=None, fk_offset_mm=None,
                    do_retract=True, sync_pb=True):
    """단일 MoveL 직선 타격."""
    p_strike = np.array(p_strike_cmd, dtype=float)
    if strike_vel_ratio is None:
        sp = speed if speed is not None else MAX_TOOL_SPEED
        strike_vel_ratio = int(np.clip(sp / MAX_TOOL_SPEED * 100, 30, 100))

    p_now = indy.get_control_data()['p']
    strike_dist = float(np.linalg.norm(p_strike[:3] - np.array(p_now[:3])))

    if strike_dist < MOVEL_MIN_DIST_MM and p_ready_cmd is not None:
        delta = p_strike[:3] - np.array(p_ready_cmd[:3])
        if np.linalg.norm(delta) >= MOVEL_MIN_DIST_MM:
            p_strike = np.array(p_now, dtype=float)
            p_strike[0:3] = p_now[:3] + delta
            p_strike[3:6] = p_now[3:6]
            strike_dist = float(np.linalg.norm(delta))
            print(f"  [REAL] Strike 보정: TCP+Δ {strike_dist:.0f}mm")

    if strike_dist < MOVEL_MIN_DIST_MM:
        print(f"  [REAL] Strike skip: {strike_dist:.1f}mm < {MOVEL_MIN_DIST_MM}mm")
        return False

    print(f"  [REAL] Strike movel vel={strike_vel_ratio}% dist={strike_dist:.0f}mm")
    indy.movel(list(p_strike), vel_ratio=strike_vel_ratio, acc_ratio=100)
    wait_indy(indy, sync_sim=sync_pb, pb=pb if sync_pb else None)
    verify_movel(indy, p_strike, tol_mm=8.0)

    if do_retract:
        p_lift = p_strike.copy()
        p_lift[2] += RETRACT_HEIGHT * 1000
        indy.movel(list(p_lift), vel_ratio=30, acc_ratio=60)
        wait_indy(indy, sync_sim=sync_pb, pb=pb if sync_pb else None)
    return True


def run_sim_trajectory(pb, ik, trajectory, phases):
    """GUI 시뮬: approach + strike 구간."""
    traj = list(trajectory)
    q0 = pb.my_robot.q.copy()
    q_traj = ik.solve_trajectory(q0, traj)
    a0, a1 = phases['approach'][0], phases['approach'][1]
    for i in range(a0, a1):
        pb.MoveRobot(q_traj[i], degree=False)
        time.sleep(0.002)
    time.sleep(0.3)
    for i in range(phases['strike'][0], len(q_traj)):
        pb.MoveRobot(q_traj[i], degree=False)
        time.sleep(0.002)


def plan_linear_shot(pb, env, ik, cue_pos, strike_dir_3d, strike_speed=0.1,
                     approach_dist=None, follow_dist=None):
    """직선/약한 타격 (Phase 1 좌표 보정용)."""
    from project.trajectory_planner import StrikeTrajectoryPlanner

    approach_dist = STRIKE_APPROACH_DIST if approach_dist is None else approach_dist
    follow_dist = STRIKE_FOLLOW_DIST if follow_dist is None else follow_dist
    T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
    trajectory, phases = StrikeTrajectoryPlanner().plan_strike(
        T_current=T_now, ball_pos=np.array(cue_pos),
        strike_direction=np.asarray(strike_dir_3d).flatten(),
        strike_speed=strike_speed,
        approach_dist=approach_dist,
        follow_dist=follow_dist,
        table_bounds=env.table_bounds,
    )
    traj_arr = np.stack([np.asarray(T) for T in trajectory], axis=0)
    return traj_arr, phases, strike_speed


def plan_cushion_shot(pb, env, ik, cue_pos, target_pos, ball2_pos):
    """3쿠션 플래너 (Phase 2, 홀 회피 포함)."""
    from project.physics.cushion_planner import CushionShotPlanner
    from project.trajectory_planner import StrikeTrajectoryPlanner

    shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
    cands = shot_planner.plan_shot(
        np.array(cue_pos), np.array(target_pos), [], ball2_pos=np.array(ball2_pos))
    if not cands:
        raise RuntimeError('플래너 후보 없음')
    best = cands[0]
    horiz = np.array(best['strike_dir'][:2]).flatten()
    horiz /= np.linalg.norm(horiz)
    strike_dir = np.array([horiz[0], horiz[1], 0.0])
    speed = float(best['strike_speed'])

    T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
    trajectory, phases = StrikeTrajectoryPlanner().plan_strike(
        T_current=T_now, ball_pos=np.array(cue_pos),
        strike_direction=strike_dir, strike_speed=speed,
        approach_dist=best.get('safe_approach_dist', STRIKE_APPROACH_DIST),
        follow_dist=STRIKE_FOLLOW_DIST,
        table_bounds=env.table_bounds,
    )
    traj_arr = np.stack([np.asarray(T) for T in trajectory], axis=0)
    return traj_arr, phases, speed, best


DEFAULT_PLAN_FILE = 'last_calib_shot.npz'


def save_shot_plan(path, trajectory, phases, speed, movel_plan, fk_offset_mm=None, meta=None):
    """Approach 직후 저장 — 별도 실행(--real-step strike)에서 Strike."""
    speed = float(speed)
    meta = meta or {}
    np.savez(
        path,
        trajectory=np.asarray(trajectory),
        approach_start=int(phases['approach'][0]),
        approach_end=int(phases['approach'][1]),
        follow_end=int(phases['follow'][1]),
        speed=float(speed),
        fk_offset_mm=np.asarray(fk_offset_mm if fk_offset_mm is not None else np.zeros(3)),
        p_ready_cmd=np.asarray(movel_plan['p_ready_cmd']),
        p_strike_cmd=np.asarray(movel_plan['p_strike_cmd']),
        meta_json=json.dumps(meta or {}),
    )
    print(f"  [PLAN] 저장: {path}  (다음: --real-step strike)")


def load_shot_plan(path):
    d = np.load(path, allow_pickle=False)
    phases = {
        'approach': (int(d['approach_start']), int(d['approach_end'])),
        'strike': (int(d['approach_end']), int(d['approach_end'])),
        'follow': (int(d['approach_end']), int(d['follow_end'])),
    }
    return {
        'trajectory': d['trajectory'],
        'phases': phases,
        'speed': float(d['speed']),
        'fk_offset_mm': np.asarray(d['fk_offset_mm']),
        'p_ready_cmd': np.asarray(d['p_ready_cmd']),
        'p_strike_cmd': np.asarray(d['p_strike_cmd']),
        'meta_json': d['meta_json'].item() if hasattr(d['meta_json'], 'item') else str(d.get('meta_json', '{}')),
    }


def run_real_approach_only(indy, pb, trajectory, phases, speed, fk_offset_mm=None,
                           plan_path=None, meta=None):
    """Approach+Align(movel) 후 정지. 계획 파일 저장."""
    plan_path = plan_path or DEFAULT_PLAN_FILE
    movel_plan = run_approach_movel(indy, pb, trajectory, phases, fk_offset_mm)
    save_shot_plan(plan_path, trajectory, phases, speed, movel_plan, fk_offset_mm, meta)
    print("  [REAL] Approach 종료 — 로봇 Ready 유지. Strike는 별도 명령으로.")
    return movel_plan


def run_real_strike_only(indy, pb, plan_path=None, speed=None, do_retract=True,
                         wait_start=True):
    """저장된 Ready에서 [Enter] START 후 Strike(movel)만."""
    plan_path = plan_path or DEFAULT_PLAN_FILE
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"계획 없음: {plan_path}  (먼저 --real-step approach)")
    loaded = load_shot_plan(plan_path)
    sp = speed if speed is not None else loaded['speed']
    if wait_start:
        prompt_strike_start()
    ok = run_strike_movel(
        indy, pb, loaded['p_strike_cmd'], loaded['p_ready_cmd'],
        speed=sp, do_retract=do_retract)
    return ok, loaded


def prompt_strike_start():
    """Approach 완료 후 로봇 정지 — 사용자 START(Enter)까지 Strike 안 함."""
    print("\n" + "=" * 56)
    print("  APPROACH 완료 — 로봇 정지 (이 구간에 movej 없음)")
    print("  큐·큐대·자세 확인 후")
    print("  >>> [Enter] = START → MoveL 직선 STRIKE 만 실행")
    print("=" * 56)
    input()


def execute_movel_split(indy, pb, trajectory, phases, speed,
                        confirm_before_strike=True, fk_offset_mm=None,
                        do_retract=True, strike_vel_ratio=None):
    """Approach(movel만) → 정지 → [START] → Strike(movel만) → Retract(movel).

    Approach와 Strike 사이에는 movej를 쓰지 않습니다.
    """
    movel_plan = run_approach_movel(indy, pb, trajectory, phases, fk_offset_mm)
    if confirm_before_strike:
        prompt_strike_start()
    else:
        print("  [WARN] --allow-auto-strike: Enter 없이 곧바로 Strike")
    ok = run_strike_movel(
        indy, pb, movel_plan['p_strike_cmd'], movel_plan['p_ready_cmd'],
        strike_vel_ratio=strike_vel_ratio, speed=speed,
        fk_offset_mm=fk_offset_mm, do_retract=do_retract)
    return ok, movel_plan
