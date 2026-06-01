# -*- coding: utf-8 -*-
"""
실기: Approach(movel) 와 Strike(movel 직선) 분리 실행
====================================================
movej 곡선 타격 대신, Ready에서 사람이 확인한 뒤 MoveL로 직선 타격.

사용법:
  # 1) 접근만 (Ready까지 movel) → last_shot_plan.npz 저장
  python real_approach_then_strike.py --phase approach

  # 2) 로봇이 Ready에 있는 상태에서, Enter 누르면 직선 타격
  python real_approach_then_strike.py --phase strike

  # 3) 접근 → Enter 대기 → 타격 (한 세션)
  python real_approach_then_strike.py --phase both

  # 4) 시뮬만 (플래너 + GUI 궤적 확인, 로봇 없음)
  python real_approach_then_strike.py --phase sim

옵션:
  --robot-ip IP
  --vision          카메라로 공 위치 (없으면 테이블 기본 배치)
  --strike-vel 0.5  타격 movel 속도 비율 (MAX_TOOL_SPEED 대비)
  --no-retract      타격 후 상승 생략
  --no-home         마지막 홈 복귀 생략
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project.config import *

PLAN_FILE = 'last_shot_plan.npz'
MOVEL_MIN_DIST_MM = 3.0
APPROACH_MOVEL_STEP = 100
APPROACH_VEL = 20
APPROACH_ACC = 50
ALIGN_VEL = 10
ALIGN_ACC = 30


def _SE3_to_p6(T, offset_mm=None):
    from src.utils import Rot2eul
    off = np.zeros(3) if offset_mm is None else np.asarray(offset_mm).reshape(3)
    p6 = np.zeros(6)
    p6[0:3] = 1000.0 * np.asarray(T)[0:3, 3] - off
    p6[3:6] = Rot2eul(np.asarray(T)[0:3, 0:3], seq='XYZ', degree=True)
    return p6


def _wait_indy(indy, timeout=60, sync_sim=False, pb=None):
    time.sleep(0.2)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        if indy.get_motion_data()['is_in_motion']:
            break
        time.sleep(0.05)
    while time.time() - t0 < timeout:
        if sync_sim and pb is not None:
            _sync_sim_from_real(indy, pb)
        if not indy.get_motion_data()['is_in_motion']:
            break
        time.sleep(0.05)
    if sync_sim and pb is not None:
        _sync_sim_from_real(indy, pb)


def _sync_sim_from_real(indy, pb):
    q_deg = list(np.asarray(indy.get_control_data()['q']).flatten())
    q_rad = np.deg2rad(q_deg).reshape(-1, 1)
    robot = pb.my_robot
    client = pb.ClientId
    import pybullet as p
    for i, jidx in enumerate(robot.RobotMovableJointIdx):
        p.resetJointState(robot.robotId, jidx, float(q_rad[i, 0]), 0.0,
                          physicsClientId=client)
    robot._q = q_rad.copy()
    robot._qdot = np.zeros_like(q_rad)
    robot.set_desired_joint_pos(q_rad)
    robot._get_robot_states()
    pb.MoveRobot(q_deg, degree=True)


def _movej_home(indy, pb, wait=True):
    indy.movej(list(HOME_Q_DEG), vel_ratio=30, acc_ratio=50)
    if wait:
        _wait_indy(indy, sync_sim=True, pb=pb)
    _sync_sim_from_real(indy, pb)


def _verify_movel(indy, p_target, tol_mm=5.0):
    p_now = indy.get_control_data()['p']
    err = float(np.linalg.norm(np.array(p_now[:3]) - np.array(p_target[:3])))
    if err > tol_mm:
        print(f"  [WARN] movel 미도달 err={err:.1f}mm, 재시도...")
        indy.movel(list(p_target), vel_ratio=50, acc_ratio=100)
        _wait_indy(indy)
    return err


def compute_fk_offset_mm(indy, pb):
    """홈에서 Pinocchio FK vs 실기 TCP 차이 (mm)."""
    _movej_home(indy, pb)
    q_rad = np.deg2rad(HOME_Q_DEG)
    T_pin = pb.my_robot.pinModel.FK(q_rad)
    p_real = indy.get_control_data()['p']
    return 1000.0 * T_pin[0:3, 3] - np.array(p_real[:3])


def save_plan(path, plan, fk_offset_mm=None):
    phases = plan['phases']
    np.savez(
        path,
        trajectory=plan['trajectory'],
        approach_start=phases['approach'][0],
        approach_end=phases['approach'][1],
        follow_end=phases['follow'][1],
        speed=plan['speed'],
        angle_deg=plan['angle_deg'],
        fk_offset_mm=np.asarray(fk_offset_mm if fk_offset_mm is not None else np.zeros(3)),
        p_ready_cmd=plan.get('p_ready_cmd', np.zeros(6)),
        p_strike_cmd=plan.get('p_strike_cmd', np.zeros(6)),
    )
    print(f"  계획 저장: {path}")


def load_plan(path):
    d = np.load(path, allow_pickle=False)
    trajectory = d['trajectory']
    phases = {
        'approach': (int(d['approach_start']), int(d['approach_end'])),
        'strike': (int(d['approach_end']), int(d['approach_end'])),  # unused
        'follow': (int(d['approach_end']), int(d['follow_end'])),
    }
    return {
        'trajectory': trajectory,
        'phases': phases,
        'speed': float(d['speed']),
        'angle_deg': float(d['angle_deg']),
        'fk_offset_mm': d['fk_offset_mm'],
        'p_ready_cmd': d['p_ready_cmd'],
        'p_strike_cmd': d['p_strike_cmd'],
    }


def run_approach_movel(indy, pb, plan, fk_offset_mm=None):
    """Phase A: Approach + Align (movel만), Ready에서 정지."""
    traj = plan['trajectory']
    phases = plan['phases']
    off = fk_offset_mm
    a0, a1 = phases['approach']
    follow_end = phases['follow'][1]

    _sync_sim_from_real(indy, pb)

    wps = list(range(a0, a1, APPROACH_MOVEL_STEP))
    if not wps or wps[-1] != a1 - 1:
        wps.append(a1 - 1)
    print(f"\n[Approach] movel {len(wps)} waypoints (vel={APPROACH_VEL}%)...")
    for i, idx in enumerate(wps):
        p = _SE3_to_p6(traj[idx], off)
        indy.movel(list(p), vel_ratio=APPROACH_VEL, acc_ratio=APPROACH_ACC)
        _wait_indy(indy, sync_sim=True, pb=pb)
        if i % max(1, len(wps) // 5) == 0:
            print(f"  waypoint {i+1}/{len(wps)}")

    T_ready = traj[a1 - 1]
    p_ready = _SE3_to_p6(T_ready, off)
    print("[Align] ready movel...")
    time.sleep(0.3)
    indy.movel(list(p_ready), vel_ratio=ALIGN_VEL, acc_ratio=ALIGN_ACC)
    _wait_indy(indy, sync_sim=True, pb=pb)
    _verify_movel(indy, p_ready, tol_mm=5.0)

    T_follow = traj[min(follow_end - 1, len(traj) - 1)]
    p_strike, planner_mm = _strike_target_from_ready(p_ready, T_ready, T_follow)

    p_now = indy.get_control_data()['p']
    plan['p_ready_cmd'] = np.array(p_ready)
    plan['p_strike_cmd'] = np.array(p_strike)

    print("\n[Approach 완료] 로봇 Ready — 타격 전 확인하세요.")
    print(f"  TCP now(mm):    [{p_now[0]:.1f}, {p_now[1]:.1f}, {p_now[2]:.1f}]")
    print(f"  ready CMD(mm):  [{p_ready[0]:.1f}, {p_ready[1]:.1f}, {p_ready[2]:.1f}]")
    print(f"  strike CMD(mm): [{p_strike[0]:.1f}, {p_strike[1]:.1f}, {p_strike[2]:.1f}]")
    print(f"  planner Δ(mm):  {planner_mm:.1f}")
    return plan


def _strike_target_from_ready(p_ready, T_ready, T_follow):
    p_strike = np.array(p_ready, dtype=float).copy()
    delta_m = np.asarray(T_follow)[0:3, 3] - np.asarray(T_ready)[0:3, 3]
    p_strike[0:3] = p_ready[0:3] + 1000.0 * delta_m
    p_strike[3:6] = p_ready[3:6]
    return p_strike, float(np.linalg.norm(1000.0 * delta_m))


def run_strike_movel(indy, pb, plan, strike_vel_ratio=None, do_retract=True):
    """Phase B: Enter 후 단일 movel 직선 타격."""
    off = plan.get('fk_offset_mm', np.zeros(3))
    p_strike = np.array(plan['p_strike_cmd'], dtype=float)
    speed = plan['speed']
    if strike_vel_ratio is None:
        strike_vel_ratio = int(np.clip(speed / MAX_TOOL_SPEED * 100, 30, 100))

    p_now = indy.get_control_data()['p']
    strike_vec = p_strike[:3] - np.array(p_now[:3])
    strike_dist = float(np.linalg.norm(strike_vec))
    print("\n--- STRIKE (MoveL) ---")
    print(f"  TCP now(mm):    [{p_now[0]:.1f}, {p_now[1]:.1f}, {p_now[2]:.1f}]")
    print(f"  strike CMD(mm): [{p_strike[0]:.1f}, {p_strike[1]:.1f}, {p_strike[2]:.1f}]")
    print(f"  distance:       {strike_dist:.1f} mm  vel={strike_vel_ratio}%")

    if strike_dist < MOVEL_MIN_DIST_MM:
        delta = p_strike[:3] - np.array(plan['p_ready_cmd'][:3])
        dn = np.linalg.norm(delta)
        if dn >= MOVEL_MIN_DIST_MM:
            p_forced = np.array(p_now, dtype=float)
            p_forced[0:3] = p_now[:3] + delta
            p_forced[3:6] = p_now[3:6]
            p_strike = p_forced
            strike_dist = dn
            print(f"  [보정] 현재 TCP + planner Δ {dn:.1f}mm")
        else:
            print(f"  [SKIP] 거리 {strike_dist:.1f}mm < {MOVEL_MIN_DIST_MM}mm")
            return

    indy.movel(list(p_strike), vel_ratio=strike_vel_ratio, acc_ratio=100)
    _wait_indy(indy, sync_sim=True, pb=pb)
    p_after = indy.get_control_data()['p']
    moved = float(np.linalg.norm(np.array(p_after[:3]) - np.array(p_now[:3])))
    print(f"  Strike 이동량: {moved:.1f} mm")
    _verify_movel(indy, p_strike, tol_mm=8.0)

    if do_retract:
        p_lift = p_strike.copy()
        p_lift[2] += RETRACT_HEIGHT * 1000
        print("[Retract] 위로 movel...")
        indy.movel(list(p_lift), vel_ratio=30, acc_ratio=60)
        _wait_indy(indy, sync_sim=True, pb=pb)
    print("[Strike 완료]")


def run_sim_preview(pb, env, ik, plan):
    """GUI에서 approach + strike 궤적만 재생."""
    traj = plan['trajectory']
    phases = plan['phases']
    q_traj = ik.solve_trajectory(pb.my_robot.q.copy(), [traj[i] for i in range(len(traj))])
    a0, a1 = phases['approach'][0], phases['approach'][1]
    print("[SIM] Approach...")
    for i in range(a0, a1):
        pb.MoveRobot(q_traj[i], degree=False)
        time.sleep(0.002)
    time.sleep(0.3)
    print("[SIM] Strike segment...")
    for i in range(phases['strike'][0], len(q_traj)):
        pb.MoveRobot(q_traj[i], degree=False)
        time.sleep(0.002)
    print("[SIM] 완료 — GUI에서 확인")


def _prompt(msg):
    print(f"\n>>> {msg}")
    input("    [Enter] 계속  |  Ctrl+C 취소\n")


def main():
    parser = argparse.ArgumentParser(description='Approach / Strike 분리 (movel)')
    parser.add_argument('--phase', choices=['approach', 'strike', 'both', 'sim'],
                        default='both')
    parser.add_argument('--robot-ip', default='192.168.0.13')
    parser.add_argument('--plan-file', default=PLAN_FILE)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--strike-vel', type=int, default=None,
                        help='movel vel_ratio (0-100), 기본=속도에서 자동')
    parser.add_argument('--no-retract', action='store_true')
    parser.add_argument('--no-home', action='store_true')
    parser.add_argument('--skip-fk-offset', action='store_true',
                        help='FK 오프셋 보정 생략 (0)')
    args = parser.parse_args()

    from src.core.pybullet_core import PybulletCore
    from project.environment.maze_env import MazeEnvironment
    from project.ik_solver import IKSolver
    pb = PybulletCore()
    pb.connect(robot_name='indy7_v2', joint_limit=True, constraint_visualization=False)
    ik = IKSolver(pb.my_robot.pinModel, gain=IK_GAIN, damping=IK_DAMPING)
    env = MazeEnvironment(pb.ClientId)
    robot_id = pb.my_robot.robotId
    ee_link = pb.my_robot.RobotEEJointIdx[-1]

    # --- plan (sim / approach / both need fresh plan unless strike-only) ---
    plan = None
    if args.phase == 'strike':
        if not os.path.exists(args.plan_file):
            print(f"[ERROR] {args.plan_file} 없음. 먼저 --phase approach 실행")
            sys.exit(1)
        plan = load_plan(args.plan_file)
        env.attach_compact_tool(robot_id, ee_link)
        env.disable_tool_env_collision()
        print(f"  계획 로드: {args.plan_file}  angle={plan['angle_deg']:.1f}°")
        print("  [주의] 로봇이 Approach 직후 Ready 자세에 있어야 합니다.")
    else:
        print("[PLAN] 타격 계획 중...")
        plan = _plan_maze_shot_impl(pb, env, ik, args.vision)
        env.attach_compact_tool(robot_id, ee_link)
        env.disable_tool_env_collision()

    if args.phase == 'sim':
        env.attach_compact_tool(robot_id, ee_link)
        env.disable_tool_env_collision()
        run_sim_preview(pb, env, ik, plan)
        print("\n시뮬만 종료. 실기는 --phase approach / both")
        pb.disconnect()
        return

    from neuromeka import IndyDCP3
    indy = IndyDCP3(robot_ip=args.robot_ip, index=0)
    print(f"  Indy 연결: {args.robot_ip}")

    fk_off = np.zeros(3) if args.skip_fk_offset else compute_fk_offset_mm(indy, pb)
    print(f"  FK_OFFSET_MM = [{fk_off[0]:.1f}, {fk_off[1]:.1f}, {fk_off[2]:.1f}]")
    plan['fk_offset_mm'] = fk_off

    if args.phase in ('approach', 'both'):
        _prompt("안전 확인 후 Approach 시작")
        plan = run_approach_movel(indy, pb, plan, fk_offset_mm=fk_off)
        save_plan(args.plan_file, plan, fk_off)
        if args.phase == 'approach':
            print("\n접근만 완료. 타격 시:")
            print(f"  python real_approach_then_strike.py --phase strike --plan-file {args.plan_file}")
            if not args.no_home:
                _prompt("홈 복귀")
                _movej_home(indy, pb)
            pb.disconnect()
            return

    if args.phase in ('strike', 'both'):
        _prompt("Ready 확인 후 STRIKE (MoveL 직선)")
        run_strike_movel(indy, pb, plan, strike_vel_ratio=args.strike_vel,
                         do_retract=not args.no_retract)
        if not args.no_home:
            _prompt("홈 복귀")
            _movej_home(indy, pb)

    pb.disconnect()
    print("\n종료.")


def _plan_maze_shot_impl(pb, env, ik, use_vision):
    from project.physics.cushion_planner import CushionShotPlanner
    from project.trajectory_planner import StrikeTrajectoryPlanner

    if use_vision:
        from project.real_env_to_pybullet import detect_balls
        cue, t1, t2 = detect_balls()
        H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
        z = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
        cue, t1, t2 = [cue[0], cue[1], z], [t1[0], t1[1], z], [t2[0], t2[1], z]
        if env.cue_ball_id is None:
            env.setup(cue_pos=cue, target_pos=t1, ball2_pos=t2, num_obstacles=0)
            env.disable_robot_env_collision(pb.my_robot.robotId)
            env.disable_tool_env_collision()
        else:
            env.reset_balls(cue_pos=cue, target_pos=t1, ball2_pos=t2)
    else:
        CX, CY, W = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
        H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
        z = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
        cue = [CX, CY - W / 4, z]
        t1 = [CX, CY + W / 8, z]
        t2 = [CX + MAZE_TABLE_LENGTH / 6, CY, z]
        if env.cue_ball_id is None:
            env.setup(cue_pos=cue, target_pos=t1, ball2_pos=t2, num_obstacles=0)
            env.disable_robot_env_collision(pb.my_robot.robotId)
            env.disable_tool_env_collision()
        else:
            env.reset_balls(cue_pos=cue, target_pos=t1, ball2_pos=t2)

    shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
    cands = shot_planner.plan_shot(np.array(cue), np.array(t1), [], ball2_pos=np.array(t2))
    if not cands:
        raise RuntimeError('플래너 후보 없음')
    best = cands[0]
    horiz = np.array(best['strike_dir'][:2]).flatten()
    horiz /= np.linalg.norm(horiz)
    strike_dir = np.array([horiz[0], horiz[1], 0.0])
    speed = float(best['strike_speed'])
    print(f"  Top: angle={best['angle_deg']:.1f}° speed={speed:.3f} cushions={best.get('cushion_count')}")

    T_now = pb.my_robot.pinModel.FK(pb.my_robot.q)
    q_now = pb.my_robot.q.copy()
    traj_planner = StrikeTrajectoryPlanner()
    trajectory, phases = traj_planner.plan_strike(
        T_current=T_now, ball_pos=np.array(cue),
        strike_direction=strike_dir, strike_speed=speed,
        approach_dist=best.get('safe_approach_dist', STRIKE_APPROACH_DIST),
        follow_dist=STRIKE_FOLLOW_DIST,
        table_bounds=env.table_bounds,
    )
    traj_arr = np.stack([np.asarray(T) for T in trajectory], axis=0)
    return {
        'trajectory': traj_arr,
        'phases': phases,
        'speed': speed,
        'angle_deg': best['angle_deg'],
        'cue': cue, 't1': t1, 't2': t2,
    }


if __name__ == '__main__':
    main()
