"""
비-벽 위치 대량 테스트 스크립트
=================================
벽에서 3cm 이상 떨어진 무작위 위치에 3개 공을 배치하고
각 배치에서 1회 타격하여 성공 여부를 기록합니다.
"""
import argparse
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from project.config import *
from project.robot_controller import RobotController
from project.trajectory_planner import StrikeTrajectoryPlanner
from project.ik_solver import IKSolver
from project.physics.cushion_planner import CushionShotPlanner
from project.perception import SimPerception
from project.environment.maze_env import MazeEnvironment
from project.state_machine import AutonomousStateMachine
import pybullet as p


def random_non_wall_pos(bounds, margin=0.03, z=0.071):
    """벽에서 margin 이상 떨어진 무작위 위치 생성"""
    x = np.random.uniform(bounds['x_min'] + margin, bounds['x_max'] - margin)
    y = np.random.uniform(bounds['y_min'] + margin, bounds['y_max'] - margin)
    return np.array([x, y, z])


def balls_too_close(positions, min_dist=0.05):
    """공들이 서로 너무 가까우면 True"""
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i][:2] - positions[j][:2])
            if d < min_dist:
                return True
    return False


def run_stress_test(num_tests=20, wall_margin=0.03, seed=None, view_time=1):
    """비-벽 위치 대량 테스트"""
    if seed is not None:
        np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  NON-WALL STRESS TEST")
    print(f"{'='*60}")
    print(f"  Tests: {num_tests}")
    print(f"  Wall margin: {wall_margin*100:.0f}cm")
    print(f"  Seed: {seed}")
    print(f"{'='*60}\n")

    # 1. 로봇 + 환경 초기화
    controller = RobotController(mode='sim')
    controller.connect()
    time.sleep(2)
    controller.move_home()
    time.sleep(2)

    robot_id = controller.pb.my_robot.robotId
    ee_link = controller.pb.my_robot.RobotEEJointIdx[-1]

    CX = MAZE_TABLE_CENTER_X
    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

    env = MazeEnvironment(controller.pb.ClientId)
    env.setup(
        cue_pos=[CX, CY - W / 4, ball_h],
        target_pos=[CX, CY + W / 8, ball_h],
        num_obstacles=0,
        seed=seed
    )
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    controller.set_environment(env)

    tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS
    shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
    perception = SimPerception(env)

    controller.boost_pd_gains(kp=5000, kd=200)
    time.sleep(1)

    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)

    bounds = env.table_bounds

    # 결과 기록
    results = []
    successes = 0
    escapes = 0

    for test_i in range(num_tests):
        print(f"\n{'='*50}")
        print(f"  TEST {test_i+1}/{num_tests}")
        print(f"{'='*50}")

        # 무작위 비-벽 위치 생성 (공 3개 모두)
        for _ in range(100):  # 최대 100회 시도
            cue_pos = random_non_wall_pos(bounds, margin=wall_margin, z=ball_h)
            tgt1_pos = random_non_wall_pos(bounds, margin=wall_margin, z=ball_h)
            tgt2_pos = random_non_wall_pos(bounds, margin=wall_margin, z=ball_h)
            if not balls_too_close([cue_pos, tgt1_pos, tgt2_pos]):
                break

        # 공 배치
        env.reset_balls(cue_pos=cue_pos, target_pos=tgt1_pos, ball2_pos=tgt2_pos)
        # start 위치 갱신 (observe에서 displacement 계산용)
        env.cue_start_pos = cue_pos.copy()
        env.target_start_pos = tgt1_pos.copy()
        env.ball2_start_pos = tgt2_pos.copy()
        time.sleep(0.3)

        print(f"  Cue:  ({cue_pos[0]:.3f}, {cue_pos[1]:.3f})")
        print(f"  Tgt1: ({tgt1_pos[0]:.3f}, {tgt1_pos[1]:.3f})")
        print(f"  Tgt2: ({tgt2_pos[0]:.3f}, {tgt2_pos[1]:.3f})")

        # 벽 거리 확인
        dx_min = cue_pos[0] - bounds['x_min']
        dx_max = bounds['x_max'] - cue_pos[0]
        dy_min = cue_pos[1] - bounds['y_min']
        dy_max = bounds['y_max'] - cue_pos[1]
        min_wall_dist = min(dx_min, dx_max, dy_min, dy_max)
        print(f"  Min wall dist: {min_wall_dist*100:.1f}cm")

        # State Machine 1회 실행
        sm = AutonomousStateMachine(
            controller=controller,
            environment=env,
            shot_planner=shot_planner,
            traj_planner=traj_planner,
            demo_type='maze',
            tool_offset=tool_offset,
            perception=perception
        )

        # 1회 시도
        sm.state = 'SCAN'
        sm.attempt = 0
        sm.attempt += 1

        # SCAN
        scan_data = sm._scan()

        # THINK
        plan = sm._think(scan_data)
        candidates = plan.get('candidates', [])
        is_escape = len(candidates) > 0 and candidates[0].get('is_escape', False)
        top_score = candidates[0]['score'] if candidates else 0

        if is_escape:
            print(f"  [ESCAPE] → 비-벽 위치인데 escape 발동!")
            escapes += 1
            result = {'test': test_i+1, 'success': False, 'escape': True,
                      'cue': cue_pos.copy(), 'tgt1': tgt1_pos.copy(), 'tgt2': tgt2_pos.copy(),
                      'angle': plan.get('strike_dir', None), 'score': top_score,
                      'min_wall_dist': min_wall_dist}
            results.append(result)
            controller.move_home()
            time.sleep(0.5)
            continue

        print(f"  Top score: {top_score}, angle: {candidates[0]['angle_deg']:.1f}deg" if candidates else "  No candidates!")

        # STRIKE
        sm._strike_skipped = False
        sm._strike_skip_reason = None
        sm._strike(scan_data, plan)

        if getattr(sm, '_strike_skipped', False):
            print(f"  [SKIP] Strike skipped")
            result = {'test': test_i+1, 'success': False, 'escape': False,
                      'cue': cue_pos.copy(), 'tgt1': tgt1_pos.copy(), 'tgt2': tgt2_pos.copy(),
                      'angle': None, 'score': top_score,
                      'min_wall_dist': min_wall_dist, 'reason': 'skip'}
            results.append(result)
            controller.move_home()
            time.sleep(0.5)
            continue

        # OBSERVE
        success = sm._observe()

        if success:
            successes += 1
            print(f"\n  [OK] SUCCESS ({successes}/{test_i+1})")
        else:
            print(f"\n  [XX] MISS ({successes}/{test_i+1})")

        result = {'test': test_i+1, 'success': success, 'escape': False,
                  'cue': cue_pos.copy(), 'tgt1': tgt1_pos.copy(), 'tgt2': tgt2_pos.copy(),
                  'angle': candidates[0]['angle_deg'] if candidates else None,
                  'score': top_score,
                  'min_wall_dist': min_wall_dist}
        results.append(result)

        controller.move_home()
        time.sleep(0.5)

    # 최종 결과
    print(f"\n{'='*60}")
    print(f"  STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total: {num_tests}")
    print(f"  Successes: {successes}")
    print(f"  Escapes (unexpected): {escapes}")
    print(f"  Misses: {num_tests - successes - escapes}")
    print(f"  Success Rate: {successes}/{num_tests - escapes} = "
          f"{successes/(num_tests-escapes)*100:.1f}% (excluding escapes)" if num_tests > escapes else "N/A")
    print(f"  Overall: {successes}/{num_tests} = {successes/num_tests*100:.1f}%")

    # 실패 상세
    failures = [r for r in results if not r['success'] and not r['escape']]
    if failures:
        print(f"\n  FAILURES:")
        for f in failures:
            print(f"    Test {f['test']}: cue=({f['cue'][0]:.3f},{f['cue'][1]:.3f}), "
                  f"angle={f.get('angle', 'N/A')}, score={f['score']}, "
                  f"wall_dist={f['min_wall_dist']*100:.1f}cm")

    # Escape 상세
    escape_list = [r for r in results if r['escape']]
    if escape_list:
        print(f"\n  UNEXPECTED ESCAPES:")
        for e in escape_list:
            print(f"    Test {e['test']}: cue=({e['cue'][0]:.3f},{e['cue'][1]:.3f}), "
                  f"wall_dist={e['min_wall_dist']*100:.1f}cm")

    print(f"\n  Viewing for {view_time} seconds...")
    time.sleep(view_time)

    if hasattr(env, 'cleanup'):
        env.cleanup()
    controller.disconnect()
    print(f"\n  Stress test finished.\n")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', type=int, default=20)
    parser.add_argument('--margin', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--view-time', type=int, default=1)
    args = parser.parse_args()
    run_stress_test(num_tests=args.tests, wall_margin=args.margin,
                    seed=args.seed, view_time=args.view_time)
