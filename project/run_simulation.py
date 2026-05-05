"""
통합 시뮬레이션 실행 스크립트
================================
미니골프 퍼팅 / 포켓볼 타격 데모를 PyBullet에서 실행

사용법:
    python run_simulation.py --demo minigolf
    python run_simulation.py --demo billiards
"""
import argparse
import time
import numpy as np
import sys
import os

# 프로젝트 루트를 path에 추가
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_root)

from project.config import *
from project.robot_controller import RobotController
from project.trajectory_planner import StrikeTrajectoryPlanner
from project.state_machine import AutonomousStateMachine
from project.physics.shot_planner import MinigolfShotPlanner, BilliardsShotPlanner


def run_minigolf():
    """미니골프 퍼팅 데모"""
    print("\n" + "=" * 60)
    print("  DEMO 1: Mini-Golf Putting Robot")
    print("  Visuo-Dynamic Fusion-Based Precision Impact Control")
    print("=" * 60)

    # 1. 로봇 초기화
    controller = RobotController(mode='sim')
    controller.connect()
    time.sleep(2)
    controller.move_home()
    time.sleep(2)

    # 2. 미니골프 환경 설정
    from project.environment.minigolf_env import MiniGolfEnvironment
    env = MiniGolfEnvironment(controller.pb.ClientId)
    env.setup(
        ball_pos=[0.45, -0.10, 0.035],
        hole_pos=[0.55, 0.12, 0.005],
        terrain_seed=42
    )
    time.sleep(1)

    # 3. 플래너 초기화
    shot_planner = MinigolfShotPlanner()
    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)

    # 4. State Machine 실행
    sm = AutonomousStateMachine(
        controller=controller,
        environment=env,
        shot_planner=shot_planner,
        traj_planner=traj_planner,
        demo_type='minigolf'
    )

    success = sm.run(max_attempts=3)

    # 5. 결과 출력
    print(f"\n{'='*40}")
    print(f"  Mini-Golf Demo Result: {'SUCCESS' if success else 'COMPLETED'}")
    print(f"  Final distance to hole: {env.get_distance_to_hole():.4f} m")
    print(f"{'='*40}")

    # 시뮬레이션 유지 (사용자가 볼 수 있도록)
    print("\nSimulation running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        env.cleanup()
        controller.disconnect()


def run_billiards():
    """포켓볼 타격 데모"""
    print("\n" + "=" * 60)
    print("  DEMO 2: Pocket Billiards Robot")
    print("  Vision-Guided Precision Impact Control")
    print("=" * 60)

    # 1. 로봇 초기화
    controller = RobotController(mode='sim')
    controller.connect()
    time.sleep(2)
    controller.move_home()
    time.sleep(2)

    # 2. 포켓볼 환경 설정
    from project.environment.billiards_env import BilliardsEnvironment
    env = BilliardsEnvironment(controller.pb.ClientId)

    L = BILLIARD_TABLE_LENGTH
    W = BILLIARD_TABLE_WIDTH
    H = BILLIARD_TABLE_SURFACE_HEIGHT
    TH = BILLIARD_TABLE_HEIGHT
    ball_h = H + TH / 2 + BILLIARD_BALL_RADIUS + 0.001

    env.setup(
        cue_pos=[0.5 - L/4, 0, ball_h],
        target_pos=[0.5 + L/8, 0.05, ball_h]
    )
    time.sleep(1)

    # 3. 플래너 초기화
    shot_planner = BilliardsShotPlanner()
    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)

    # 4. State Machine 실행
    sm = AutonomousStateMachine(
        controller=controller,
        environment=env,
        shot_planner=shot_planner,
        traj_planner=traj_planner,
        demo_type='billiards'
    )

    success = sm.run(max_attempts=3)

    # 5. 결과 출력
    target_pos = env.get_target_ball_position()
    nearest_pocket, dist = env.get_nearest_pocket(target_pos)
    print(f"\n{'='*40}")
    print(f"  Billiards Demo Result: {'POCKETED!' if success else 'COMPLETED'}")
    print(f"  Target ball distance to nearest pocket: {dist:.4f} m")
    print(f"{'='*40}")

    print("\nSimulation running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description='Indy7 Robot Strike Demo Simulation'
    )
    parser.add_argument(
        '--demo', type=str, default='minigolf',
        choices=['minigolf', 'billiards', 'both'],
        help='Demo type: minigolf, billiards, or both'
    )
    args = parser.parse_args()

    if args.demo == 'minigolf':
        run_minigolf()
    elif args.demo == 'billiards':
        run_billiards()
    elif args.demo == 'both':
        print("Running Mini-Golf demo first...")
        run_minigolf()
        print("\nRunning Billiards demo...")
        run_billiards()


if __name__ == '__main__':
    main()
