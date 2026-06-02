"""
포켓볼 데모 실행
===============
Phase 1: 목적구 3개(노/빨/검) 포켓에 넣기 (최적 공 자동 선택)
Phase 2: 목적구 3개를 ArUco 마커 위치에 정밀 정지 (1cm)

두 Phase는 독립 데모로 별도 실행.
물리 파라미터는 기존 실측 보정된 MAZE_BALL 값 재활용.

Usage:
    python run_pocket_demo.py --phase 1
    python run_pocket_demo.py --phase 2
    python run_pocket_demo.py --phase 1 --attempts 6
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
from project.physics.pocket_planner import PocketShotPlanner
from project.environment.maze_env import MazeEnvironment
from project.state_machine import AutonomousStateMachine


def run_pocket_demo(max_attempts=4, view_time=15, friction=None, phase=None):
    """포켓볼 데모 전체 실행"""

    if friction is not None:
        import project.config as cfg
        cfg.POCKET_DEMO_FRICTION = friction
        cfg.POCKET_DEMO_ROLLING_FRICTION = friction * 0.08
        print(f"  [CONFIG] Friction overridden: {friction}")

    eff_friction = POCKET_DEMO_FRICTION if friction is None else friction

    print(f"\n{'='*60}")
    print(f"  Pocket Ball Demo Simulation")
    print(f"{'='*60}")
    print(f"  Attempts per ball: {max_attempts}")
    print(f"  Friction: {eff_friction}")
    print(f"  Precision tolerance: {PRECISION_STOP_TOLERANCE*100:.0f}cm")
    print(f"  Phase: {phase if phase else 'all'}")
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
    CY, W, L = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH, MAZE_TABLE_LENGTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

    env = MazeEnvironment(controller.pb.ClientId)

    # Phase에 따라 초기 배치 결정
    if phase == 2:
        # Phase 2만: 일렬 배치
        env.setup(
            cue_pos=[CX, CY - W / 4, ball_h],
            target_pos=[CX, CY - LINEUP_SPACING, ball_h],
            ball2_pos=[CX, CY, ball_h],
            ball3_pos=[CX, CY + LINEUP_SPACING, ball_h],
            num_obstacles=0,
            setup_pockets=True,
        )
    else:
        # Phase 1 or 전체: 랜덤 배치
        env.setup(
            cue_pos=[CX, CY - W / 4, ball_h],
            target_pos=[CX + L / 8, CY + W / 8, ball_h],
            ball2_pos=[CX - L / 8, CY, ball_h],
            ball3_pos=[CX, CY + W / 6, ball_h],
            num_obstacles=0,
            setup_pockets=True,
        )

    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    controller.set_environment(env)

    tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS
    shot_planner = PocketShotPlanner(table_bounds=env.table_bounds)

    controller.boost_pd_gains(kp=5000, kd=200)
    time.sleep(3)

    # GUI 시각화: 포켓 위치 표시
    import pybullet as _p
    client = controller.pb.ClientId
    surface_z = H + TH / 2 + 0.002

    # 포켓 위치에 빨간 원 표시
    for pp in env.pocket_positions:
        n_seg = 16
        for i in range(n_seg):
            th0 = 2 * np.pi * i / n_seg
            th1 = 2 * np.pi * (i + 1) / n_seg
            r = POCKET_RADIUS
            _p.addUserDebugLine(
                [pp[0] + r * np.cos(th0), pp[1] + r * np.sin(th0), surface_z],
                [pp[0] + r * np.cos(th1), pp[1] + r * np.sin(th1), surface_z],
                [1, 0, 0], lineWidth=3, lifeTime=0, physicsClientId=client)

    # 로봇 작업 반경
    n_seg = 64
    for radius, color in [(0.70, [0, 1, 0]), (0.80, [1, 0.5, 0])]:
        for i in range(n_seg):
            th0 = 2 * np.pi * i / n_seg
            th1 = 2 * np.pi * (i + 1) / n_seg
            _p.addUserDebugLine(
                [radius * np.cos(th0), radius * np.sin(th0), surface_z],
                [radius * np.cos(th1), radius * np.sin(th1), surface_z],
                color, lineWidth=2, lifeTime=0, physicsClientId=client)

    print(f"  [VIS] {len(env.pocket_positions)} pockets displayed")

    # 2. 궤적 + 상태머신
    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)

    demo_type = 'pocket_phase2' if phase == 2 else 'pocket_phase1'

    sm = AutonomousStateMachine(
        controller=controller,
        environment=env,
        shot_planner=shot_planner,
        traj_planner=traj_planner,
        demo_type=demo_type,
        tool_offset=tool_offset,
    )

    # 3. 실행
    success = sm.run(max_attempts=max_attempts)

    print(f"\n  Result: {'SUCCESS' if success else 'COMPLETED'}")
    print(f"  Viewing for {view_time} seconds...")
    time.sleep(view_time)

    if hasattr(env, 'cleanup'):
        env.cleanup()
    controller.disconnect()
    print(f"\n  Pocket demo finished.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pocket Ball Demo')
    parser.add_argument('--attempts', type=int, default=4,
                        help='Max attempts per ball')
    parser.add_argument('--friction', type=float, default=None,
                        help='Override friction coefficient (0.6~1.0)')
    parser.add_argument('--phase', type=int, default=None, choices=[1, 2],
                        help='Run specific phase only')
    parser.add_argument('--view', type=int, default=15,
                        help='View time after completion (seconds)')
    args = parser.parse_args()

    run_pocket_demo(
        max_attempts=args.attempts,
        view_time=args.view,
        friction=args.friction,
        phase=args.phase,
    )
