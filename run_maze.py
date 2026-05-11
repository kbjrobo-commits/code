"""
쓰리쿠션 당구 시뮬레이션 — 단일 실행 파일
=============================================
이 파일 하나만 실행하면 아래 전체가 자동으로 진행됩니다:

  1. GUI 시뮬레이터 실행 (PyBullet)
  2. 3공 headless 탐색 (최적 타격 각도/속도)
  3. 다중 후보 IK+장애물 검증
  4. 로봇 타격 실행 (3회 시도)
  5. 공 궤적 시각화 (PyBullet debug lines)
  6. 궤적 플롯 저장 (관절각, 태스크공간, 3D)

사용법:
    python run_maze.py
    python run_maze.py --obstacles 8
    python run_maze.py --attempts 5
    python run_maze.py --no-plot
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


def run_maze(num_obstacles=5, max_attempts=3, view_time=15,
             seed=None, save_plots=True):
    """쓰리쿠션 전체 파이프라인 실행"""

    print(f"\n{'='*60}")
    print(f"  3-Cushion Billiards Simulation")
    print(f"{'='*60}")
    print(f"  장애물: {num_obstacles}개")
    print(f"  시도 횟수: {max_attempts}회")
    print(f"  도구: 길이={TOOL_HEAD_LENGTH*100:.0f}cm, 질량={TOOL_HEAD_MASS:.1f}kg")
    print(f"  마찰: lateral={MAZE_BALL_FRICTION}, rolling={MAZE_BALL_ROLLING_FRICTION}")
    print(f"{'='*60}\n")

    # ── 1. 로봇 + 환경 초기화 ──
    controller = RobotController(mode='sim')
    controller.connect()
    time.sleep(2)
    controller.move_home()
    time.sleep(2)

    robot_id = controller.pb.my_robot.robotId
    ee_link = controller.pb.my_robot.RobotEEJointIdx[-1]

    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

    env = MazeEnvironment(controller.pb.ClientId)
    env.setup(
        cue_pos=[0.5, CY - W / 4, ball_h],
        target_pos=[0.5, CY + W / 8, ball_h],
        num_obstacles=num_obstacles,
        seed=seed
    )
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()

    tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS
    shot_planner = CushionShotPlanner(table_bounds=env.table_bounds)
    perception = SimPerception(env)

    controller.boost_pd_gains(kp=800, kd=40)
    time.sleep(3)

    # ── 2. 궤적 생성기 + 상태머신 ──
    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)

    sm = AutonomousStateMachine(
        controller=controller,
        environment=env,
        shot_planner=shot_planner,
        traj_planner=traj_planner,
        demo_type='maze',
        tool_offset=tool_offset,
        perception=perception
    )

    # ── 3. 실행 ──
    success = sm.run(max_attempts=max_attempts)

    print(f"\n  Result: {'SUCCESS' if success else 'COMPLETED'}")
    print(f"  Viewing for {view_time} seconds...")
    time.sleep(view_time)

    # ── 4. 궤적 플롯 생성 ──
    if save_plots:
        print(f"\n{'='*60}")
        print(f"  궤적 플롯 생성 중...")
        print(f"{'='*60}")
        try:
            _generate_maze_plots(controller, env, shot_planner, traj_planner,
                                 tool_offset, perception)
        except Exception as e:
            print(f"  [WARNING] 플롯 생성 실패: {e}")

    if hasattr(env, 'cleanup'):
        env.cleanup()
    controller.disconnect()
    print(f"\n  3-Cushion simulation finished.\n")


def _generate_maze_plots(controller, env, planner, traj_planner,
                         tool_offset, perception):
    """궤적 데이터로 플롯 생성"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 현재 공 위치로 샘플 궤적 계산
    scan = perception.scan_environment()
    cue_pos = scan['cue_pos']
    target_pos = scan['target_pos']
    ball2_pos = scan.get('ball2_pos')
    obstacles = scan.get('obstacles', [])

    candidates = planner.plan_shot(cue_pos, target_pos, obstacles,
                                   ball2_pos=ball2_pos)
    if not candidates:
        print("  플롯 생성 불가: 후보 없음")
        return

    best = candidates[0]
    strike_dir_2d = best['strike_dir']
    strike_speed = best['strike_speed']

    # 3D 방향 변환
    angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
    horiz = strike_dir_2d / np.linalg.norm(strike_dir_2d)
    strike_dir_3d = np.array([
        horiz[0] * np.cos(angle_rad),
        horiz[1] * np.cos(angle_rad),
        -np.sin(angle_rad)
    ])
    strike_dir_3d = strike_dir_3d / np.linalg.norm(strike_dir_3d)

    T_current = controller.get_current_T()
    q_current = controller.get_current_q()

    ball_h = cue_pos[2]
    trajectory, phases = traj_planner.plan_strike(
        T_current=T_current, ball_pos=cue_pos,
        strike_direction=strike_dir_3d, strike_speed=strike_speed,
        approach_dist=STRIKE_APPROACH_DIST, follow_dist=STRIKE_FOLLOW_DIST,
        strike_height=ball_h, tool_offset=tool_offset
    )

    q_traj = controller.ik.solve_trajectory(q_current, trajectory)

    dt = 0.002
    times = np.arange(len(trajectory)) * dt
    joint_angles = np.array([q.flatten() for q in q_traj])
    task_positions = np.array([T[:3, 3] for T in trajectory])

    save_dir = os.path.dirname(os.path.abspath(__file__))
    phase_colors = {'approach': '#3498db', 'strike': '#e74c3c', 'follow': '#2ecc71'}

    # ── Figure 1: Joint Angles ──
    fig1, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig1.suptitle('3-CUSHION — Joint Angles vs Time', fontsize=16, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        q_deg = np.degrees(joint_angles[:, i])
        for phase_name, (start, end) in phases.items():
            ax.plot(times[start:end], q_deg[start:end],
                    color=phase_colors[phase_name], linewidth=1.5, label=phase_name.capitalize())
        ax.set_ylabel(f'Joint {i+1} (deg)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(save_dir, 'plot_joints_maze.png')
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ── Figure 2: Task Space XYZ ──
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle('3-CUSHION — EE Position vs Time', fontsize=16, fontweight='bold')
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    for i, ax in enumerate(axes2):
        for phase_name, (start, end) in phases.items():
            ax.plot(times[start:end], task_positions[start:end, i],
                    color=phase_colors[phase_name], linewidth=1.5, label=phase_name)
        ax.axhline(y=cue_pos[i], color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Cue Ball')
        ax.set_ylabel(labels[i], fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, ncol=4, loc='best')
    axes2[-1].set_xlabel('Time (s)', fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    path2 = os.path.join(save_dir, 'plot_taskspace_maze.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: 3D Trajectory ──
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_title('3-CUSHION — 3D EE Trajectory', fontsize=14, fontweight='bold')
    for phase_name, (start, end) in phases.items():
        ax3.plot(task_positions[start:end, 0], task_positions[start:end, 1],
                 task_positions[start:end, 2],
                 color=phase_colors[phase_name], linewidth=2, label=phase_name.capitalize())
    ax3.scatter(*cue_pos, color='orange', s=100, marker='o', label='Cue Ball', zorder=5)
    ax3.scatter(*task_positions[0], color='black', s=60, marker='^', label='Start', zorder=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    path3 = os.path.join(save_dir, 'plot_3d_maze.png')
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    print(f"  Saved: {path3}")

    # ── Figure 4: 공 궤적 (2D 평면) ──
    if best.get('ball_path') and len(best['ball_path']) > 1:
        fig4, ax4 = plt.subplots(figsize=(10, 7))
        ax4.set_title('3-CUSHION — Planned Ball Trajectory (Headless PyBullet)',
                       fontsize=14, fontweight='bold')

        ball_path = np.array(best['ball_path'])
        ax4.plot(ball_path[:, 0], ball_path[:, 1], 'b-', linewidth=2, label='Cue path')
        ax4.scatter(ball_path[0, 0], ball_path[0, 1], color='white', edgecolor='black',
                    s=150, zorder=5, label='Cue start')

        ax4.scatter(target_pos[0], target_pos[1], color='gold', edgecolor='black',
                    s=150, zorder=5, label='Target 1')
        if ball2_pos is not None:
            ax4.scatter(ball2_pos[0], ball2_pos[1], color='red', edgecolor='black',
                        s=150, zorder=5, label='Target 2')

        # 장애물
        for ox, oy, orr in obstacles:
            circle = plt.Circle((ox, oy), orr, color='gray', alpha=0.5)
            ax4.add_patch(circle)

        # 테이블 경계
        b = env.table_bounds
        rect = plt.Rectangle((b['x_min'], b['y_min']),
                              b['x_max']-b['x_min'], b['y_max']-b['y_min'],
                              fill=False, edgecolor='brown', linewidth=2)
        ax4.add_patch(rect)

        ax4.set_xlabel('X (m)', fontsize=12)
        ax4.set_ylabel('Y (m)', fontsize=12)
        ax4.set_aspect('equal')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        info_text = (f"Score: {best['score']:.0f}  |  "
                     f"Cushions: {best['cushion_count']}  |  "
                     f"Hit T1: {best['hit_t1']}  |  Hit T2: {best['hit_t2']}")
        ax4.set_xlabel(info_text, fontsize=11)

        fig4.tight_layout()
        path4 = os.path.join(save_dir, 'plot_ball_trajectory_maze.png')
        fig4.savefig(path4, dpi=150)
        plt.close(fig4)
        print(f"  Saved: {path4}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='쓰리쿠션 당구 시뮬레이션')
    parser.add_argument('--obstacles', type=int, default=0, help='장애물 개수 (기본: 0)')
    parser.add_argument('--attempts', type=int, default=3, help='시도 횟수 (기본: 3)')
    parser.add_argument('--view-time', type=int, default=15, help='결과 확인 시간 (초)')
    parser.add_argument('--seed', type=int, default=None, help='장애물 랜덤 시드')
    parser.add_argument('--no-plot', action='store_true', help='플롯 생성 생략')
    args = parser.parse_args()

    run_maze(
        num_obstacles=args.obstacles,
        max_attempts=args.attempts,
        view_time=args.view_time,
        seed=args.seed,
        save_plots=not args.no_plot
    )
