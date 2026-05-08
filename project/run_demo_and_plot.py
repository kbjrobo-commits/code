"""
GUI 데모 실행 + 궤적 플롯 생성
=================================
1. 미니골프 GUI 데모 → 자동 종료
2. 포켓볼 GUI 데모 → 자동 종료
3. 궤적 플롯 (joint angles, task space XYZ) 저장

사용법:
    python project/run_demo_and_plot.py --demo minigolf
    python project/run_demo_and_plot.py --demo billiards
    python project/run_demo_and_plot.py --demo both
    python project/run_demo_and_plot.py --plot-only
"""
import argparse
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from project.config import *
from project.robot_controller import RobotController
from project.trajectory_planner import StrikeTrajectoryPlanner
from project.ik_solver import IKSolver
from project.physics.shot_planner import MinigolfShotPlanner, BilliardsShotPlanner


# ============================================================
# 궤적 데이터 수집 + 플롯
# ============================================================
def generate_trajectory_data(demo_type='minigolf'):
    """Headless에서 궤적 데이터를 생성하고 반환"""
    controller = RobotController(mode='sim', headless=True)
    controller.connect()
    controller.move_home()

    T_home = controller.get_current_T()
    q_home = controller.get_current_q()

    # 도구 오프셋: 컴팩트 헤드 길이 + 공 반지름
    tool_off = TOOL_HEAD_LENGTH + BILLIARD_BALL_RADIUS

    if demo_type == 'minigolf':
        ball_pos = [0.45, -0.10, 0.035]
        hole_pos = [0.55, 0.12, 0.005]
        planner = MinigolfShotPlanner()
        strike_dir, speed = planner.plan_shot(ball_pos, hole_pos)
        tool_off = TOOL_HEAD_LENGTH + MINIGOLF_BALL_RADIUS
    else:
        CY = BILLIARD_TABLE_CENTER_Y
        W = BILLIARD_TABLE_WIDTH
        H = BILLIARD_TABLE_SURFACE_HEIGHT
        TH = BILLIARD_TABLE_HEIGHT
        ball_h = H + TH / 2 + BILLIARD_BALL_RADIUS + 0.001
        ball_pos = [0.5, CY - W / 4, ball_h]
        target_pos = [0.5, CY + W / 8, ball_h]
        pocket_pos = [0.5, CY + W / 2, H + TH / 2]
        planner = BilliardsShotPlanner()
        strike_dir, speed, _ = planner.plan_shot(ball_pos, target_pos, pocket_pos)

        # 빌리아드: 대각선 방향으로 변환
        angle_rad = np.radians(BILLIARD_STRIKE_ANGLE_DEG)
        horiz = strike_dir[:2] / np.linalg.norm(strike_dir[:2])
        strike_dir = np.array([
            horiz[0] * np.cos(angle_rad),
            horiz[1] * np.cos(angle_rad),
            -np.sin(angle_rad)
        ])
        strike_dir = strike_dir / np.linalg.norm(strike_dir)

    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)
    trajectory, phases = traj_planner.plan_strike(
        T_current=T_home,
        ball_pos=ball_pos,
        strike_direction=strike_dir,
        strike_speed=speed,
        approach_dist=0.08,
        follow_dist=0.10,
        strike_height=ball_pos[2],
        tool_offset=tool_off
    )

    q_traj = controller.ik.solve_trajectory(q_home, trajectory)

    dt = 0.002
    times = np.arange(len(trajectory)) * dt
    joint_angles = np.array([q.flatten() for q in q_traj])
    task_positions = np.array([T[:3, 3] for T in trajectory])
    fk_positions = np.array([controller._pinModel.FK(q)[:3, 3] for q in q_traj])

    controller.disconnect()

    return {
        'times': times,
        'joint_angles': joint_angles,
        'task_positions': task_positions,
        'fk_positions': fk_positions,
        'phases': phases,
        'demo_type': demo_type,
        'dt': dt,
        'ball_pos': np.array(ball_pos),
        'strike_dir': strike_dir,
        'speed': speed,
    }


def plot_trajectory(data, save_dir='project'):
    """궤적 플롯 생성 및 저장"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    demo = data['demo_type']
    times = data['times']
    joint_angles = data['joint_angles']
    task_pos = data['task_positions']
    fk_pos = data['fk_positions']
    phases = data['phases']

    phase_colors = {'approach': '#3498db', 'strike': '#e74c3c', 'follow': '#2ecc71'}

    # --- Figure 1: Joint Angles ---
    fig1, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig1.suptitle(f'{demo.upper()} — Joint Angles vs Time', fontsize=16, fontweight='bold')
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    for i, ax in enumerate(axes.flat):
        q_deg = np.degrees(joint_angles[:, i])
        for phase_name, (start, end) in phases.items():
            ax.plot(times[start:end], q_deg[start:end],
                    color=phase_colors[phase_name], linewidth=1.5, label=phase_name.capitalize())
        ax.set_ylabel(f'{joint_names[i]} (deg)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(save_dir, f'plot_joints_{demo}.png')
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # --- Figure 2: Task Space XYZ ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle(f'{demo.upper()} — EE Position vs Time', fontsize=16, fontweight='bold')
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    for i, ax in enumerate(axes2):
        for phase_name, (start, end) in phases.items():
            ax.plot(times[start:end], task_pos[start:end, i],
                    color=phase_colors[phase_name], linewidth=1.5, label=f'{phase_name}')
        if i < 3:
            ax.axhline(y=data['ball_pos'][i], color='orange', linestyle=':',
                       linewidth=1, alpha=0.7, label='Ball')
        ax.set_ylabel(labels[i], fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, ncol=3, loc='best')
    axes2[-1].set_xlabel('Time (s)', fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    path2 = os.path.join(save_dir, f'plot_taskspace_{demo}.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # --- Figure 3: 3D Trajectory ---
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_title(f'{demo.upper()} — 3D EE Trajectory', fontsize=14, fontweight='bold')
    for phase_name, (start, end) in phases.items():
        ax3.plot(task_pos[start:end, 0], task_pos[start:end, 1], task_pos[start:end, 2],
                 color=phase_colors[phase_name], linewidth=2, label=phase_name.capitalize())
    ax3.scatter(*data['ball_pos'], color='orange', s=100, marker='o', label='Ball', zorder=5)
    ax3.scatter(*task_pos[0], color='black', s=60, marker='^', label='Start', zorder=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    path3 = os.path.join(save_dir, f'plot_3d_{demo}.png')
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    print(f"  Saved: {path3}")

    return [path1, path2, path3]


# ============================================================
# GUI 데모 실행
# ============================================================
def run_gui_demo(demo_type, view_time=15, num_obstacles=5, seed=None):
    """GUI 모드 데모 실행"""
    import pybullet as p

    titles = {'minigolf': 'Mini-Golf', 'billiards': 'Billiards', 'maze': 'Maze (3-Cushion)'}
    title = titles.get(demo_type, demo_type)
    print(f"\n{'='*60}")
    print(f"  {title} GUI Demo")
    print(f"{'='*60}")

    controller = RobotController(mode='sim')
    controller.connect()
    time.sleep(2)
    controller.move_home()
    time.sleep(2)

    robot_id = controller.pb.my_robot.robotId
    ee_link = controller.pb.my_robot.RobotEEJointIdx[-1]
    perception = None

    if demo_type == 'minigolf':
        from project.environment.minigolf_env import MiniGolfEnvironment
        env = MiniGolfEnvironment(controller.pb.ClientId)
        env.setup(
            ball_pos=[0.45, -0.10, 0.035],
            hole_pos=[0.55, 0.12, 0.005],
            terrain_seed=42
        )
        env.disable_robot_env_collision(robot_id)
        env.attach_compact_tool(robot_id, ee_link)
        env.disable_tool_env_collision()
        tool_offset = TOOL_HEAD_LENGTH + MINIGOLF_BALL_RADIUS
        shot_planner = MinigolfShotPlanner()
    elif demo_type == 'maze':
        from project.environment.maze_env import MazeEnvironment
        from project.physics.cushion_planner import CushionShotPlanner
        from project.perception import SimPerception
        env = MazeEnvironment(controller.pb.ClientId)
        CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
        H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
        ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
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
    else:  # billiards
        from project.environment.billiards_env import BilliardsEnvironment
        env = BilliardsEnvironment(controller.pb.ClientId)
        CY = BILLIARD_TABLE_CENTER_Y
        W = BILLIARD_TABLE_WIDTH
        H = BILLIARD_TABLE_SURFACE_HEIGHT
        TH = BILLIARD_TABLE_HEIGHT
        ball_h = H + TH / 2 + BILLIARD_BALL_RADIUS + 0.001
        env.setup(
            cue_pos=[0.5, CY - W / 4, ball_h],
            target_pos=[0.5, CY + W / 8, ball_h]
        )
        env.disable_robot_env_collision(robot_id)
        env.attach_compact_tool(robot_id, ee_link)
        env.disable_tool_env_collision()
        tool_offset = TOOL_HEAD_LENGTH + BILLIARD_BALL_RADIUS
        shot_planner = BilliardsShotPlanner()

    # PD 게인 강화 — 도구 질량에 의한 흔들림 방지
    controller.boost_pd_gains(kp=800, kd=40)

    # 도구 부착 후 안정화 대기
    time.sleep(3)

    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)

    from project.state_machine import AutonomousStateMachine
    sm = AutonomousStateMachine(
        controller=controller,
        environment=env,
        shot_planner=shot_planner,
        traj_planner=traj_planner,
        demo_type=demo_type,
        tool_offset=tool_offset,
        perception=perception
    )

    success = sm.run(max_attempts=3)

    print(f"\n  Result: {'SUCCESS' if success else 'COMPLETED'}")
    print(f"  Viewing for {view_time} seconds...")
    time.sleep(view_time)

    if hasattr(env, 'cleanup'):
        env.cleanup()
    controller.disconnect()
    print(f"  {title} demo finished.\n")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Indy7 Demo + Trajectory Plots')
    parser.add_argument('--demo', type=str, default='both',
                        choices=['minigolf', 'billiards', 'maze', 'both', 'none'],
                        help='Demo to run')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip GUI demo, generate plots only')
    parser.add_argument('--view-time', type=int, default=15,
                        help='Seconds to view each demo')
    args = parser.parse_args()

    demos = []
    if args.plot_only or args.demo == 'none':
        demos = []
    elif args.demo == 'both':
        demos = ['minigolf', 'billiards']
    else:
        demos = [args.demo]

    for demo in demos:
        run_gui_demo(demo, view_time=args.view_time)

    print("\n" + "=" * 60)
    print("  Generating Trajectory Plots")
    print("=" * 60)
    for demo_type in ['minigolf', 'billiards']:
        print(f"\n  [{demo_type.upper()}] Computing trajectory data...")
        data = generate_trajectory_data(demo_type)
        print(f"    Trajectory: {len(data['times'])} points")
        print(f"    Strike dir: {data['strike_dir']}, speed: {data['speed']:.3f}")
        print(f"  [{demo_type.upper()}] Generating plots...")
        plot_trajectory(data)

    print(f"\n{'='*60}")
    print("  ALL DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
