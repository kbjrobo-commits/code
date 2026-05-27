"""
?곕━荑좎뀡 ?밴뎄 ?쒕??덉씠?????⑥씪 ?ㅽ뻾 ?뚯씪
=============================================
???뚯씪 ?섎굹留??ㅽ뻾?섎㈃ ?꾨옒 ?꾩껜媛 ?먮룞?쇰줈 吏꾪뻾?⑸땲??

  1. GUI ?쒕??덉씠???ㅽ뻾 (PyBullet)
  2. 3怨?headless ?먯깋 (理쒖쟻 ?寃?媛곷룄/?띾룄)
  3. ?ㅼ쨷 ?꾨낫 IK+?μ븷臾?寃利?  4. 濡쒕큸 ?寃??ㅽ뻾 (3???쒕룄)
  5. 怨?沅ㅼ쟻 ?쒓컖??(PyBullet debug lines)
  6. 沅ㅼ쟻 ?뚮’ ???(愿?덇컖, ?쒖뒪?ш났媛? 3D)

?ъ슜踰?
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
    """?곕━荑좎뀡 ?꾩껜 ?뚯씠?꾨씪???ㅽ뻾"""

    print(f"\n{'='*60}")
    print(f"  2-Cushion Billiards Simulation")
    print(f"{'='*60}")
    print(f"  Obstacles: {num_obstacles}")
    print(f"  Attempts: {max_attempts}")
    print(f"  Tool: length={TOOL_HEAD_LENGTH*100:.0f}cm, mass={TOOL_HEAD_MASS:.1f}kg")
    print(f"  Friction: lateral={MAZE_BALL_FRICTION}, rolling={MAZE_BALL_ROLLING_FRICTION}")
    print(f"{'='*60}\n")

    # ?? 1. 濡쒕큸 + ?섍꼍 珥덇린????
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
        num_obstacles=num_obstacles,
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
    time.sleep(3)

    # ?? 1b. GUI ?뚰겕?ㅽ럹?댁뒪 諛섍꼍 ?쒖떆 ??
    import pybullet as _p
    client = controller.pb.ClientId
    surface_z = MAZE_TABLE_SURFACE_HEIGHT + MAZE_TABLE_HEIGHT / 2 + 0.002
    n_seg = 64
    for radius, color, label in [
        (0.70, [0, 1, 0], "safe"),    # ?덉쟾 踰붿쐞 (?뱀깋)
        (0.80, [1, 0.5, 0], "max"),   # 臾쇰━???쒓퀎 (二쇳솴)
    ]:
        for i in range(n_seg):
            th0 = 2 * np.pi * i / n_seg
            th1 = 2 * np.pi * (i + 1) / n_seg
            x0, y0 = radius * np.cos(th0), radius * np.sin(th0)
            x1, y1 = radius * np.cos(th1), radius * np.sin(th1)
            _p.addUserDebugLine(
                [x0, y0, surface_z], [x1, y1, surface_z],
                color, lineWidth=2, lifeTime=0, physicsClientId=client)
    # 濡쒕큸 踰좎씠???쒖떆
    _p.addUserDebugText("ROBOT", [0, 0, surface_z + 0.02],
                        textColorRGB=[1, 1, 1], textSize=1.2,
                        physicsClientId=client)
    print(f"  [VIS] Workspace: safe=0.70m (green), max=0.80m (orange)")

    # ?? 2. 沅ㅼ쟻 ?앹꽦湲?+ ?곹깭癒몄떊 ??
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

    # ?? 3. ?ㅽ뻾 ??
    success = sm.run(max_attempts=max_attempts)

    print(f"\n  Result: {'SUCCESS' if success else 'COMPLETED'}")
    print(f"  Viewing for {view_time} seconds...")
    time.sleep(view_time)

    # ?? 4. 沅ㅼ쟻 ?뚮’ ?앹꽦 ??
    if save_plots:
        print(f"\n{'='*60}")
        print(f"  沅ㅼ쟻 ?뚮’ ?앹꽦 以?..")
        print(f"{'='*60}")
        try:
            _generate_maze_plots(controller, env, shot_planner, traj_planner,
                                 tool_offset, perception, sm)
        except Exception as e:
            print(f"  [WARNING] ?뚮’ ?앹꽦 ?ㅽ뙣: {e}")

    if hasattr(env, 'cleanup'):
        env.cleanup()
    controller.disconnect()
    print(f"\n  2-Cushion simulation finished.\n")


def _generate_maze_plots(controller, env, planner, traj_planner,
                         tool_offset, perception, state_machine=None):
    """沅ㅼ쟻 ?곗씠?곕줈 ?뚮’ ?앹꽦"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    executed_candidate = None
    executed_trajectory = None
    executed_phases = None
    executed_q_traj = None
    if state_machine is not None:
        executed_candidate = getattr(state_machine, 'last_chosen_candidate', None)
        executed_trajectory = getattr(state_machine, 'last_executed_trajectory', None)
        executed_phases = getattr(state_machine, 'last_executed_phases', None)
        executed_q_traj = getattr(state_machine, 'last_executed_q_trajectory', None)

    # ?꾩옱 怨??꾩튂濡??섑뵆 沅ㅼ쟻 怨꾩궛
    scan = getattr(state_machine, 'last_executed_scan', None) if state_machine else None
    if scan is None:
        scan = perception.scan_environment()
    cue_pos = scan['cue_pos']
    target_pos = scan['target_pos']
    ball2_pos = scan.get('ball2_pos')
    obstacles = scan.get('obstacles', [])

    best = executed_candidate
    trajectory = executed_trajectory
    phases = executed_phases
    q_traj = executed_q_traj

    if best is None or trajectory is None or phases is None:
        print("  [PLOT] No executed candidate cached; replanning as fallback.")
        candidates = planner.plan_shot(cue_pos, target_pos, obstacles,
                                       ball2_pos=ball2_pos)
        if not candidates:
            print("  ?뚮’ ?앹꽦 遺덇?: ?꾨낫 ?놁쓬")
            return

        best = candidates[0]

    strike_dir_2d = best['strike_dir']
    strike_speed = best['strike_speed']

    # 3D 諛⑺뼢 蹂??    angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
    if q_traj is None:
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

    # ?? Figure 1: Joint Angles ??
    fig1, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig1.suptitle('2-CUSHION Joint Angles vs Time', fontsize=16, fontweight='bold')
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

    # ?? Figure 2: Task Space XYZ ??
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle('2-CUSHION EE Position vs Time', fontsize=16, fontweight='bold')
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

    # ?? Figure 3: 3D Trajectory ??
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_title('2-CUSHION 3D EE Trajectory', fontsize=14, fontweight='bold')
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

    # ?? Figure 4: 怨?沅ㅼ쟻 (2D ?됰㈃) ??
    if best.get('ball_path') and len(best['ball_path']) > 1:
        fig4, ax4 = plt.subplots(figsize=(10, 7))
        ax4.set_title('2-CUSHION Planned Ball Trajectory (Headless PyBullet)',
                       fontsize=14, fontweight='bold')

        ball_path = np.array(best['ball_path'])
        ax4.plot(ball_path[:, 0], ball_path[:, 1], 'b-', linewidth=2, label='Cue path')
        ax4.scatter(ball_path[0, 0], ball_path[0, 1], color='white', edgecolor='black',
                    s=150, zorder=5, label='Cue start')

        def _plot_moving_path(path, color, label):
            if path is None or len(path) <= 1:
                return
            pts = np.array(path)
            if pts.ndim != 2 or pts.shape[1] < 2:
                return
            xy = pts[:, :2]
            keep = np.ones(len(xy), dtype=bool)
            if len(xy) > 1:
                keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.003
            xy = xy[keep]
            if len(xy) > 1:
                ax4.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2,
                         linestyle='--', label=label)

        _plot_moving_path(best.get('tgt1_path'), 'gold', 'Target 1 path')
        _plot_moving_path(best.get('tgt2_path'), 'red', 'Target 2 path')

        ax4.scatter(target_pos[0], target_pos[1], color='gold', edgecolor='black',
                    s=150, zorder=5, label='Target 1')
        if ball2_pos is not None:
            ax4.scatter(ball2_pos[0], ball2_pos[1], color='red', edgecolor='black',
                        s=150, zorder=5, label='Target 2')

        for ox, oy, orr in obstacles:
            circle = plt.Circle((ox, oy), orr, color='gray', alpha=0.5)
            ax4.add_patch(circle)

        # ?뚯씠釉?寃쎄퀎
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

        events = best.get('events', [])
        valid_2c = best.get('valid_2cushion', False)
        valid_3c = best.get('valid_3cushion', False)
        info_text = (f"Score: {best['score']:.0f}  |  Valid2C: {valid_2c}  |  "
                     f"Valid3C: {valid_3c}  |  Cushions: {best.get('cushion_count', 0)}  |  "
                     f"Hit T1: {best.get('hit_t1')}  |  Hit T2: {best.get('hit_t2')}\n"
                     f"Tool/pred ball speed: "
                     f"{best.get('tool_speed_cmd', strike_speed):.3f}/"
                     f"{best.get('pred_ball_speed', best.get('ball_speed', 0.0)):.3f} m/s  |  "
                     f"Events: {events}")
        ax4.set_xlabel(info_text, fontsize=10)

        fig4.tight_layout()
        path4 = os.path.join(save_dir, 'plot_ball_trajectory_maze.png')
        fig4.savefig(path4, dpi=150)
        plt.close(fig4)
        print(f"  Saved: {path4}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2-cushion maze simulation')
    parser.add_argument('--obstacles', type=int, default=0, help='number of obstacles (default: 0)')
    parser.add_argument('--attempts', type=int, default=3, help='number of attempts (default: 3)')
    parser.add_argument('--view-time', type=int, default=15, help='result viewing time in seconds')
    parser.add_argument('--seed', type=int, default=None, help='obstacle random seed')
    parser.add_argument('--no-plot', action='store_true', help='skip plot generation')
    args = parser.parse_args()

    run_maze(
        num_obstacles=args.obstacles,
        max_attempts=args.attempts,
        view_time=args.view_time,
        seed=args.seed,
        save_plots=not args.no_plot
    )
