"""
Barebone diagnostic: 240Hz trajectory streaming strike 
- Measures actual EE tracking error during strike
- Compares headless vs GUI ball trajectory
"""
import numpy as np
import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.config import *
from project.robot_controller import RobotController
from project.trajectory_planner import StrikeTrajectoryPlanner, TrajectoryPlanner
from project.environment.maze_env import MazeEnvironment
import pybullet as pb


def run_headless(angle_rad, ee_speed, cue_3d, tgt1_3d, tgt2_3d):
    """Headless shot — CushionShotPlanner와 동일한 로봇 PD 사용"""
    from project.physics.cushion_planner import CushionShotPlanner
    from project.environment.maze_env import MazeEnvironment

    # 임시 table_bounds
    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    L = MAZE_TABLE_LENGTH
    bounds = {
        'x_min': 0.5 - L/2, 'x_max': 0.5 + L/2,
        'y_min': CY - W/2, 'y_max': CY + W/2,
    }

    planner = CushionShotPlanner(table_bounds=bounds)
    env = planner._create_robot_env(cue_3d, tgt1_3d, tgt2_3d, [])
    (sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids, obstacle_ids,
     tool_id, robot_id, movable_joints, ee_link, tool_cid,
     ik_solver, pin_model) = env

    score, info = planner._simulate_one_robot(
        sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids, tool_id,
        robot_id, movable_joints, ik_solver, pin_model,
        cue_3d, tgt1_3d, tgt2_3d,
        angle_rad, ee_speed
    )

    cue_final, _ = pb.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
    pb.disconnect(sim_id)
    return np.array(cue_final)


def main():
    print(f"\n{'='*60}")
    print(f"  240Hz Streaming Strike Test")
    print(f"{'='*60}")

    CY, W = MAZE_TABLE_CENTER_Y, MAZE_TABLE_WIDTH
    H, TH = MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT
    ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001
    ee_speed = 1.0

    cue_pos = np.array([0.5, CY - W/4, ball_h])
    tgt1_pos = np.array([0.5, CY + W/8, ball_h])
    tgt2_pos = np.array([0.5 + MAZE_TABLE_LENGTH/6, CY, ball_h])

    dir_2d = tgt1_pos[:2] - cue_pos[:2]
    angle_rad = np.arctan2(dir_2d[1], dir_2d[0])

    print(f"  Angle: {np.degrees(angle_rad):.1f} deg, Speed: {ee_speed} m/s")

    # 1. Headless (로봇 PD 사용)
    hl_final = run_headless(angle_rad, ee_speed, cue_pos, tgt1_pos, tgt2_pos)
    print(f"  Headless final: [{hl_final[0]:.4f}, {hl_final[1]:.4f}]")

    # 2. GUI
    ctrl = RobotController(mode='sim')
    ctrl.connect()
    time.sleep(2)
    ctrl.move_home()
    time.sleep(2)

    robot_id = ctrl.pb.my_robot.robotId
    ee_link = ctrl.pb.my_robot.RobotEEJointIdx[-1]

    env = MazeEnvironment(ctrl.pb.ClientId)
    env.setup(cue_pos=list(cue_pos), target_pos=list(tgt1_pos),
              ball2_pos=list(tgt2_pos), num_obstacles=0)
    env.disable_robot_env_collision(robot_id)
    env.attach_compact_tool(robot_id, ee_link)
    env.disable_tool_env_collision()
    ctrl.boost_pd_gains(kp=800, kd=40)
    ctrl.set_environment(env)
    time.sleep(3)

    # EE tracking diagnostic: hook into _thread_pre to log EE position
    robot = ctrl.pb.my_robot
    ee_log = []
    _orig_post = ctrl.pb._thread_post
    def _logging_post():
        _orig_post()
        if hasattr(robot, '_strike_buf') and robot._strike_idx > 0:
            ee_pos = robot._T_end[:3, 3].copy()
            ee_log.append((robot._strike_idx - 1, ee_pos))
    ctrl.pb._thread_post = _logging_post

    # Generate & execute trajectory
    tool_offset = TOOL_HEAD_LENGTH + MAZE_BALL_RADIUS
    angle_strike = np.radians(MAZE_STRIKE_ANGLE_DEG)
    horiz = dir_2d / np.linalg.norm(dir_2d)
    strike_dir_3d = np.array([
        horiz[0] * np.cos(angle_strike),
        horiz[1] * np.cos(angle_strike),
        -np.sin(angle_strike)
    ])
    strike_dir_3d /= np.linalg.norm(strike_dir_3d)

    traj_planner = StrikeTrajectoryPlanner(approach_duration=3.0, dt=0.002)
    T_home = ctrl.get_current_T()
    trajectory, phases = traj_planner.plan_strike(
        T_current=T_home, ball_pos=cue_pos,
        strike_direction=strike_dir_3d, strike_speed=ee_speed,
        approach_dist=STRIKE_APPROACH_DIST, follow_dist=STRIKE_FOLLOW_DIST,
        strike_height=cue_pos[2], tool_offset=tool_offset
    )

    ctrl.execute_trajectory(trajectory, dt=0.002, phase_indices=phases,
                            strike_speed=ee_speed)

    # Restore post
    ctrl.pb._thread_post = _orig_post

    env.wait_balls_stop(timeout=8.0)
    gui_final = env.get_cue_ball_position()
    print(f"  GUI final:     [{gui_final[0]:.4f}, {gui_final[1]:.4f}]")

    diff = np.linalg.norm(gui_final[:2] - hl_final[:2])
    print(f"\n  XY difference: {diff*1000:.1f} mm")
    print(f"  {'GOOD (<50mm)' if diff < 0.05 else 'MISMATCH (>50mm)'}")

    # Print EE tracking diagnostics
    if ee_log:
        print(f"\n  EE tracking log ({len(ee_log)} entries):")
        for idx, pos in ee_log[::max(1, len(ee_log)//10)]:
            print(f"    step {idx:3d}: EE=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    time.sleep(3)
    ctrl.disconnect()


if __name__ == '__main__':
    main()
