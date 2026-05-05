"""
자율 루프 State Machine
=========================
SCAN → THINK → ALIGN & STRIKE → OBSERVE & RECALCULATE
"""
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class AutonomousStateMachine:
    """자율 타격 루프 상태 머신

    States:
        SCAN: 공/홀/포켓 위치 인식
        THINK: 최적 타격 벡터 계산
        STRIKE: 타격 궤적 생성 및 실행
        OBSERVE: 결과 관찰 (성공/실패 판정)
    """

    def __init__(self, controller, environment, shot_planner, traj_planner,
                 demo_type='minigolf', tool_offset=0.0):
        """
        Args:
            controller: RobotController 인스턴스
            environment: MiniGolfEnvironment 또는 BilliardsEnvironment
            shot_planner: ShotPlanner 인스턴스
            traj_planner: StrikeTrajectoryPlanner 인스턴스
            demo_type: 'minigolf' or 'billiards'
            tool_offset: EE에서 도구 끝까지 거리 (m)
        """
        self.controller = controller
        self.env = environment
        self.planner = shot_planner
        self.traj = traj_planner
        self.demo_type = demo_type
        self.tool_offset = tool_offset
        self.state = 'INIT'
        self.attempt = 0
        self.history = []

    def run(self, max_attempts=5):
        """자율 루프 실행

        Returns:
            success: 최종 성공 여부
        """
        self.state = 'SCAN'
        self.attempt = 0

        print("=" * 60)
        print(f"  Autonomous {self.demo_type.upper()} State Machine Started")
        print("=" * 60)

        successes = 0
        while self.attempt < max_attempts:
            self.attempt += 1
            print(f"\n{'='*40}")
            print(f"  Attempt {self.attempt}/{max_attempts}")
            print(f"{'='*40}")

            # === SCAN ===
            print(f"\n[STATE: SCAN] Detecting objects...")
            scan_data = self._scan()
            print(f"  Scan result: {scan_data}")

            # === THINK ===
            print(f"\n[STATE: THINK] Computing optimal strike...")
            plan = self._think(scan_data)
            print(f"  Strike direction: {plan['strike_dir']}")
            print(f"  Strike speed: {plan['strike_speed']:.3f} m/s")

            # === ALIGN & STRIKE ===
            print(f"\n[STATE: ALIGN & STRIKE] Executing...")
            self._strike_skipped = False
            self._strike(scan_data, plan)

            if getattr(self, '_strike_skipped', False):
                print(f"  Strike skipped (ball unreachable)")
                if self.attempt < max_attempts:
                    self.controller.move_home()
                    time.sleep(1)
                continue

            print(f"  Strike executed!")

            # === OBSERVE ===
            print(f"\n[STATE: OBSERVE] Waiting for result...")
            success = self._observe()

            dist = self._get_result_distance()
            self.history.append({
                'attempt': self.attempt,
                'success': success,
                'distance': dist,
                'plan': plan
            })

            if success:
                print(f"\n{'*'*40}")
                print(f"  SUCCESS! Goal achieved in attempt {self.attempt}!")
                print(f"{'*'*40}")
                successes += 1
            else:
                print(f"  Miss! Distance to target: {dist:.4f} m")

            if self.attempt < max_attempts:
                if success:
                    print(f"  Continuing to next round...")
                else:
                    print(f"  Returning to SCAN for retry...")
                self.controller.move_home()
                time.sleep(1)

        print(f"\n{'='*40}")
        print(f"  FINAL: {successes}/{max_attempts} successful")
        print(f"{'='*40}")
        return successes > 0

    def _scan(self):
        """SCAN: 공/홀/포켓 위치 인식"""
        if self.demo_type == 'minigolf':
            ball_pos = self.env.get_ball_position()
            return {
                'ball_pos': ball_pos,
                'hole_pos': self.env.hole_pos
            }
        else:  # billiards
            cue_pos = self.env.get_cue_ball_position()
            target_pos = self.env.get_target_ball_position()
            return {
                'cue_pos': cue_pos,
                'target_pos': target_pos,
                'pocket_positions': self.env.pocket_positions
            }

    def _think(self, scan_data):
        """THINK: 최적 타격 벡터 계산"""
        if self.demo_type == 'minigolf':
            terrain_path = getattr(self.env, 'terrain_obj_path', None)
            terrain_offset = getattr(self.env, 'terrain_offset', [0.5, 0, 0])
            if terrain_path:
                print(f"  Running physics-based Grid Search...")
                strike_dir, speed = self.planner.plan_shot_physics_search(
                    scan_data['ball_pos'], scan_data['hole_pos'],
                    terrain_path, terrain_offset
                )
            else:
                strike_dir, speed = self.planner.plan_shot(
                    scan_data['ball_pos'], scan_data['hole_pos']
                )
            return {
                'strike_dir': strike_dir,
                'strike_speed': speed,
                'ball_pos': scan_data['ball_pos']
            }
        else:  # billiards
            result = self.planner.find_best_pocket_shot(
                scan_data['cue_pos'],
                scan_data['target_pos'],
                scan_data['pocket_positions']
            )
            return {
                'strike_dir': result['strike_dir'],
                'strike_speed': result['strike_speed'],
                'ball_pos': scan_data['cue_pos'],
                'contact_point': result['contact_point'],
                'target_pocket': result['pocket']
            }

    def _strike(self, scan_data, plan):
        """ALIGN & STRIKE: 궤적 생성 및 실행"""
        T_current = self.controller.get_current_T()
        ball_pos = plan['ball_pos']
        strike_dir_2d = plan['strike_dir']  # 수평 타격 방향
        strike_speed = plan['strike_speed']

        # 도달 가능성 확인 — 공이 로봇 base(원점)에서 너무 멀면 리셋
        ball_dist_from_base = np.linalg.norm(ball_pos[:2])
        if ball_dist_from_base > 0.70:
            print(f"  [WARNING] Ball too far from robot ({ball_dist_from_base:.3f}m > 0.70m).")
            if self.demo_type == 'billiards' and hasattr(self.env, 'reset_balls'):
                print(f"  Resetting cue ball to start position...")
                self.env.reset_balls(cue_pos=self.env.cue_start_pos)
                import time as t
                t.sleep(0.5)
                ball_pos = self.env.get_cue_ball_position()
                plan['ball_pos'] = ball_pos
            else:
                print(f"  Skipping strike.")
                self._strike_skipped = True
                return

        # 타격 높이: 공 중심 높이
        strike_height = ball_pos[2]

        # 빌리아드: 위에서 대각선으로 내려치기
        if self.demo_type == 'billiards':
            from project.config import BILLIARD_STRIKE_ANGLE_DEG
            angle_rad = np.radians(BILLIARD_STRIKE_ANGLE_DEG)
            horiz = np.array(strike_dir_2d[:2]).flatten()
            horiz_norm = np.linalg.norm(horiz)
            if horiz_norm > 1e-6:
                horiz = horiz / horiz_norm
            strike_dir_3d = np.array([
                horiz[0] * np.cos(angle_rad),
                horiz[1] * np.cos(angle_rad),
                -np.sin(angle_rad)
            ])
            strike_dir_3d = strike_dir_3d / np.linalg.norm(strike_dir_3d)
        else:
            # 미니골프: 수평 타격
            strike_dir_3d = np.array(strike_dir_2d).flatten()

        # 궤적 생성 (follow_dist=0.05: 공 너머 5cm까지 가속 궤적 생성)
        trajectory, phases = self.traj.plan_strike(
            T_current=T_current,
            ball_pos=ball_pos,
            strike_direction=strike_dir_3d,
            strike_speed=strike_speed,
            approach_dist=0.06,
            follow_dist=0.05,      # 공 너머를 목표로 하여 감속 없이 타격
            strike_height=strike_height,
            tool_offset=self.tool_offset
        )

        print(f"  Trajectory: {len(trajectory)} points")
        print(f"    Strike dir 3D: [{strike_dir_3d[0]:.3f}, {strike_dir_3d[1]:.3f}, {strike_dir_3d[2]:.3f}]")
        print(f"    EE strike speed: {strike_speed:.3f} m/s")
        print(f"    Approach: {phases['approach'][1] - phases['approach'][0]} pts")
        print(f"    Strike:   {phases['strike'][1] - phases['strike'][0]} pts")

        self._strike_skipped = False

        # 실행 — 타격 후 즉시 후퇴
        self.controller.execute_trajectory(
            trajectory, dt=0.002, phase_indices=phases,
            strike_speed=strike_speed
        )

    def _observe(self):
        """OBSERVE: 결과 관찰"""
        # 임팩트 직후 공 속도 측정 (계획 vs 실제 비교용)
        if self.demo_type == 'minigolf':
            ball_vel = self.env.get_ball_velocity()
            ball_speed = np.linalg.norm(ball_vel[:2])
            print(f"  Ball velocity after impact: {ball_speed:.3f} m/s "
                  f"[{ball_vel[0]:.3f}, {ball_vel[1]:.3f}, {ball_vel[2]:.3f}]")

        # 공이 멈출 때까지 대기
        if self.demo_type == 'minigolf':
            self.env.wait_ball_stop(timeout=8.0)
            return self.env.is_hole_in()
        else:  # billiards
            self.env.wait_balls_stop(timeout=8.0)
            # 공이 테이블 밖으로 이탈했으면 리셋
            if self.env.is_ball_out_of_table(self.env.cue_ball_id):
                print(f"  [WARNING] Cue ball fell off table! Resetting to start position.")
                self.env.reset_balls(cue_pos=self.env.cue_start_pos)
                import time
                time.sleep(0.5)
                return False
            return self.env.is_pocketed()

    def _get_result_distance(self):
        """결과 거리 측정"""
        if self.demo_type == 'minigolf':
            return self.env.get_distance_to_hole()
        else:
            target_pos = self.env.get_target_ball_position()
            nearest, dist = self.env.get_nearest_pocket(target_pos)
            return dist
