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

from project.config import *


class AutonomousStateMachine:
    """자율 타격 루프 상태 머신

    States:
        SCAN: 공/홀/포켓 위치 인식
        THINK: 최적 타격 벡터 계산
        STRIKE: 타격 궤적 생성 및 실행
        OBSERVE: 결과 관찰 (성공/실패 판정)
    """

    def __init__(self, controller, environment, shot_planner, traj_planner,
                 demo_type='minigolf', tool_offset=0.0, perception=None):
        """
        Args:
            controller: RobotController 인스턴스
            environment: MiniGolfEnvironment / BilliardsEnvironment / MazeEnvironment
            shot_planner: ShotPlanner 인스턴스
            traj_planner: StrikeTrajectoryPlanner 인스턴스
            demo_type: 'minigolf', 'billiards', or 'maze'
            tool_offset: EE에서 도구 끝까지 거리 (m)
            perception: PerceptionInterface (None이면 직접 env 접근)
        """
        self.controller = controller
        self.env = environment
        self.planner = shot_planner
        self.traj = traj_planner
        self.demo_type = demo_type
        self.tool_offset = tool_offset
        self.perception = perception
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
            self.last_scan = scan_data  # 궤적 장애물 체크용
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
        """SCAN: 공/홀/포켓/장애물 위치 인식"""
        if self.demo_type == 'maze' and self.perception is not None:
            return self.perception.scan_environment()

        if self.demo_type == 'minigolf':
            ball_pos = self.env.get_ball_position()
            return {
                'ball_pos': ball_pos,
                'hole_pos': self.env.hole_pos
            }
        elif self.demo_type == 'maze':
            cue_pos = self.env.get_cue_ball_position()
            target_pos = self.env.get_target_ball_position()
            obstacles = self.env.get_obstacle_positions()
            return {
                'cue_pos': cue_pos,
                'target_pos': target_pos,
                'obstacles': obstacles,
                'table_bounds': self.env.table_bounds
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
        elif self.demo_type == 'maze':
            print(f"  Running 3-ball cushion search...")
            candidates = self.planner.plan_shot(
                scan_data['cue_pos'],
                scan_data['target_pos'],
                scan_data['obstacles'],
                ball2_pos=scan_data.get('ball2_pos')
            )
            best = candidates[0]
            print(f"  Found {len(candidates)} diverse candidates")
            print(f"  Top: angle={best['angle_deg']:.1f}deg, "
                  f"cushions={best['cushion_count']}, "
                  f"hit_t1={best.get('hit_t1',False)}, "
                  f"hit_t2={best.get('hit_t2',False)}, "
                  f"score={best['score']:.0f}")
            return {
                'strike_dir': best['strike_dir'],
                'strike_speed': best['strike_speed'],
                'ball_pos': scan_data['cue_pos'],
                'ball_path': best.get('ball_path'),
                'candidates': candidates,  # 전체 후보 리스트
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
        """ALIGN & STRIKE: 후보 순회 → IK+장애물 검증 → 첫 번째 유효 후보만 실행

        핵심: 계획기가 여러 후보를 리턴하면, 각 후보에 대해
        모든 φ를 시도하여 IK+장애물 전부 통과하는 조합을 찾음.
        전부 실패하면 skip.
        """
        T_current = self.controller.get_current_T()
        ball_pos = plan['ball_pos']

        # 도달 가능성 확인
        ball_dist_from_base = np.linalg.norm(ball_pos[:2])
        if ball_dist_from_base > 0.80:
            print(f"  [WARNING] Ball too far from robot ({ball_dist_from_base:.3f}m > 0.80m).")
            if hasattr(self.env, 'reset_balls'):
                print(f"  Resetting cue ball to start position...")
                self.env.reset_balls(cue_pos=self.env.cue_start_pos)
                time.sleep(0.5)
                ball_pos = self.env.get_cue_ball_position()
                plan['ball_pos'] = ball_pos
            else:
                print(f"  Skipping strike.")
                self._strike_skipped = True
                return

        strike_height = ball_pos[2]

        # 장애물 좌표
        obs_list = []
        if hasattr(self, 'last_scan') and self.last_scan is not None:
            obs_list = self.last_scan.get('obstacles', [])

        # 후보 리스트 (maze는 다중 후보, 그 외는 단일)
        candidates = plan.get('candidates', [plan])

        q_current = self.controller.get_current_q()
        phi_candidates = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        # === 후보 순회: 각 후보 × 각 φ 조합 시도 ===
        found = False
        chosen_candidate = None

        for ci, candidate in enumerate(candidates):
            strike_dir_2d = candidate['strike_dir']
            strike_speed = candidate['strike_speed']

            # 2D → 3D 방향 변환
            if self.demo_type in ('billiards', 'maze'):
                angle_deg = BILLIARD_STRIKE_ANGLE_DEG if self.demo_type == 'billiards' else MAZE_STRIKE_ANGLE_DEG
                angle_rad = np.radians(angle_deg)
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
                strike_dir_3d = np.array(strike_dir_2d).flatten()

            best_result = None
            best_phi = 0.0
            best_min_w = -1
            best_trajectory = None
            best_phases = None

            for phi in phi_candidates:
                trajectory, phases = self.traj.plan_strike(
                    T_current=T_current,
                    ball_pos=ball_pos,
                    strike_direction=strike_dir_3d,
                    strike_speed=strike_speed,
                    approach_dist=STRIKE_APPROACH_DIST,
                    follow_dist=STRIKE_FOLLOW_DIST,
                    strike_height=strike_height,
                    tool_offset=self.tool_offset,
                    tool_rotation=phi
                )

                # 장애물 근접 체크
                obstacle_clear = True
                if obs_list:
                    clearance = 0.07  # 장애물 r(1.5cm) + EE(3cm) + 여유(2.5cm)
                    check_start = max(phases['approach'][1] - 200, 0)
                    check_end = phases['strike'][1]
                    for k in range(check_start, min(check_end, len(trajectory)), 10):
                        ee_pos = trajectory[k][0:3, 3]
                        for ox, oy, orr in obs_list:
                            dx = ee_pos[0] - ox
                            dy = ee_pos[1] - oy
                            dist_xy = np.sqrt(dx*dx + dy*dy)
                            if dist_xy < orr + clearance:
                                obstacle_clear = False
                                break
                        if not obstacle_clear:
                            break

                if not obstacle_clear:
                    continue

                # IK 사전검증
                result = self.controller.ik.solve_trajectory_validated(
                    q_current, trajectory
                )

                if result['valid']:
                    if result['min_manipulability'] > best_min_w:
                        best_min_w = result['min_manipulability']
                        best_result = result
                        best_phi = phi
                        best_trajectory = trajectory
                        best_phases = phases

            # 이 후보에서 유효한 궤적을 찾았으면 사용
            if best_result is not None and best_result['valid']:
                print(f"  [OK] Candidate #{ci+1}/{len(candidates)} valid "
                      f"(angle={candidate['angle_deg']:.1f}deg, phi={np.degrees(best_phi):.0f}deg, "
                      f"w={best_min_w:.4f}, "
                      f"hit_t1={candidate.get('hit_t1',False)}, "
                      f"hit_t2={candidate.get('hit_t2',False)}, "
                      f"score={candidate['score']:.0f})")
                found = True
                chosen_candidate = candidate
                trajectory = best_trajectory
                phases = best_phases
                break
            else:
                reason = "IK invalid" if best_result else "obstacle collision"
                print(f"  [SKIP] Candidate #{ci+1} (angle={candidate['angle_deg']:.1f}deg): {reason}")

        if not found:
            print(f"  [FAIL] All {len(candidates)} candidates failed IK+obstacle check. Skipping strike.")
            self._strike_skipped = True
            return

        # 시각화 (선택된 후보의 3공 궤적)
        if hasattr(self.env, 'client'):
            import pybullet as _p
            surface_z = ball_pos[2]

            # 큐볼 경로 (파란색)
            ball_path = chosen_candidate.get('ball_path')
            if ball_path is not None and len(ball_path) > 1:
                for i in range(len(ball_path) - 1):
                    d = np.linalg.norm(np.array(ball_path[i]) - np.array(ball_path[i+1]))
                    if d > 0.01:
                        p1 = [ball_path[i][0], ball_path[i][1], surface_z]
                        p2 = [ball_path[i+1][0], ball_path[i+1][1], surface_z]
                        _p.addUserDebugLine(p1, p2, [0, 0.5, 1], lineWidth=3,
                                           lifeTime=30, physicsClientId=self.env.client)

            # 목표공1 경로 (노란색)
            tgt1_path = chosen_candidate.get('tgt1_path')
            if tgt1_path is not None and len(tgt1_path) > 1:
                for i in range(len(tgt1_path) - 1):
                    d = np.linalg.norm(np.array(tgt1_path[i]) - np.array(tgt1_path[i+1]))
                    if d > 0.01:
                        p1 = [tgt1_path[i][0], tgt1_path[i][1], surface_z]
                        p2 = [tgt1_path[i+1][0], tgt1_path[i+1][1], surface_z]
                        _p.addUserDebugLine(p1, p2, [1, 0.9, 0], lineWidth=2,
                                           lifeTime=30, physicsClientId=self.env.client)

            # 목표공2 경로 (빨간색)
            tgt2_path = chosen_candidate.get('tgt2_path')
            if tgt2_path is not None and len(tgt2_path) > 1:
                for i in range(len(tgt2_path) - 1):
                    d = np.linalg.norm(np.array(tgt2_path[i]) - np.array(tgt2_path[i+1]))
                    if d > 0.01:
                        p1 = [tgt2_path[i][0], tgt2_path[i][1], surface_z]
                        p2 = [tgt2_path[i+1][0], tgt2_path[i+1][1], surface_z]
                        _p.addUserDebugLine(p1, p2, [1, 0.2, 0.2], lineWidth=2,
                                           lifeTime=30, physicsClientId=self.env.client)

            n_pts = len(ball_path) if ball_path else 0
            print(f"    [VIS] 3-ball planned paths drawn (cue:{n_pts}pts)")

        print(f"  Trajectory: {len(trajectory)} points")
        print(f"    EE strike speed: {strike_speed:.3f} m/s")

        # 실행 전: 도구-큐볼 충돌 재활성화
        if hasattr(self.controller, '_reenable_tool_cue_collision'):
            self.controller._reenable_tool_cue_collision()

        # 접촉 추적 리셋
        if hasattr(self.env, 'reset_contact_tracking'):
            self.env.reset_contact_tracking()

        # 실행 (도구가 물리적으로 공을 타격)
        result = self.controller.execute_trajectory(
            trajectory, dt=0.002, phase_indices=phases,
            strike_speed=strike_speed,
            q_trajectory=best_result['q_trajectory']
        )
        if result is False:
            print(f"  Strike aborted (Ready err too large)")
            self._strike_skipped = True

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
        elif self.demo_type == 'maze':
            self.env.wait_balls_stop(timeout=8.0)
            # 공이 테이블 밖으로 나갔으면 리셋
            if hasattr(self.env, 'is_ball_out_of_table'):
                cue_out = self.env.is_ball_out_of_table(self.env.cue_ball_id)
                tgt_out = self.env.is_ball_out_of_table(self.env.target_ball_id)
                b2_out = self.env.is_ball_out_of_table(self.env.ball2_id) if hasattr(self.env, 'ball2_id') else False
                if cue_out or tgt_out or b2_out:
                    print(f"  [WARNING] Ball off table! cue={cue_out}, tgt1={tgt_out}, tgt2={b2_out}")
                    reset_cue = self.env.cue_start_pos if cue_out else None
                    reset_tgt = self.env.target_start_pos if tgt_out else None
                    self.env.reset_balls(cue_pos=reset_cue, target_pos=reset_tgt)
                    # ball2 리셋 (pybullet 직접 호출)
                    if b2_out and hasattr(self.env, 'ball2_start_pos'):
                        import pybullet
                        pybullet.resetBasePositionAndOrientation(
                            self.env.ball2_id, list(self.env.ball2_start_pos),
                            [0,0,0,1], physicsClientId=self.env.client)
                        pybullet.resetBaseVelocity(self.env.ball2_id, [0,0,0], [0,0,0],
                                                   physicsClientId=self.env.client)
                    import time as _t; _t.sleep(0.5)
                    return False
            hit = self.env.is_target_hit()
            events = getattr(self.env, '_contact_events', [])
            cushion_count = getattr(self.env, '_contact_cushion_count', 0)
            cue = self.env.get_cue_ball_position()
            tgt1 = self.env.get_target_ball_position()
            tgt2 = self.env.get_ball2_position() if hasattr(self.env, 'get_ball2_position') else cue
            cue_moved = np.linalg.norm(cue[:2] - self.env.cue_start_pos[:2])
            tgt1_moved = np.linalg.norm(tgt1[:2] - self.env.target_start_pos[:2])
            tgt2_moved = np.linalg.norm(tgt2[:2] - self.env.ball2_start_pos[:2]) if hasattr(self.env, 'ball2_start_pos') else 0

            # 3쿠션 순서 검증
            valid_3cushion = False
            hit_t1 = getattr(self.env, '_contact_hit_t1', False)
            hit_t2 = getattr(self.env, '_contact_hit_t2', False)
            if events and hit_t1 and hit_t2:
                t1_idx = events.index('t1') if 't1' in events else -1
                t2_idx = events.index('t2') if 't2' in events else -1
                if t1_idx >= 0 and t2_idx >= 0:
                    if t1_idx < t2_idx:
                        c_between = events[t1_idx+1:t2_idx].count('c')
                        c_before = events[:t1_idx].count('c')
                        if c_between >= 3 or c_before >= 3:
                            valid_3cushion = True
                    else:
                        c_before = events[:t2_idx].count('c')
                        c_between = events[t2_idx+1:t1_idx].count('c')
                        if c_between >= 3 or c_before >= 3:
                            valid_3cushion = True

            print(f"  3-cushion: events={events}, cushions={cushion_count}")
            print(f"  Displacements: cue={cue_moved*100:.1f}cm, tgt1={tgt1_moved*100:.1f}cm, tgt2={tgt2_moved*100:.1f}cm")
            if valid_3cushion:
                print(f"  [OK] Valid 3-cushion sequence!")
            elif hit_t1 and hit_t2:
                print(f"  [FAIL] Both hit but NOT valid 3-cushion")
            return valid_3cushion
        else:  # billiards
            self.env.wait_balls_stop(timeout=8.0)
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
        elif self.demo_type == 'maze':
            cue = self.env.get_cue_ball_position()
            tgt = self.env.get_target_ball_position()
            return np.linalg.norm(cue[:2] - tgt[:2])
        else:
            target_pos = self.env.get_target_ball_position()
            nearest, dist = self.env.get_nearest_pocket(target_pos)
            return dist
