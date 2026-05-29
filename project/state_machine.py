"""
?먯쑉 猷⑦봽 State Machine
=========================
SCAN ??THINK ??ALIGN & STRIKE ??OBSERVE & RECALCULATE
"""
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from project.config import *


class AutonomousStateMachine:
    """?먯쑉 ?寃?猷⑦봽 ?곹깭 癒몄떊

    States:
        SCAN: 怨??/?ъ폆 ?꾩튂 ?몄떇
        THINK: 理쒖쟻 ?寃?踰≫꽣 怨꾩궛
        STRIKE: ?寃?沅ㅼ쟻 ?앹꽦 諛??ㅽ뻾
        OBSERVE: 寃곌낵 愿李?(?깃났/?ㅽ뙣 ?먯젙)
    """

    def __init__(self, controller, environment, shot_planner, traj_planner,
                 demo_type='minigolf', tool_offset=0.0, perception=None):
        """
        Args:
            controller: RobotController ?몄뒪?댁뒪
            environment: MiniGolfEnvironment / BilliardsEnvironment / MazeEnvironment
            shot_planner: ShotPlanner ?몄뒪?댁뒪
            traj_planner: StrikeTrajectoryPlanner ?몄뒪?댁뒪
            demo_type: 'minigolf', 'billiards', or 'maze'
            tool_offset: EE?먯꽌 ?꾧뎄 ?앷퉴吏 嫄곕━ (m)
            perception: PerceptionInterface (None?대㈃ 吏곸젒 env ?묎렐)
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
        self.last_chosen_candidate = None
        self.last_executed_scan = None
        self.last_executed_plan = None
        self.last_executed_trajectory = None
        self.last_executed_phases = None
        self.last_executed_q_trajectory = None
        self.last_planned_angle_deg = None
        self.last_planned_strike_dir_3d = None

    def run(self, max_attempts=5):
        """?먯쑉 猷⑦봽 ?ㅽ뻾

        Returns:
            success: 理쒖쥌 ?깃났 ?щ?
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
            self.last_scan = scan_data  # 沅ㅼ쟻 ?μ븷臾?泥댄겕??
            print(f"  Scan result: {scan_data}")

            # === THINK ===
            print(f"\n[STATE: THINK] Computing optimal strike...")
            plan = self._think(scan_data)
            print(f"  Strike direction: {plan['strike_dir']}")
            print(f"  Strike speed: {plan['strike_speed']:.3f} m/s")

            # === ALIGN & STRIKE ===
            print(f"\n[STATE: ALIGN & STRIKE] Executing...")
            self._strike_skipped = False
            self._strike_skip_reason = None
            self._strike(scan_data, plan)

            if getattr(self, '_strike_skipped', False):
                reason = getattr(self, '_strike_skip_reason', None) or "ball unreachable or no valid path"
                print(f"  Strike skipped ({reason})")
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
        """SCAN: 怨??/?ъ폆/?μ븷臾??꾩튂 ?몄떇"""
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
        """THINK: 理쒖쟻 ?寃?踰≫꽣 怨꾩궛"""
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
            if not candidates:
                print("  [FAIL] Planner returned no physically valid 3-cushion candidate.")
                self.last_chosen_candidate = None
                self.last_executed_scan = scan_data.copy() if isinstance(scan_data, dict) else scan_data
                self.last_executed_plan = None
                self.last_executed_trajectory = None
                self.last_executed_phases = None
                self.last_executed_q_trajectory = None
                self.last_planned_angle_deg = None
                return {
                    'strike_dir': np.array([1.0, 0.0]),
                    'strike_speed': 0.0,
                    'ball_pos': scan_data['cue_pos'],
                    'ball_path': None,
                    'candidates': [],
                }
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
                'candidates': candidates,  # ?꾩껜 ?꾨낫 由ъ뒪??
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
        """ALIGN & STRIKE: ?꾨낫 ?쒗쉶 ??IK+?μ븷臾?寃利???泥?踰덉㎏ ?좏슚 ?꾨낫留??ㅽ뻾

        ?듭떖: 怨꾪쉷湲곌? ?щ윭 ?꾨낫瑜?由ы꽩?섎㈃, 媛??꾨낫?????
        紐⑤뱺 ?瑜??쒕룄?섏뿬 IK+?μ븷臾??꾨? ?듦낵?섎뒗 議고빀??李얠쓬.
        ?꾨? ?ㅽ뙣?섎㈃ skip.
        """
        T_current = self.controller.get_current_T()
        ball_pos = plan['ball_pos']

        if self.demo_type == 'maze' and not plan.get('candidates'):
            print("  [FAIL] No valid 3-cushion candidate to execute.")
            self._strike_skipped = True
            self._strike_skip_reason = "no valid 3-cushion candidate"
            return

        # ?꾨떖 媛?μ꽦 ?뺤씤
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
                self._strike_skip_reason = "ball unreachable"
                return

        strike_height = ball_pos[2]

        # ?μ븷臾?醫뚰몴
        obs_list = []
        if hasattr(self, 'last_scan') and self.last_scan is not None:
            obs_list = self.last_scan.get('obstacles', [])

        # ?꾨낫 由ъ뒪??(maze???ㅼ쨷 ?꾨낫, 洹??몃뒗 ?⑥씪)
        candidates = plan.get('candidates', [plan])

        q_current = self.controller.get_current_q()
        # ㄴ자 도구는 비대칭 → phi 회전하면 도구 끝이 공에서 벗어남
        if self.demo_type == 'maze':
            phi_candidates = [0.0]
        else:
            phi_candidates = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        # === Collect IK-valid candidates ===
        MAX_VERIFY = 5
        ik_valid_list = []

        for ci, candidate in enumerate(candidates):
            if len(ik_valid_list) >= MAX_VERIFY:
                break
            strike_dir_2d = candidate['strike_dir']
            strike_speed = candidate['strike_speed']

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
                traj_c, ph_c = self.traj.plan_strike(
                    T_current=T_current,
                    ball_pos=ball_pos,
                    strike_direction=strike_dir_3d,
                    strike_speed=strike_speed,
                    approach_dist=candidate.get('safe_approach_dist', STRIKE_APPROACH_DIST),
                    follow_dist=STRIKE_FOLLOW_DIST,
                    strike_height=strike_height,
                    tool_offset=self.tool_offset,
                    tool_rotation=phi,
                    table_bounds=scan_data.get('table_bounds') if isinstance(scan_data, dict) else None
                )

                obstacle_clear = True
                if obs_list:
                    clearance = 0.07
                    check_start = max(ph_c['approach'][1] - 200, 0)
                    check_end = ph_c['strike'][1]
                    for k in range(check_start, min(check_end, len(traj_c)), 10):
                        ee_pos = traj_c[k][0:3, 3]
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

                approach_end = ph_c.get('approach', (0, 0))[1]
                val_from = int(approach_end * 0.65)
                ik_result = self.controller.ik.solve_trajectory_validated(
                    q_current, traj_c, validate_from=val_from
                )

                if ik_result['valid']:
                    if ik_result['min_manipulability'] > best_min_w:
                        best_min_w = ik_result['min_manipulability']
                        best_result = ik_result
                        best_phi = phi
                        best_trajectory = traj_c
                        best_phases = ph_c

            if best_result is not None and best_result['valid']:
                print(f"  [IK-OK] #{ci+1}/{len(candidates)} "
                      f"(angle={candidate['angle_deg']:.1f}, score={candidate['score']:.0f})")
                ik_valid_list.append((candidate, best_result, best_trajectory, best_phases, strike_dir_3d.copy()))
            else:
                reason = "IK invalid" if best_result else ("obstacle" if not obstacle_clear else "IK failed")
                print(f"  [SKIP] #{ci+1} (angle={candidate['angle_deg']:.1f}): {reason}")

        if not ik_valid_list:
            print(f"  [FAIL] All candidates failed IK. Skipping.")
            self._strike_skipped = True
            self._strike_skip_reason = "IK or obstacle validation failed"
            return

        # === saveState verification (maze only, multiple candidates) ===
        import pybullet as _p
        has_gui = hasattr(self.env, 'client')
        use_verify = (self.demo_type == 'maze' and has_gui and len(ik_valid_list) > 1)

        if use_verify:
            print(f"  [VERIFY] Testing {len(ik_valid_list)} candidates via saveState...")

        verified_idx = None
        for vi, (cand, ik_res, traj, ph, sd3d) in enumerate(ik_valid_list):
            is_last = (vi == len(ik_valid_list) - 1)

            if use_verify and not is_last:
                state_id = _p.saveState(physicsClientId=self.env.client)

                if hasattr(self.env, 'reset_contact_tracking'):
                    self.env.reset_contact_tracking()
                if hasattr(self.controller, '_reenable_tool_cue_collision'):
                    self.controller._reenable_tool_cue_collision()

                exec_ok = self.controller.execute_trajectory(
                    traj, dt=0.002, phase_indices=ph,
                    strike_speed=cand['strike_speed'],
                    q_trajectory=ik_res['q_trajectory']
                )

                if exec_ok is False:
                    print(f"    [V#{vi+1}] Exec aborted")
                    _p.restoreState(stateId=state_id, physicsClientId=self.env.client)
                    _p.removeState(state_id, physicsClientId=self.env.client)
                    continue

                self.env.wait_balls_stop(timeout=8.0)

                events = getattr(self.env, '_contact_events', [])
                hit_t1 = getattr(self.env, '_contact_hit_t1', False)
                hit_t2 = getattr(self.env, '_contact_hit_t2', False)
                c_total = sum(1 for e in events if e == 'c')
                valid_shot = hit_t1 and hit_t2 and c_total >= 2

                if valid_shot:
                    print(f"    [V#{vi+1}] OK angle={cand['angle_deg']:.1f} events={events}")
                    _p.removeState(state_id, physicsClientId=self.env.client)
                    verified_idx = vi
                    break
                else:
                    print(f"    [V#{vi+1}] MISS angle={cand['angle_deg']:.1f} events={events}")
                    _p.restoreState(stateId=state_id, physicsClientId=self.env.client)
                    _p.removeState(state_id, physicsClientId=self.env.client)
                    self.controller.move_home()
                    time.sleep(0.3)
            else:
                verified_idx = vi
                break

        chosen_idx = verified_idx if verified_idx is not None else 0
        chosen_candidate, best_result, trajectory, phases, chosen_strike_dir_3d = ik_valid_list[chosen_idx]
        strike_speed = chosen_candidate['strike_speed']
        found = True

        print(f"  [SELECTED] #{chosen_idx+1}/{len(ik_valid_list)} "
              f"angle={chosen_candidate['angle_deg']:.1f}, score={chosen_candidate['score']:.0f}")

        self.last_chosen_candidate = chosen_candidate
        self.last_executed_scan = scan_data.copy() if isinstance(scan_data, dict) else scan_data
        self.last_executed_plan = plan.copy() if isinstance(plan, dict) else plan
        self.last_executed_trajectory = trajectory
        self.last_executed_phases = phases
        self.last_executed_q_trajectory = best_result['q_trajectory']
        self.last_planned_strike_dir_3d = chosen_strike_dir_3d
        self.last_planned_angle_deg = chosen_candidate.get('angle_deg')
        print(f"    [PLAN] valid_3cushion={chosen_candidate.get('valid_3cushion')}, "
              f"events={chosen_candidate.get('events', [])}, "
              f"planned_ball_angle={chosen_candidate.get('initial_ball_angle_deg')}")

        if has_gui:
            surface_z = ball_pos[2]
            for path_data, color, width, min_d in [
                (chosen_candidate.get('ball_path'), [0, 0.5, 1], 3, 0.003),
                (chosen_candidate.get('tgt1_path'), [1, 0.9, 0], 2, 0.003),
                (chosen_candidate.get('tgt2_path'), [1, 0.2, 0.2], 2, 0.01),
            ]:
                if path_data and len(path_data) > 1:
                    for i in range(len(path_data) - 1):
                        d = np.linalg.norm(np.array(path_data[i]) - np.array(path_data[i+1]))
                        if d > min_d:
                            p1 = [path_data[i][0], path_data[i][1], surface_z]
                            p2 = [path_data[i+1][0], path_data[i+1][1], surface_z]
                            _p.addUserDebugLine(p1, p2, color, lineWidth=width,
                                               lifeTime=30, physicsClientId=self.env.client)
            print(f"    [VIS] paths drawn")

        print(f"  Trajectory: {len(trajectory)} points")
        print(f"    EE strike speed: {strike_speed:.3f} m/s")

        # Already executed via saveState verification - skip re-execution
        already_executed = (use_verify and verified_idx is not None
                            and verified_idx < len(ik_valid_list) - 1)
        if already_executed:
            self._verified_success = True
            print(f"    [VERIFY] Already executed, skip re-execution")
            return

        if hasattr(self.controller, '_reenable_tool_cue_collision'):
            self.controller._reenable_tool_cue_collision()
        if hasattr(self.env, 'reset_contact_tracking'):
            self.env.reset_contact_tracking()

        result = self.controller.execute_trajectory(
            trajectory, dt=0.002, phase_indices=phases,
            strike_speed=strike_speed,
            q_trajectory=best_result['q_trajectory']
        )
        if result is False:
            print(f"  Strike aborted (Ready err too large)")
            self._strike_skipped = True
            self._strike_skip_reason = "trajectory execution aborted"


    def _observe(self):
        """Observe the shot result."""

        # ?꾪뙥??吏곹썑 怨??띾룄 痢≪젙 (怨꾪쉷 vs ?ㅼ젣 鍮꾧탳??
        if self.demo_type == 'minigolf':
            ball_vel = self.env.get_ball_velocity()
            ball_speed = np.linalg.norm(ball_vel[:2])
            print(f"  Ball velocity after impact: {ball_speed:.3f} m/s "
                  f"[{ball_vel[0]:.3f}, {ball_vel[1]:.3f}, {ball_vel[2]:.3f}]")

        # 怨듭씠 硫덉텧 ?뚭퉴吏 ?湲?
        if self.demo_type == 'minigolf':
            self.env.wait_ball_stop(timeout=8.0)
            return self.env.is_hole_in()
        elif self.demo_type == 'maze':
            self.env.wait_balls_stop(timeout=8.0)
            # 怨듭씠 ?뚯씠釉?諛뽰쑝濡??섍컮?쇰㈃ 由ъ뀑
            if hasattr(self.env, 'is_ball_out_of_table'):
                cue_out = self.env.is_ball_out_of_table(self.env.cue_ball_id)
                tgt_out = self.env.is_ball_out_of_table(self.env.target_ball_id)
                b2_out = self.env.is_ball_out_of_table(self.env.ball2_id) if hasattr(self.env, 'ball2_id') else False
                if cue_out or tgt_out or b2_out:
                    print(f"  [WARNING] Ball off table! cue={cue_out}, tgt1={tgt_out}, tgt2={b2_out}")
                    reset_cue = self.env.cue_start_pos if cue_out else None
                    reset_tgt = self.env.target_start_pos if tgt_out else None
                    self.env.reset_balls(cue_pos=reset_cue, target_pos=reset_tgt)
                    # ball2 由ъ뀑 (pybullet 吏곸젒 ?몄텧)
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

            # 3荑좎뀡 ?쒖꽌 寃利?
            valid_3cushion = False
            valid_2cushion = False
            rule_case = 'missing-target-contact'
            cushions_before_first = 0
            cushions_between_targets = 0
            hit_t1 = getattr(self.env, '_contact_hit_t1', False)
            hit_t2 = getattr(self.env, '_contact_hit_t2', False)
            if events and hit_t1 and hit_t2:
                t1_idx = events.index('t1') if 't1' in events else -1
                t2_idx = events.index('t2') if 't2' in events else -1
                if t1_idx >= 0 and t2_idx >= 0:
                    second_idx = max(t1_idx, t2_idx)
                    first_idx = min(t1_idx, t2_idx)
                    c_total = sum(1 for e in events if e == 'c')  # 전체 쿠션 수
                    cushions_before_first = events[:first_idx].count('c')
                    cushions_between_targets = events[first_idx:second_idx].count('c')
                    if c_total >= 3:
                        valid_3cushion = True
                        rule_case = 'valid-3cushion'
                    elif c_total >= 2:
                        valid_2cushion = True
                        rule_case = 'valid-2cushion'
                    else:
                        rule_case = 'both-hit-not-enough-cushions'

            print(f"  Cushion result: events={events}, cushions={cushion_count}")
            print(f"  Detail: hit_t1={hit_t1}, hit_t2={hit_t2}, "
                  f"before_first={cushions_before_first}, "
                  f"between_targets={cushions_between_targets}, rule={rule_case}")
            print(f"  Displacements: cue={cue_moved*100:.1f}cm, tgt1={tgt1_moved*100:.1f}cm, tgt2={tgt2_moved*100:.1f}cm")
            success = valid_3cushion or valid_2cushion
            if valid_3cushion:
                print(f"  [OK] Valid 3-cushion sequence!")
            elif valid_2cushion:
                print(f"  [OK] Valid 2-cushion sequence!")
            elif hit_t1 and hit_t2:
                print(f"  [FAIL] Both hit but NOT valid 2/3-cushion")
            actual_speed = getattr(self.env, '_last_actual_ball_speed', None)
            actual_angle = getattr(self.env, '_last_actual_ball_angle_deg', None)
            if actual_speed is not None:
                angle_text = "n/a" if actual_angle is None else f"{actual_angle:.1f}deg"
                diff_text = "n/a"
                if actual_angle is not None and self.last_planned_angle_deg is not None:
                    diff = abs((actual_angle - self.last_planned_angle_deg + 180) % 360 - 180)
                    diff_text = f"{diff:.1f}deg"
                print(f"  Actual initial cue ball: speed={actual_speed:.3f}m/s, "
                      f"angle={angle_text}, planned={self.last_planned_angle_deg}, diff={diff_text}")
            return success
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
        """Measure result distance."""
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
