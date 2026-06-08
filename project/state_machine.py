#수정 전 state_machin.py (06061824 기준)

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
                 demo_type='pocket_phase1', tool_offset=0.0, perception=None):
        """
        Args:
            controller: RobotController ?몄뒪?댁뒪
            environment: MiniGolfEnvironment / BilliardsEnvironment / MazeEnvironment
            shot_planner: ShotPlanner ?몄뒪?댁뒪
            traj_planner: StrikeTrajectoryPlanner ?몄뒪?댁뒪
            demo_type: 'pocket_phase1' or 'pocket_phase2'
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

        if self.demo_type == 'pocket_phase1':
            return self._run_phase1_pocket(max_attempts)
        if self.demo_type == 'pocket_phase2':
            return self._run_phase2_trickshot(max_attempts)

        raise ValueError(
            f"Unsupported demo_type '{self.demo_type}'. "
            "Only 'pocket_phase1' and 'pocket_phase2' are active.")

    def _think_pocket(self, scan_data, phase, target_ball_id, marker_pos=None):
        """pocket_demo용 THINK: 포켓/정밀정지 탐색."""
        cue_pos = scan_data['cue_pos']
        target_pos = scan_data['target_pos']
        other_balls = scan_data.get('other_balls', [])

        # Escape shot: cue ball is too close to a wall.
        # In this case, prioritize moving the cue ball back toward table center
        # instead of searching a normal pocket shot.
        if phase == 'pocket':
            escape_plan = self._make_escape_plan(cue_pos, scan_data=scan_data)
            if escape_plan is not None:
                print("  [THINK] Cue ball is near wall -> escape shot fallback")
                return escape_plan

        if phase == 'pocket':
            print(f"  Running pocket shot search...")
            candidates = self.planner.plan_pocket_shot(
                cue_pos, target_pos, other_balls)
        else:  # precision
            print(f"  Running precision shot search (marker={marker_pos})...")
            candidates = self.planner.plan_precision_shot(
                cue_pos, target_pos, marker_pos, other_balls)

        if not candidates:
            print("  [FAIL] No valid candidate found.")
            return {
                'strike_dir': np.array([1.0, 0.0]),
                'strike_speed': 0.0,
                'ball_pos': cue_pos,
                'candidates': [],
            }

        best = candidates[0]
        print(f"  Found {len(candidates)} candidates")
        print(f"  Top: angle={best['angle_deg']:.1f}deg, score={best['score']:.0f}")
        return {
            'strike_dir': best['strike_dir'],
            'strike_speed': best['strike_speed'],
            'ball_pos': cue_pos,
            'candidates': candidates,
        }

    def _cue_wall_escape_info(self, cue_pos):
        """큐볼이 벽에 1cm 이하로 붙었는지 확인하고 escape 방향을 계산한다.

        여기서 gap은 '공 표면과 벽 사이 거리'로 계산한다.
        즉, 공 중심-벽 거리에서 공 반지름을 뺀 값이다.
        """
        if not hasattr(self.env, 'table_bounds'):
            return False, None, {}

        cue = np.array(cue_pos).flatten()
        cue_xy = cue[:2]
        b = self.env.table_bounds

        threshold = globals().get('ESCAPE_WALL_GAP_THRESHOLD', 0.01)
        ball_r = globals().get('MAZE_BALL_RADIUS', 0.012)

        # 공 표면과 각 벽 사이 간격
        gaps = {
            'x_min': cue_xy[0] - b['x_min'] - ball_r,
            'x_max': b['x_max'] - cue_xy[0] - ball_r,
            'y_min': cue_xy[1] - b['y_min'] - ball_r,
            'y_max': b['y_max'] - cue_xy[1] - ball_r,
        }

        near_keys = [key for key, gap in gaps.items() if gap <= threshold]
        if not near_keys:
            return False, None, {
                'gaps': gaps,
                'near_keys': [],
                'min_gap': min(gaps.values()),
            }

        inward = np.zeros(2)
        if 'x_min' in near_keys:
            inward += np.array([1.0, 0.0])
        if 'x_max' in near_keys:
            inward += np.array([-1.0, 0.0])
        if 'y_min' in near_keys:
            inward += np.array([0.0, 1.0])
        if 'y_max' in near_keys:
            inward += np.array([0.0, -1.0])

        n = np.linalg.norm(inward)
        if n < 1e-9:
            return False, None, {
                'gaps': gaps,
                'near_keys': near_keys,
                'min_gap': min(gaps.values()),
            }

        inward = inward / n
        return True, inward, {
            'gaps': gaps,
            'near_keys': near_keys,
            'min_gap': min(gaps.values()),
        }

    def _make_escape_plan(self, cue_pos, scan_data=None):
        """벽에 붙은 큐볼을 테이블 중앙 방향으로 빼내는 escape plan 생성.

        포켓 성공용 plan이 아니라 reposition용 plan이다.
        _strike()는 candidate의 strike_height_offset을 읽어 평소보다 높은 위치에서 친다.
        """
        near_wall, inward_dir, wall_info = self._cue_wall_escape_info(cue_pos)
        if not near_wall:
            return None

        height_offset = globals().get('ESCAPE_STRIKE_HEIGHT_OFFSET', 0.016)
        escape_speed = globals().get('ESCAPE_BALL_SPEED', 0.45)
        safe_approach = globals().get('ESCAPE_SAFE_APPROACH_DIST', 0.035)
        follow_dist = globals().get('ESCAPE_FOLLOW_DIST', 0.025)

        base_angle = np.arctan2(inward_dir[1], inward_dir[0])
        angle_offsets_deg = [0.0, -8.0, 8.0, -15.0, 15.0]
        candidates = []

        print(
            f"  [ESCAPE] cue near wall: keys={wall_info['near_keys']}, "
            f"min_gap={wall_info['min_gap']*100:.1f}cm, "
            f"inward_dir=[{inward_dir[0]:.2f}, {inward_dir[1]:.2f}], "
            f"height_offset={height_offset*100:.1f}cm"
        )

        for rank, off_deg in enumerate(angle_offsets_deg):
            angle = base_angle + np.radians(off_deg)
            strike_dir = np.array([np.cos(angle), np.sin(angle)])
            angle_deg = np.degrees(angle) % 360
            score = 9000 - rank * 200 - abs(off_deg) * 20

            candidates.append({
                'strike_dir': strike_dir,
                'strike_speed': MAX_TOOL_SPEED,
                'ball_speed': escape_speed,
                'score': score,
                'angle_deg': angle_deg,
                'angle': angle,
                'safe_approach_dist': safe_approach,
                'follow_dist': follow_dist,
                'strike_height_offset': height_offset,

                # escape marker
                'is_escape_shot': True,
                'escape_wall_keys': wall_info['near_keys'],
                'escape_min_gap': wall_info['min_gap'],

                # pocket candidate 호환 필드
                'target_pocketed': False,
                'hit_target': False,
                'illegal_contact': False,
                'cue_scratched': False,
                'pocket_idx': -1,
                'cue_path': None,
                'target_path': None,
                'tgt1_path': None,
                'tgt2_path': None,
                'cushion_count': 0,
                'hit_t1': False,
                'hit_t2': False,
                'events': [],
                'alignment_quality': 0.0,
                'alignment_error_deg': 180.0,
                'center_quality': 0.0,
                'robust_count': 0,
            })

        candidates.sort(key=lambda c: -c['score'])
        return {
            'strike_dir': candidates[0]['strike_dir'],
            'strike_speed': candidates[0]['strike_speed'],
            'ball_pos': np.array(cue_pos),
            'candidates': candidates,
            'is_escape_plan': True,
        }

    def _strike(self, scan_data, plan):
        """ALIGN & STRIKE: ?꾨낫 ?쒗쉶 ??IK+?μ븷臾?寃利???泥?踰덉㎏ ?좏슚 ?꾨낫留??ㅽ뻾

        ?듭떖: 怨꾪쉷湲곌? ?щ윭 ?꾨낫瑜?由ы꽩?섎㈃, 媛??꾨낫?????
        紐⑤뱺 ?瑜??쒕룄?섏뿬 IK+?μ븷臾??꾨? ?듦낵?섎뒗 議고빀??李얠쓬.
        ?꾨? ?ㅽ뙣?섎㈃ skip.
        """
        T_current = self.controller.get_current_T()
        ball_pos = plan['ball_pos']

        if not plan.get('candidates'):
            print("  [FAIL] No valid candidate to execute.")
            self._strike_skipped = True
            self._strike_skip_reason = "no valid candidate"
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
        phi_candidates = [0.0]

        # === Collect IK-valid candidates ===
        MAX_VERIFY = 5  # 후보 검증/IK 후보 최대 5개
        ik_valid_list = []

        for ci, candidate in enumerate(candidates):
            if len(ik_valid_list) >= MAX_VERIFY:
                break
            strike_dir_2d = candidate['strike_dir']
            strike_speed = candidate['strike_speed']

            angle_deg = MAZE_STRIKE_ANGLE_DEG
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

            best_result = None
            best_phi = 0.0
            best_min_w = -1
            best_trajectory = None
            best_phases = None

            for phi in phi_candidates:
                candidate_strike_height = strike_height + candidate.get('strike_height_offset', 0.0)
                candidate_follow_dist = candidate.get('follow_dist', STRIKE_FOLLOW_DIST)

                if candidate.get('is_escape_shot', False):
                    print(
                        f"  [ESCAPE] planning elevated strike: "
                        f"height={candidate_strike_height:.3f}m "
                        f"(+{candidate.get('strike_height_offset', 0.0)*100:.1f}cm), "
                        f"follow={candidate_follow_dist:.3f}m"
                    )

                traj_c, ph_c = self.traj.plan_strike(
                    T_current=T_current,
                    ball_pos=ball_pos,
                    strike_direction=strike_dir_3d,
                    strike_speed=strike_speed,
                    approach_dist=candidate.get('safe_approach_dist', STRIKE_APPROACH_DIST),
                    follow_dist=candidate_follow_dist,
                    strike_height=candidate_strike_height,
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

        # === saveState verification (maze + pocket phases, multiple candidates) ===
        import pybullet as _p
        has_gui = hasattr(self.env, 'client')
        has_escape_candidate = any(cand.get('is_escape_shot', False) for cand, *_ in ik_valid_list)
        use_verify = (self.demo_type == 'pocket_phase1'
                      and has_gui and len(ik_valid_list) > 1
                      and not has_escape_candidate)

        if use_verify:
            print(f"  [VERIFY] Testing {len(ik_valid_list)} candidates via saveState...")

        verified_idx = None
        for vi, (cand, ik_res, traj, ph, sd3d) in enumerate(ik_valid_list):
            is_last = (vi == len(ik_valid_list) - 1)

            if use_verify and not is_last:
                # saveState/restoreState는 GUI + constraint 환경에서 multibody 개수 mismatch로
                # physics server를 끊는 경우가 있어 사용하지 않는다.
                # 대신 후보 검증 후 공 위치를 수동 복원한다.
                state_id = None
                # 복원용 공 위치 저장
                _saved_cue = list(_p.getBasePositionAndOrientation(self.env.cue_ball_id, physicsClientId=self.env.client)[0])
                _saved_t1 = list(_p.getBasePositionAndOrientation(self.env.target_ball_id, physicsClientId=self.env.client)[0])
                _saved_t2 = list(_p.getBasePositionAndOrientation(self.env.ball2_id, physicsClientId=self.env.client)[0]) if hasattr(self.env, 'ball2_id') else None
                _saved_t3 = list(_p.getBasePositionAndOrientation(self.env.ball3_id, physicsClientId=self.env.client)[0]) if hasattr(self.env, 'ball3_id') and self.env.ball3_id is not None else None
                # _pocketed_balls 셋 저장
                _saved_pocketed = set(getattr(self.env, '_pocketed_balls', set()))

                if hasattr(self.env, 'reset_contact_tracking'):
                    self.env.reset_contact_tracking()
                if hasattr(self.controller, '_reenable_tool_cue_collision'):
                    self.controller._reenable_tool_cue_collision()

                exec_ok = self.controller.execute_trajectory(
                    traj, dt=0.002, phase_indices=ph,
                    strike_speed=cand['strike_speed'],
                    q_trajectory=ik_res['q_trajectory']
                )

                def _safe_restore():
                    """후보 검증 실패 시 수동 복원.

                    PyBullet restoreState는 GUI/constraint/multibody가 섞인 환경에서
                    실패하며 physics server를 끊을 수 있으므로 사용하지 않는다.
                    여기서는 공 위치/속도와 pocketed set만 복원한다.
                    """
                    for bid, pos in [
                        (self.env.cue_ball_id, _saved_cue),
                        (self.env.target_ball_id, _saved_t1),
                    ]:
                        _p.resetBasePositionAndOrientation(
                            bid, pos, [0, 0, 0, 1], physicsClientId=self.env.client)
                        _p.resetBaseVelocity(
                            bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.env.client)

                    if _saved_t2 is not None and hasattr(self.env, 'ball2_id'):
                        _p.resetBasePositionAndOrientation(
                            self.env.ball2_id, _saved_t2, [0, 0, 0, 1],
                            physicsClientId=self.env.client)
                        _p.resetBaseVelocity(
                            self.env.ball2_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.env.client)

                    if _saved_t3 is not None and hasattr(self.env, 'ball3_id'):
                        _p.resetBasePositionAndOrientation(
                            self.env.ball3_id, _saved_t3, [0, 0, 0, 1],
                            physicsClientId=self.env.client)
                        _p.resetBaseVelocity(
                            self.env.ball3_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.env.client)

                    if hasattr(self.env, '_pocketed_balls'):
                        self.env._pocketed_balls = set(_saved_pocketed)


                if exec_ok is False:
                    print(f"    [V#{vi+1}] Exec aborted")
                    _safe_restore()
                    continue

                self.env.wait_balls_stop(timeout=8.0)

                # 데모 타입에 따라 검증 기준 분리
                if self.demo_type == 'pocket_phase1':
                    # Phase 1: 목적구가 포켓에 들어갔는지 + 큐볼 스크래치 체크
                    target_id = scan_data.get('_target_ball_id')
                    target_in = target_id and self.env.is_ball_pocketed(target_id)
                    cue_scratched = self.env.is_ball_pocketed(self.env.cue_ball_id)
                    if target_in and not cue_scratched:
                        valid_shot = True
                    elif target_in and cue_scratched:
                        valid_shot = True
                        print(f"    [V#{vi+1}] SCRATCH — target pocketed but cue also scratched")
                    else:
                        valid_shot = False
                elif self.demo_type == 'pocket_phase2':
                    # Phase 2: 목적구가 마커 근처에 정지했는지 체크
                    target_id = scan_data.get('_target_ball_id')
                    marker_pos = scan_data.get('_marker_pos')
                    cue_scratched = self.env.is_ball_pocketed(self.env.cue_ball_id)
                    if target_id and marker_pos is not None:
                        t_pos, _ = _p.getBasePositionAndOrientation(
                            target_id, physicsClientId=self.env.client)
                        dist = np.linalg.norm(
                            np.array(t_pos[:2]) - np.array(marker_pos[:2]))
                        if dist <= PRECISION_STOP_TOLERANCE and not cue_scratched:
                            valid_shot = True
                            print(f"    [V#{vi+1}] OK dist={dist*100:.1f}cm")
                        elif cue_scratched:
                            valid_shot = False
                            print(f"    [V#{vi+1}] SCRATCH dist={dist*100:.1f}cm")
                        else:
                            valid_shot = False
                    else:
                        valid_shot = False
                else:
                    events = getattr(self.env, '_contact_events', [])
                    from project.physics.cushion_rules import valid_cushion_sequence
                    valid_shot = valid_cushion_sequence(events, 2)

                if valid_shot:
                    print(f"    [V#{vi+1}] OK angle={cand['angle_deg']:.1f}")
                    try:
                        _p.removeState(state_id, physicsClientId=self.env.client)
                    except Exception:
                        pass
                    verified_idx = vi
                    break
                else:
                    print(f"    [V#{vi+1}] MISS angle={cand['angle_deg']:.1f}")
                    _safe_restore()
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


    def _run_phase1_pocket(self, max_attempts_per_ball=4):
        """Phase 1: 목적구 3개를 포켓에 넣기.

        고정 순서 없음.
        매 shot마다 남은 공 전체를 평가한 뒤,
        가장 우선순위가 높은 candidate를 가진 공을 선택한다.
        """
        print("\n" + "=" * 60)
        print("  PHASE 1: POCKET SHOT (3 balls)")
        print("  Target strategy: evaluate all remaining balls, then choose best")
        print("=" * 60)

        ball_names = ['yellow', 'red', 'black']

        ball_id_getters = [
            lambda: self.env.target_ball_id,
            lambda: self.env.ball2_id,
            lambda: getattr(self.env, 'ball3_id', None),
        ]

        ball_pos_getters = [
            lambda: self.env.get_target_ball_position(),
            lambda: self.env.get_ball2_position(),
            lambda: self.env.get_ball3_position(),
        ]

        pocketed = [False, False, False]
        total_shots = 0
        max_total_shots = max_attempts_per_ball * 3

        def _candidate_priority(candidate):
            """Phase 1 target 선택용 우선순위.

            우선순위:
              1) target_pocketed=True
              2) illegal_contact=False
              3) cue_scratched=False
              4) score (점수가 높은 샷 우선 — 빗나간 샷이 직선성만 좋아서
                 선택되면 큐볼이 포켓으로 직행하는 자살골 발생)
              5) alignment_quality (동점일 때 직선에 가까운 후보 우선)
              6) robust_count
              7) center_quality
            """
            if candidate is None:
                return (-1, -1, -1, -float('inf'), -1.0, 0, 0.0)

            pocket_success = 1 if candidate.get('target_pocketed', False) else 0
            legal = 0 if candidate.get('illegal_contact', False) else 1
            no_scratch = 0 if candidate.get('cue_scratched', False) else 1
            alignment_quality = candidate.get('alignment_quality', 0.0) or 0.0
            score = candidate.get('score', -float('inf'))
            robust = candidate.get('robust_count', 0) or 0
            center_quality = candidate.get('center_quality', 0.0) or 0.0

            return (
                pocket_success,
                legal,
                no_scratch,
                score,
                alignment_quality,
                robust,
                center_quality,
            )

        while sum(pocketed) < 3 and total_shots < max_total_shots:
            total_shots += 1

            remaining = [
                i for i in range(3)
                if not pocketed[i] and ball_id_getters[i]() is not None
            ]

            if not remaining:
                break

            print(f"\n--- Shot {total_shots}/{max_total_shots} "
                  f"(remaining: {[ball_names[i] for i in remaining]}) ---")

            # ------------------------------------------------------------
            # THINK: 남은 모든 공에 대해 planner 실행 후 최고 우선순위 공 선택
            # ------------------------------------------------------------
            best_ball_idx = None
            best_plan = None
            best_scan = None
            best_candidate = None
            best_priority = (-1, -1, -1, -1.0, -float('inf'), 0, 0.0)

            for ball_idx in remaining:
                ball_id = ball_id_getters[ball_idx]()
                scan_data = self._scan_pocket(ball_idx, pocketed)

                # 이미 포켓된 공이면 pocketed 처리 후 skip
                if scan_data is None:
                    pocketed[ball_idx] = True
                    continue

                plan = self._think_pocket(scan_data, 'pocket', ball_id)
                candidate = plan['candidates'][0] if plan.get('candidates') else None

                top_score = candidate.get('score', -1) if candidate is not None else -1
                priority = _candidate_priority(candidate)

                print(
                    f"  {ball_names[ball_idx]}: "
                    f"top_score={top_score:.0f}, "
                    f"pocketed={candidate.get('target_pocketed', False) if candidate else False}, "
                    f"illegal={candidate.get('illegal_contact', False) if candidate else True}, "
                    f"scratch={candidate.get('cue_scratched', False) if candidate else False}, "
                    f"align_err={candidate.get('alignment_error_deg', 180.0) if candidate else 180.0:.1f}deg, "
                    f"align_q={candidate.get('alignment_quality', 0.0) if candidate else 0.0:.2f}, "
                    f"priority={priority}"
                )

                # 모든 공을 끝까지 평가해야 진짜 best target을 고를 수 있음.
                # 따라서 score >= 100000이어도 early break하지 않는다.
                if priority > best_priority:
                    best_priority = priority
                    best_ball_idx = ball_idx
                    best_plan = plan
                    best_scan = scan_data
                    best_candidate = candidate

            if best_ball_idx is None or best_plan is None or not best_plan.get('candidates'):
                print("  [FAIL] No viable shot for any remaining ball → escape")
                # 해가 전혀 없으면 큐볼을 테이블 중앙으로 밀어 리포지셔닝
                # 가장 마지막으로 스캔한 데이터에서 큐볼 위치 가져오기
                if best_scan is not None:
                    cue_pos = best_scan['cue_pos']
                else:
                    # scan도 없으면 skip
                    continue
                escape_plan = self._make_escape_plan(cue_pos, scan_data=best_scan)
                if escape_plan is None:
                    b = self.env.table_bounds
                    center = np.array([
                        (b['x_min'] + b['x_max']) / 2,
                        (b['y_min'] + b['y_max']) / 2
                    ])
                    cue2 = np.array(cue_pos[:2])
                    escape_dir = center - cue2
                    escape_norm = np.linalg.norm(escape_dir)
                    if escape_norm > 1e-6:
                        escape_dir = escape_dir / escape_norm
                    else:
                        escape_dir = np.array([0.0, 1.0])
                    escape_angle = np.arctan2(escape_dir[1], escape_dir[0])
                    escape_plan = {
                        'strike_dir': escape_dir,
                        'strike_speed': MAX_TOOL_SPEED,
                        'ball_pos': cue_pos,
                        'candidates': [{
                            'strike_dir': escape_dir,
                            'strike_speed': MAX_TOOL_SPEED,
                            'ball_speed': 0.3,
                            'score': 1,
                            'angle_deg': np.degrees(escape_angle),
                            'angle': escape_angle,
                            'safe_approach_dist': STRIKE_APPROACH_DIST,
                            'follow_dist': STRIKE_FOLLOW_DIST,
                            'is_escape_shot': True,
                            'target_pocketed': False,
                            'hit_target': False,
                            'illegal_contact': False,
                            'cue_scratched': False,
                            'pocket_idx': -1,
                            'cue_path': None,
                            'target_path': None,
                            'alignment_quality': 0.0,
                            'alignment_error_deg': 180.0,
                            'center_quality': 0.0,
                            'robust_count': 0,
                        }],
                    }
                    print(f"  [ESCAPE] No candidates → pushing cue toward center at {np.degrees(escape_angle):.1f}deg")
                best_plan = escape_plan
                best_ball_idx = remaining[0]
                ball_idx = best_ball_idx
                best_scan['_target_ball_id'] = ball_id_getters[ball_idx]()

            ball_idx = best_ball_idx
            ball_id = ball_id_getters[ball_idx]()
            chosen = best_candidate if best_candidate is not None else {}

            # === 자살골 방지: scratch 예측일 때만 escape shot ===
            chosen_score = chosen.get('score', -1)
            chosen_scratch = chosen.get('cue_scratched', False)
            if chosen_scratch:
                print(
                    f"\n  [SAFETY] Best candidate predicts cue scratch "
                    f"(score={chosen_score:.0f}). "
                    f"Attempting escape shot instead."
                )
                # escape plan 시도 (벽 근접 아니어도 테이블 중앙으로 밀기)
                cue_pos = best_scan['cue_pos']
                escape_plan = self._make_escape_plan(cue_pos, scan_data=best_scan)
                if escape_plan is None:
                    # 벽 근처가 아니면 수동 escape: 테이블 중앙 방향으로
                    b = self.env.table_bounds
                    center = np.array([
                        (b['x_min'] + b['x_max']) / 2,
                        (b['y_min'] + b['y_max']) / 2
                    ])
                    cue2 = np.array(cue_pos[:2])
                    escape_dir = center - cue2
                    escape_norm = np.linalg.norm(escape_dir)
                    if escape_norm > 1e-6:
                        escape_dir = escape_dir / escape_norm
                    else:
                        escape_dir = np.array([0.0, 1.0])
                    escape_angle = np.arctan2(escape_dir[1], escape_dir[0])
                    escape_plan = {
                        'strike_dir': escape_dir,
                        'strike_speed': MAX_TOOL_SPEED,
                        'ball_pos': cue_pos,
                        'candidates': [{
                            'strike_dir': escape_dir,
                            'strike_speed': MAX_TOOL_SPEED,
                            'ball_speed': 0.3,
                            'score': 1,
                            'angle_deg': np.degrees(escape_angle),
                            'angle': escape_angle,
                            'safe_approach_dist': STRIKE_APPROACH_DIST,
                            'follow_dist': STRIKE_FOLLOW_DIST,
                            'is_escape_shot': True,
                            'target_pocketed': False,
                            'illegal_contact': False,
                            'cue_scratched': False,
                        }],
                    }
                    print(f"  [SAFETY] Pushing cue toward center at {np.degrees(escape_angle):.1f}deg")
                best_plan = escape_plan
                best_scan['_target_ball_id'] = ball_id

            print(
                f"\n  [CHOSEN] {ball_names[ball_idx]} "
                f"score={chosen.get('score', -1):.0f}, "
                f"angle={chosen.get('angle_deg', float('nan')):.1f}deg, "
                f"pocketed={chosen.get('target_pocketed', False)}, "
                f"illegal={chosen.get('illegal_contact', False)}, "
                f"scratch={chosen.get('cue_scratched', False)}, "
                f"align_err={chosen.get('alignment_error_deg', 180.0):.1f}deg, "
                f"align_q={chosen.get('alignment_quality', 0.0):.2f}, "
                f"robust={chosen.get('robust_count', 0)}, "
                f"center_q={chosen.get('center_quality', 0.0):.2f}"
            )

            # ------------------------------------------------------------
            # STRIKE: 기존 _strike() 재활용 — IK + saveState 검증 포함
            # ------------------------------------------------------------
            self._strike_skipped = False
            self._strike(best_scan, best_plan)

            if getattr(self, '_strike_skipped', False):
                print(f"  Strike skipped: {getattr(self, '_strike_skip_reason', 'unknown')}")
                self.controller.move_home()
                time.sleep(0.5)
                continue

            last_cand = getattr(self, 'last_chosen_candidate', None)
            if last_cand is not None and last_cand.get('is_escape_shot', False):
                print("  [ESCAPE] Escape shot executed. Replanning next shot.")
                self.env.wait_balls_stop(timeout=5.0)
                self.controller.move_home()
                time.sleep(0.5)
                continue

            # ------------------------------------------------------------
            # OBSERVE
            # ------------------------------------------------------------
            self.env.wait_balls_stop(timeout=8.0)

            if self.env.is_ball_pocketed(ball_id):
                pocketed[ball_idx] = True
                pocket_idx = self.env.which_pocket(ball_id)
                print(f"  [SUCCESS] {ball_names[ball_idx]} pocketed! (pocket {pocket_idx})")

                # if self.env.is_ball_pocketed(self.env.cue_ball_id):
                #     print("  [WARNING] CUE BALL SCRATCHED! Resetting cue...")
                #     self.env.reset_balls(cue_pos=self.env.cue_start_pos)
                #     time.sleep(0.5)

            else:
                cue_scratched = self.env.is_ball_pocketed(self.env.cue_ball_id)

                # if cue_scratched:
                #     print("  [FAIL] Cue ball scratched! Resetting...")
                #     self.env.reset_balls(cue_pos=self.env.cue_start_pos)
                #     time.sleep(0.5)

                # else:
                target_final = ball_pos_getters[ball_idx]()
                if target_final is not None:
                    nearest_dist = min(
                        np.linalg.norm(target_final[:2] - pp[:2])
                        for pp in self.env.pocket_positions
                    )
                    print(f"  [MISS] Nearest pocket dist: {nearest_dist * 100:.1f}cm")

            self.controller.move_home()
            time.sleep(0.5)

        n_pocketed = sum(pocketed)
        print(f"\n{'=' * 40}")
        print(f"  PHASE 1 RESULT: {n_pocketed}/3 pocketed in {total_shots} shots")
        print(f"{'=' * 40}")

        return n_pocketed > 0

    def _run_phase2_trickshot(self, max_attempts=1):
        """Phase 2 POSTECH 트릭샷: 큐볼 한번으로 2공을 O 위치로 보내기."""
        import pybullet as _p

        print("\n  === POSTECH Trick Shot ===")
        print("  PC'S.TECH → POSTECH")

        # POSTECH O 배치
        meta = self.env.setup_postech_o()

        cue_pos = self.env.get_cue_ball_position()
        trick1_pos = np.array(meta['trick1_pos'])
        trick2_pos = np.array(meta['trick2_pos'])
        target1_goal = np.array(meta['target1_goal'])
        target2_goal = np.array(meta['target2_goal'])
        c_positions = [
            np.array(_p.getBasePositionAndOrientation(
                bid, physicsClientId=self.env.client)[0])
            for bid in meta['c_ball_ids']
        ]

        time.sleep(2.0)  # 배치 확인용 대기

        # 트릭샷 탐색
        print("\n  Running trick shot search...")
        candidates = self.planner.plan_trick_shot(
            cue_pos, trick1_pos, trick2_pos,
            target1_goal, target2_goal, c_positions
        )

        if not candidates:
            print("  [FAIL] No trick shot candidate found")
            return False

        # 최고 후보 표시
        top = candidates[0]
        print(f"\n  Best candidate:")
        print(f"    Angle: {top['angle_deg']:.1f}°")
        print(f"    Speed: {top['ball_speed']:.2f} m/s (tool: {top['strike_speed']:.2f} m/s)")
        print(f"    Expected dist: {top['total_dist']*100:.1f}cm")
        print(f"    Match: {top['match']}")

        # scan_data 구성 (기존 _strike와 호환)
        scan_data = {
            'cue_pos': cue_pos,
            'target_pos': trick1_pos,  # 첫 번째 trick ball
            'other_positions': [trick2_pos] + c_positions,
        }

        plan = {
            'candidates': candidates,
            'ball_pos': cue_pos,
        }

        # IK + 실행
        self._strike_skipped = False
        self._strike(scan_data, plan)

        if getattr(self, '_strike_skipped', False):
            print(f"  Strike skipped: {getattr(self, '_strike_skip_reason', 'unknown')}")
            # 다음 후보 시도
            for ci in range(1, min(len(candidates), 5)):
                print(f"\n  Trying candidate #{ci+1}...")
                plan['candidates'] = candidates[ci:]
                self._strike_skipped = False
                self._strike(scan_data, plan)
                if not getattr(self, '_strike_skipped', False):
                    break

        self.env.wait_balls_stop(timeout=10.0)

        # 결과 측정
        t1_final = np.array(_p.getBasePositionAndOrientation(
            self.env.target_ball_id, physicsClientId=self.env.client)[0])
        t2_final = np.array(_p.getBasePositionAndOrientation(
            self.env.ball2_id, physicsClientId=self.env.client)[0])

        # 두 가지 매칭
        d_a1 = np.linalg.norm(t1_final[:2] - target1_goal[:2])
        d_a2 = np.linalg.norm(t2_final[:2] - target2_goal[:2])
        d_b1 = np.linalg.norm(t1_final[:2] - target2_goal[:2])
        d_b2 = np.linalg.norm(t2_final[:2] - target1_goal[:2])

        if d_a1 + d_a2 <= d_b1 + d_b2:
            dist1, dist2 = d_a1, d_a2
        else:
            dist1, dist2 = d_b1, d_b2

        total = dist1 + dist2

        print(f"\n{'='*50}")
        print(f"  TRICK SHOT RESULT")
        print(f"  Ball 1 distance to target: {dist1*100:.1f}cm")
        print(f"  Ball 2 distance to target: {dist2*100:.1f}cm")
        print(f"  Total distance: {total*100:.1f}cm")
        print(f"{'='*50}")

        self.controller.move_home()
        return total < 0.10  # 10cm 이내면 성공

    def _scan_pocket(self, target_ball_idx, pocketed_list):
        """pocket_demo용 SCAN: 현재 타격 대상 공 + 다른 공 위치."""
        cue_pos = self.env.get_cue_ball_position()

        ball_getters = [
            self.env.get_target_ball_position,
            self.env.get_ball2_position,
            lambda: self.env.get_ball3_position(),
        ]

        target_pos = ball_getters[target_ball_idx]()
        if target_pos is None:
            return None

        # 포켓된 공은 테이블 아래에 있으므로 필터
        ball_ids = [
            self.env.target_ball_id,
            self.env.ball2_id,
            getattr(self.env, 'ball3_id', None),
        ]
        target_id = ball_ids[target_ball_idx]
        if target_id is not None and self.env.is_ball_pocketed(target_id):
            return None

        # 다른 공 위치 (포켓되지 않은 것만)
        other_balls = []
        for i in range(3):
            if i == target_ball_idx:
                continue
            if pocketed_list[i]:
                continue
            bid = ball_ids[i]
            if bid is None:
                continue
            if self.env.is_ball_pocketed(bid):
                continue
            pos = ball_getters[i]()
            if pos is not None:
                other_balls.append(pos)

        return {
            'cue_pos': cue_pos,
            'target_pos': target_pos,
            'other_balls': other_balls,
            'table_bounds': self.env.table_bounds,
            '_target_ball_id': target_id,
        }
