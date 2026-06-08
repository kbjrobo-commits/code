#<포켓 플래너 수정 전 0607 1543>
"""
포켓볼 타격 탐색기 — Headless PyBullet 공 직접 시뮬
===================================================
공에 직접 속도를 부여하여 빠르게 탐색 (로봇/IK/PD 없음).
Phase 1: 목적구를 포켓에 넣기 (다른 공 접촉 금지, 스크래치 금지)
Phase 2: 목적구를 마커 위치에 정밀 정지 (다른 공 접촉 금지)
"""
import numpy as np
import pybullet as p
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.config import *
from project.physics.contact_model import (
    apply_ball_dynamics, apply_table_dynamics, apply_cushion_dynamics)

# ----------------------------------------------------------------
# Escape-shot defaults
# config.py에 같은 이름이 있으면 그 값을 사용하고,
# 없으면 아래 기본값을 사용한다.
# ----------------------------------------------------------------
try:
    ESCAPE_WALL_GAP_THRESHOLD
except NameError:
    ESCAPE_WALL_GAP_THRESHOLD = MAZE_BALL_RADIUS + 0.015      # 공 표면-벽 간격 1cm 이하

try:
    ESCAPE_STRIKE_HEIGHT_OFFSET
except NameError:
    ESCAPE_STRIKE_HEIGHT_OFFSET = 0.017   # 기존 strike height보다 +1.7cm

try:
    ESCAPE_BALL_SPEED
except NameError:
    ESCAPE_BALL_SPEED = 0.2              # escape용 공 속도

try:
    ESCAPE_SAFE_APPROACH_DIST
except NameError:
    ESCAPE_SAFE_APPROACH_DIST = 0.035

try:
    ESCAPE_FOLLOW_DIST
except NameError:
    ESCAPE_FOLLOW_DIST = 0.02


class PocketShotPlanner:
    """Headless 공 시뮬 기반 포켓/정밀정지 탐색"""

    def __init__(self, table_bounds, ball_radius=MAZE_BALL_RADIUS):
        self.bounds = table_bounds
        self.ball_r = ball_radius

    # ================================================================
    # 공통: Headless 환경 생성 (포켓 갭 쿠션 포함)
    # ================================================================

    def _create_pocket_env(self):
        """포켓 갭이 있는 headless 환경 생성.

        Returns:
            sim: pybullet client id
            table_id: table body id
            cushion_ids: list of cushion segment ids
            pocket_positions: list of 6 pocket positions [x,y,z]
        """
        sim = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=sim)
        p.setTimeStep(1./240, physicsClientId=sim)

        L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
        TH, H = MAZE_TABLE_HEIGHT, MAZE_TABLE_SURFACE_HEIGHT
        CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y
        center = [CX, CY, H]

        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=sim)

        # 테이블
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2],
                                     physicsClientId=sim)
        table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                     basePosition=center, physicsClientId=sim)
        apply_table_dynamics(table_id, sim)

        # 포켓 갭 쿠션 (maze_env._create_cushions_with_pockets와 동일)
        CH = MAZE_CUSHION_HEIGHT
        top_z = center[2] + TH / 2 + CH / 2
        thickness = 0.03
        gap = POCKET_RADIUS * 1.6  # 시뮬 마진: 좁은 갭에서도 확실히 들어가는 각도만 선택

        x_min, x_max = CX - L / 2, CX + L / 2
        y_min, y_max = CY - W / 2, CY + W / 2

        cushion_ids = []

        def _add(pos, half_ext):
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                       physicsClientId=sim)
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                    basePosition=pos, physicsClientId=sim)
            apply_cushion_dynamics(cid, sim)
            cushion_ids.append(cid)

        # 상/하변: 코너2 갭 → 2세그먼트
        seg_len = (L - 2 * gap) / 2
        for y_pos in [y_max + thickness / 2, y_min - thickness / 2]:
            _add([x_min + gap + seg_len / 2, y_pos, top_z],
                 [seg_len / 2, thickness / 2, CH / 2])
            _add([x_max - gap - seg_len / 2, y_pos, top_z],
                 [seg_len / 2, thickness / 2, CH / 2])

        # 좌/우변: 코너2 + 사이드1 갭 → 2세그먼트
        seg_len_side = (W - 2 * gap - gap) / 2
        for x_pos in [x_min - thickness / 2, x_max + thickness / 2]:
            _add([x_pos, y_min + gap + seg_len_side / 2, top_z],
                 [thickness / 2, seg_len_side / 2, CH / 2])
            _add([x_pos, y_max - gap - seg_len_side / 2, top_z],
                 [thickness / 2, seg_len_side / 2, CH / 2])

        # 포켓 좌표
        sz = H + TH / 2
        pocket_positions = [
            np.array([x_min, y_min, sz]),
            np.array([x_max, y_min, sz]),
            np.array([x_min, y_max, sz]),
            np.array([x_max, y_max, sz]),
            np.array([x_min, (y_min + y_max) / 2, sz]),
            np.array([x_max, (y_min + y_max) / 2, sz]),
        ]

        return sim, table_id, cushion_ids, pocket_positions

    def _make_ball(self, sim, pos):
        """공 하나 생성 (포켓 데모 마찰 파라미터)."""
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                     physicsClientId=sim)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS,
                                baseCollisionShapeIndex=col,
                                basePosition=list(pos), physicsClientId=sim)
        apply_ball_dynamics(bid, sim)
        return bid

    def _set_rolling_velocity(self, sim, ball_id, speed, angle):
        """순수 구름 조건으로 초기 속도 부여 (슬라이딩 없음).

        현실에서 공은 치자마자 구르므로, angular velocity를
        rolling 조건 ω = v/r 에 맞춰 설정.
        """
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        # rolling: ω × r = v at contact. r = (0,0,-R)
        # ω = (v*sinθ/R, -v*cosθ/R, 0)
        r = MAZE_BALL_RADIUS
        wx = speed * np.sin(angle) / r
        wy = -speed * np.cos(angle) / r
        p.resetBaseVelocity(ball_id, [vx, vy, 0], [wx, wy, 0],
                            physicsClientId=sim)

    def _is_pocketed(self, sim, ball_id, surface_z):
        """공이 포켓에 빠졌는지 판정."""
        pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=sim)
        return pos[2] < surface_z - 0.02

    def _which_pocket(self, sim, ball_id, pocket_positions):
        """어떤 포켓에 들어갔는지 (-1 = 없음)."""
        pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=sim)
        for i, pp in enumerate(pocket_positions):
            if np.linalg.norm(np.array(pos[:2]) - pp[:2]) < POCKET_RADIUS * 3:
                if pos[2] < pp[2] - 0.01:
                    return i
        return -1

    def _compute_alignment_metric(self, cue_pos, target_pos, pocket_positions, pocket_idx=None):
        """포켓-목적구-큐볼 정렬 품질 계산.

        alignment_error_deg:
            cue -> target 방향과 target -> pocket 방향 사이의 각도.
            0도에 가까울수록 흰공-목적구-포켓이 거의 일직선이다.

        alignment_quality:
            0~1 정규화 점수. 1에 가까울수록 직선샷에 가깝다.
        """
        cue_xy = np.array(cue_pos[:2], dtype=float)
        target_xy = np.array(target_pos[:2], dtype=float)

        cue_to_target = target_xy - cue_xy
        cue_dist = np.linalg.norm(cue_to_target)
        if cue_dist < 1e-9:
            return 0.0, 180.0, -1
        cue_to_target = cue_to_target / cue_dist

        indices = [pocket_idx] if pocket_idx is not None and pocket_idx >= 0 else range(len(pocket_positions))

        best_quality = 0.0
        best_error_deg = 180.0
        best_idx = -1

        for pi in indices:
            pp = pocket_positions[pi]
            target_to_pocket = np.array(pp[:2], dtype=float) - target_xy
            pocket_dist = np.linalg.norm(target_to_pocket)
            if pocket_dist < 1e-9:
                continue
            target_to_pocket = target_to_pocket / pocket_dist

            cosang = float(np.clip(np.dot(cue_to_target, target_to_pocket), -1.0, 1.0))
            err_rad = float(np.arccos(cosang))
            err_deg = float(np.degrees(err_rad))

            # 0도 = 완전 직선, 90도 이상 = 매우 어려운 컷으로 간주
            quality = max(0.0, 1.0 - err_deg / 90.0)

            if quality > best_quality:
                best_quality = quality
                best_error_deg = err_deg
                best_idx = int(pi)

        return best_quality, best_error_deg, best_idx

    # ================================================================
    # Phase 1: 포켓 샷 탐색
    # ================================================================

    def plan_pocket_shot(self, cue_pos, target_pos, other_ball_positions,
                         next_target_pos=None):
        """하나의 목적구를 포켓에 넣는 각도 탐색.

        Args:
            cue_pos: 큐볼 현재 위치 [x,y,z]
            target_pos: 포켓에 넣을 목적구 위치 [x,y,z]
            other_ball_positions: 접촉 금지 다른 공들 [[x,y,z], ...]
            next_target_pos: 다음에 칠 공 위치 (포지션 보너스용, optional)

        Returns:
            candidates: sorted list of dicts with strike_dir, strike_speed, score, etc.
        """
        cue_3d = np.array(cue_pos).flatten()
        target_3d = np.array(target_pos).flatten()
        others_3d = [np.array(o).flatten() for o in other_ball_positions]

        # ------------------------------------------------------------
        # Escape shot priority
        # 큐볼 표면과 벽 사이 간격이 1cm 이하이면 일반 포켓샷 탐색을 하지 않고,
        # 테이블 중앙 방향으로 큐볼을 빼내는 escape 후보만 반환한다.
        # ------------------------------------------------------------
        near_wall, _, wall_info = self._detect_cue_wall_proximity(cue_3d)
        if near_wall:
            t0 = time.time()
            candidates = self._make_escape_candidates(cue_3d)
            elapsed = time.time() - t0
            if candidates:
                best = candidates[0]
                print(
                    f"  [SearchResult] mode=ESCAPE, total_sims=0, "
                    f"final_candidates={len(candidates)}, "
                    f"selected_angle={best.get('angle_deg', float('nan')):.2f}deg, "
                    f"score={best.get('score', 0):.0f}, "
                    f"escape=True, time={elapsed:.2f}s"
                )
            else:
                print("  [SearchResult] mode=ESCAPE, no valid escape candidate")
            return candidates

        sim, table_id, cushion_ids, pocket_positions = self._create_pocket_env()
        surface_z = MAZE_TABLE_SURFACE_HEIGHT + MAZE_TABLE_HEIGHT / 2

        # 공 생성
        cue_id = self._make_ball(sim, cue_3d)
        target_id = self._make_ball(sim, target_3d)
        other_ids = [self._make_ball(sim, o) for o in others_3d]

        # 안정화
        for _ in range(50):
            p.stepSimulation(physicsClientId=sim)

        # ============================================================
        # 포켓 기하학 기반 이상 각도 계산 (Ghost-Ball Aiming)
        # ============================================================
        # 각 포켓에 대해: target → pocket 방향의 반대편에 ghost ball을 놓고,
        # cue → ghost 방향이 이상적 타격 각도.
        ideal_angles = []
        for pp in pocket_positions:
            pocket_dir = pp[:2] - target_3d[:2]
            pocket_dist = np.linalg.norm(pocket_dir)
            if pocket_dist < 1e-6:
                continue
            pocket_dir_n = pocket_dir / pocket_dist
            # Ghost ball: 목적구 뒤(포켓 반대방향)에 공 2개 지름만큼 떨어진 위치
            ghost = target_3d[:2] - pocket_dir_n * (2 * MAZE_BALL_RADIUS)
            cue_to_ghost = ghost - cue_3d[:2]
            dist_cg = np.linalg.norm(cue_to_ghost)
            if dist_cg < 1e-6:
                continue
            angle = np.arctan2(cue_to_ghost[1], cue_to_ghost[0])
            ideal_angles.append(angle)

        # ============================================================
        # 계층적 각도 탐색 (Goal 4)
        #   Tier A 다이렉트 로컬 → Tier B 확장 로컬 → Tier C 글로벌 폴백.
        #   쿠션 반발이 낮고 품질이 불안정하므로 다이렉트 샷을 우선하고,
        #   상위 티어에서 포켓 성공이 없을 때만 다음 티어로 확장한다.
        # ============================================================
        speed_center = 1.25
        test_speeds = [speed_center, speed_center * 1.05, speed_center * 0.95]
        POCKET_SUCCESS_SCORE = 100000

        t0 = time.time()
        results = []
        scanned_buckets = set()  # 0.5deg 각도 버킷 중복 시뮬 방지

        def _angles_around(bases, width_deg, step_deg):
            out = []
            for base in bases:
                for off in np.arange(-width_deg, width_deg + step_deg / 2, step_deg):
                    out.append(base + np.radians(off))
            return out

        def _scan(angle_set):
            new_angles = []
            for a in angle_set:
                bucket = round(np.degrees(a) * 2) / 2
                if bucket in scanned_buckets:
                    continue
                scanned_buckets.add(bucket)
                new_angles.append(a)
            tier_results = []
            for spd in test_speeds:
                for angle in new_angles:
                    score, info = self._simulate_pocket_shot(
                        sim, cue_id, target_id, other_ids, cushion_ids,
                        pocket_positions, cue_3d, target_3d, others_3d,
                        angle, spd, surface_z)
                    tier_results.append({
                        'angle': angle, 'speed': spd, 'score': score, **info
                    })
            results.extend(tier_results)
            return new_angles, tier_results

        def _n_success(recs):
            return sum(1 for r in recs if r['score'] >= POCKET_SUCCESS_SCORE)

        # --- Tier A: 다이렉트 로컬 (±5° @0.5°) ---
        a_angles, rA = _scan(_angles_around(ideal_angles, 10.0, 0.2))
        search_mode = 'DIRECT'
        print(f"  [Search:DIRECT] {len(ideal_angles)} ideal dirs, range=±10.0deg@0.2deg, "
              f"angles={len(a_angles)}, candidates={len(rA)}, pocket_hits={_n_success(rA)}")

        # --- Tier B: 확장 로컬 (±15° @1.0°) ---
        if _n_success(results) == 0:
            b_angles, rB = _scan(_angles_around(ideal_angles, 20.0, 0.2))
            search_mode = 'EXPANDED'
            print(f"  [Search:EXPANDED] range=±20.0deg@0.2deg, "
                  f"angles={len(b_angles)}, candidates={len(rB)}, pocket_hits={_n_success(rB)}")

        # --- Tier C: 글로벌/쿠션 폴백 (360° @2.0°) ---
        if _n_success(results) == 0:
            c_set = list(np.linspace(0, 2 * np.pi, 180, endpoint=False))
            c_angles, rC = _scan(c_set)
            search_mode = 'GLOBAL'
            print(f"  [Search:GLOBAL] range=360deg@2.0deg (fallback), "
                  f"angles={len(c_angles)}, candidates={len(rC)}, pocket_hits={_n_success(rC)}")

        self._last_search_mode = search_mode

        # 상위 각도 정밀 탐색 (0.1도 간격)
        results.sort(key=lambda r: r['score'], reverse=True)
        seen = set()
        top_angles = []
        for r in results:
            bucket = round(np.degrees(r['angle']) * 2) / 2
            if bucket not in seen and r['score'] >= 1000:
                seen.add(bucket)
                top_angles.append(r['angle'])
            if len(top_angles) >= 20:
                break

        offsets = [np.radians(d) for d in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]]
        for base_angle in top_angles:
            for offset in offsets:
                a = base_angle + offset
                for spd in test_speeds:
                    score, info = self._simulate_pocket_shot(
                        sim, cue_id, target_id, other_ids, cushion_ids,
                        pocket_positions, cue_3d, target_3d, others_3d,
                        a, spd, surface_z)
                    results.append({
                        'angle': a, 'speed': spd, 'score': score, **info
                    })

        p.disconnect(sim)

        # Robustness bonus
        self._add_robustness_bonus(results, threshold=3000)

        # approach 경로 체크용 공 위치 저장
        self._target_ball_2d = target_3d[:2]
        self._other_balls_2d = [o[:2] for o in others_3d]

        candidates = self._format_candidates(results, cue_3d)
        elapsed = time.time() - t0

        if candidates:
            best = candidates[0]
            print(
                f"  [SearchResult] mode={search_mode}, total_sims={len(results)}, "
                f"final_candidates={len(candidates)}, "
                f"selected_angle={best.get('angle_deg', float('nan')):.2f}deg, "
                f"score={best.get('score', 0):.0f}, "
                f"escape={best.get('is_escape_shot', False)}, "
                f"time={elapsed:.2f}s"
            )
        else:
            print(
                f"  [SearchResult] mode={search_mode}, total_sims={len(results)}, "
                f"no valid candidate, time={elapsed:.2f}s"
            )

        return candidates

    def _simulate_pocket_shot(self, sim, cue_id, target_id, other_ids,
                               cushion_ids, pocket_positions,
                               cue_start, target_start, others_start,
                               angle, speed, surface_z, max_steps=2000):
        """공에 직접 속도 부여 → 포켓 성공/실패 판정."""
        # 리셋
        for bid, pos in [(cue_id, cue_start), (target_id, target_start)]:
            p.resetBasePositionAndOrientation(bid, list(pos), [0,0,0,1],
                                              physicsClientId=sim)
            p.resetBaseVelocity(bid, [0,0,0], [0,0,0], physicsClientId=sim)
        for bid, pos in zip(other_ids, others_start):
            p.resetBasePositionAndOrientation(bid, list(pos), [0,0,0,1],
                                              physicsClientId=sim)
            p.resetBaseVelocity(bid, [0,0,0], [0,0,0], physicsClientId=sim)

        # 시뮬 시작 전 초기 쿠션 접촉 기록 (벽에 붙은 큐볼의 기존 접촉 제외)
        # 속도 부여 전에 기록해야 공이 아직 움직이지 않은 상태의 접촉만 잡힘
        initial_cushion_contacts = set()
        for c in p.getContactPoints(bodyA=cue_id, physicsClientId=sim):
            if c[2] in cushion_ids:
                initial_cushion_contacts.add(c[2])

        # 큐볼 속도 부여 (순수 구름 조건)
        self._set_rolling_velocity(sim, cue_id, speed, angle)

        # 시뮬 + 접촉 추적
        hit_target = False
        illegal_contact = False
        cue_hit_cushion_before_target = False
        cue_scratched = False
        target_min_pocket_dist = float('inf')  # 경로 상 최소 포켓 거리
        target_closest_pocket_idx = -1
        cue_path = [[cue_start[0], cue_start[1]]]
        target_path = [[target_start[0], target_start[1]]]

        for step in range(max_steps):
            p.stepSimulation(physicsClientId=sim)

            # 큐볼 접촉 체크
            # - 큐볼이 타겟에 먼저 맞기 전에 다른 공에 접촉하면 illegal
            # - 타겟에 맞은 후에는 큐볼이 다른 공에 맞아도 허용 (당구 규칙)
            cue_contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim)
            for c in cue_contacts:
                other_body = c[2]
                if other_body == target_id:
                    hit_target = True
                elif other_body in cushion_ids and not hit_target:
                    if other_body not in initial_cushion_contacts:
                        cue_hit_cushion_before_target = True
                elif other_body in other_ids and not hit_target:
                    # 타겟을 먼저 맞추기 전에 다른 공 접촉 = illegal
                    illegal_contact = True
                    break
            if illegal_contact:
                break

            # 목적구가 다른 공에 부딪히는 것은 허용
            # (타겟이 홀로 가는 경로에 다른 공이 있어도 해를 찾을 수 있음)

            # 큐볼 포켓 스크래치 체크
            cue_pos_now, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
            for pp in pocket_positions:
                if np.linalg.norm(np.array(cue_pos_now[:2]) - pp[:2]) < POCKET_CAPTURE_RADIUS:
                    cue_scratched = True
                    break
            if cue_scratched:
                break

            # 목적구 포켓 진입 체크 (XY 반경 내 → 아래로 제거)
            target_pos_now, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=sim)
            for pi, pp in enumerate(pocket_positions):
                d = np.linalg.norm(np.array(target_pos_now[:2]) - pp[:2])
                if d < target_min_pocket_dist:
                    target_min_pocket_dist = d
                    target_closest_pocket_idx = pi
                if d < POCKET_CAPTURE_RADIUS:
                    # 포켓 진입! 공을 테이블 아래로 이동
                    p.resetBasePositionAndOrientation(
                        target_id, [pp[0], pp[1], surface_z - 0.1],
                        [0,0,0,1], physicsClientId=sim)
                    p.resetBaseVelocity(target_id, [0,0,0], [0,0,0],
                                        physicsClientId=sim)
                    break

            # 경로 기록
            if step % 20 == 0:
                pos, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
                cue_path.append([pos[0], pos[1]])
                pos_t, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=sim)
                target_path.append([pos_t[0], pos_t[1]])

            # 조기 종료: 모든 공 정지
            if step > 200 and step % 50 == 0:
                all_ids = [cue_id, target_id] + other_ids
                speeds = [np.linalg.norm(p.getBaseVelocity(bid, physicsClientId=sim)[0][:2])
                          for bid in all_ids]
                if all(s < 0.005 for s in speeds):
                    break

        # 스코어링
        target_pocketed = self._is_pocketed(sim, target_id, surface_z)
        pocket_idx = self._which_pocket(sim, target_id, pocket_positions)

        # 큐볼 최종 위치
        cue_final, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
        target_final, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=sim)

        # XY 기반 큐볼 스크래치 판정 (최종 위치)
        if not cue_scratched:
            for pp in pocket_positions:
                if np.linalg.norm(np.array(cue_final[:2]) - pp[:2]) < POCKET_CAPTURE_RADIUS:
                    cue_scratched = True
                    break
        # GUI 포켓 갭(2.0x)이 시뮬(1.6x)보다 넓어 시뮬에서 놓칠 수 있음
        # GUI 기준 넓은 반경으로 추가 판정하여 자살골 방지
        if not cue_scratched:
            gui_scratch_radius = POCKET_RADIUS * 1.5
            for pp in pocket_positions:
                if np.linalg.norm(np.array(cue_final[:2]) - pp[:2]) < gui_scratch_radius:
                    cue_scratched = True
                    break

        # XY 기반 포켓 판정 (테이블에 물리적 구멍 없으므로 z 체크만으로 불충분)
        # 최종 위치 또는 경로 상 최근접 거리 중 하나라도 POCKET_RADIUS 이내면 성공
        if not target_pocketed and hit_target:
            # 최종 위치 체크
            for i, pp in enumerate(pocket_positions):
                dist_to_pocket = np.linalg.norm(
                    np.array(target_final[:2]) - pp[:2])
                if dist_to_pocket < POCKET_CAPTURE_RADIUS:
                    target_pocketed = True
                    pocket_idx = i
                    break
        if not target_pocketed and hit_target:
            # 경로 상 최근접 체크 (시뮬 중 포켓을 스쳐 지나간 경우)
            if target_min_pocket_dist < POCKET_CAPTURE_RADIUS:
                target_pocketed = True
                pocket_idx = target_closest_pocket_idx

        alignment_idx = pocket_idx if pocket_idx >= 0 else target_closest_pocket_idx
        alignment_quality, alignment_error_deg, alignment_pocket_idx = self._compute_alignment_metric(
            cue_start, target_start, pocket_positions, alignment_idx
        )

        score = 0
        if illegal_contact:
            score = -10000
        elif cue_hit_cushion_before_target:
            score = -1000
        elif cue_scratched:
            score = -3000
        elif target_pocketed:
            # 포켓 성공 후보 내부에서도 흰공-목적구-포켓이 직선에 가까울수록 우선.
            # 최대 +30000점 보너스.
            score = 100000 + int(20000 * alignment_quality)
        elif hit_target:
            # 포켓 진입 실패지만 타격함 — 포켓까지 거리로 부분 점수 (상한 4000)
            min_pocket_dist = min(
                np.linalg.norm(np.array(target_final[:2]) - pp[:2])
                for pp in pocket_positions
            )
            score = min(4000, max(1, int(1000 / (min_pocket_dist + 0.01))))
        else:
            # 큐볼이 목적구에 맞지도 않음
            score = 0

        info = {
            'hit_target': hit_target,
            'target_pocketed': target_pocketed,
            'pocket_idx': pocket_idx,
            'illegal_contact': illegal_contact,
            'cue_hit_cushion_before_target': cue_hit_cushion_before_target,
            'cue_scratched': cue_scratched,
            'cue_final': [cue_final[0], cue_final[1]],
            'target_final': [target_final[0], target_final[1]],
            'alignment_quality': alignment_quality,
            'alignment_error_deg': alignment_error_deg,
            'alignment_pocket_idx': alignment_pocket_idx,
            'target_min_pocket_dist': target_min_pocket_dist,
            'target_closest_pocket_idx': target_closest_pocket_idx,
            'cue_path': cue_path,
            'target_path': target_path,
        }
        return score, info

    # ================================================================
    # Phase 2: 정밀 정지 탐색
    # ================================================================

    def plan_precision_shot(self, cue_pos, target_pos, marker_pos,
                             other_ball_positions):
        """하나의 목적구를 마커 위치에 정밀 정지시키는 각도+속도 탐색.

        Args:
            cue_pos: 큐볼 위치
            target_pos: 옮길 목적구 위치
            marker_pos: 목표 마커 위치 [x,y,z]
            other_ball_positions: 접촉 금지 다른 공들

        Returns:
            candidates: sorted list
        """
        cue_3d = np.array(cue_pos).flatten()
        target_3d = np.array(target_pos).flatten()
        marker_3d = np.array(marker_pos).flatten()
        others_3d = [np.array(o).flatten() for o in other_ball_positions]

        sim, table_id, cushion_ids, pocket_positions = self._create_pocket_env()
        surface_z = MAZE_TABLE_SURFACE_HEIGHT + MAZE_TABLE_HEIGHT / 2

        cue_id = self._make_ball(sim, cue_3d)
        target_id = self._make_ball(sim, target_3d)
        other_ids = [self._make_ball(sim, o) for o in others_3d]

        for _ in range(50):
            p.stepSimulation(physicsClientId=sim)

        # 마커 방향 기반 이상 각도 계산 (Ghost-Ball Aiming)
        marker_dir = marker_3d[:2] - target_3d[:2]
        marker_dist = np.linalg.norm(marker_dir)
        if marker_dist > 1e-6:
            marker_dir_n = marker_dir / marker_dist
            ghost = target_3d[:2] - marker_dir_n * (2 * MAZE_BALL_RADIUS)
            cue_to_ghost = ghost - cue_3d[:2]
            ideal_angle = np.arctan2(cue_to_ghost[1], cue_to_ghost[0])
        else:
            ideal_angle = np.arctan2(
                target_3d[1] - cue_3d[1], target_3d[0] - cue_3d[0])

        # 이상 각도 ±20도 정밀 탐색 + 글로벌 보충
        search_angles = set()
        for offset_deg in np.arange(-20, 20.5, 0.5):
            search_angles.add(ideal_angle + np.radians(offset_deg))
        for a in np.linspace(0, 2 * np.pi, 120, endpoint=False):
            search_angles.add(a)
        search_angles = sorted(search_angles)

        speed_min, speed_max = PRECISION_SPEED_RANGE
        n_speeds = PRECISION_SPEED_STEPS
        speeds = np.linspace(speed_min, speed_max, n_speeds)

        print(f"  [PrecisionSearch] ideal_angle={np.degrees(ideal_angle):.1f}, "
              f"{len(search_angles)} angles × {n_speeds} speeds, "
              f"marker_dist={marker_dist*100:.1f}cm")

        results = []
        for spd in speeds:
            for angle in search_angles:
                score, info = self._simulate_precision_shot(
                    sim, cue_id, target_id, other_ids, pocket_positions,
                    cue_3d, target_3d, marker_3d, others_3d,
                    angle, spd, surface_z)
                results.append({
                    'angle': angle, 'speed': spd, 'score': score, **info
                })

        # 상위 정밀 탐색
        results.sort(key=lambda r: r['score'], reverse=True)
        seen = set()
        top_combos = []
        for r in results:
            key = (round(np.degrees(r['angle'])), round(r['speed'] * 10))
            if key not in seen and r['score'] >= 2000:
                seen.add(key)
                top_combos.append((r['angle'], r['speed']))
            if len(top_combos) >= 10:
                break

        a_offsets = [np.radians(d) for d in [-0.3, -0.1, 0.1, 0.3]]
        s_offsets = [-0.03, -0.01, 0.01, 0.03]
        for base_a, base_s in top_combos:
            for ao in a_offsets:
                for so in s_offsets:
                    spd = max(speed_min, min(speed_max, base_s + so))
                    score, info = self._simulate_precision_shot(
                        sim, cue_id, target_id, other_ids, pocket_positions,
                        cue_3d, target_3d, marker_3d, others_3d,
                        base_a + ao, spd, surface_z)
                    results.append({
                        'angle': base_a + ao, 'speed': spd,
                        'score': score, **info
                    })

        # 3단계: Ultra-fine (0.005 m/s × 0.05도)
        results.sort(key=lambda r: r['score'], reverse=True)
        seen_ultra = set()
        top_ultra = []
        for r in results:
            key = (round(np.degrees(r['angle']) * 2), round(r['speed'] * 20))
            if key not in seen_ultra and r['score'] >= 500:
                seen_ultra.add(key)
                top_ultra.append((r['angle'], r['speed']))
            if len(top_ultra) >= 5:
                break

        ultra_a = [np.radians(d) for d in [-0.1, -0.05, 0.05, 0.1]]
        ultra_s = np.arange(-0.05, 0.055, 0.005)
        for base_a, base_s in top_ultra:
            for ao in ultra_a:
                for so in ultra_s:
                    spd = max(speed_min, min(speed_max, base_s + so))
                    score, info = self._simulate_precision_shot(
                        sim, cue_id, target_id, other_ids, pocket_positions,
                        cue_3d, target_3d, marker_3d, others_3d,
                        base_a + ao, spd, surface_z)
                    results.append({
                        'angle': base_a + ao, 'speed': spd,
                        'score': score, **info
                    })

        p.disconnect(sim)

        self._add_robustness_bonus(results, threshold=2000)

        # approach 경로 체크용 공 위치 저장
        self._target_ball_2d = target_3d[:2]
        self._other_balls_2d = [o[:2] for o in others_3d]

        return self._format_candidates(results, cue_3d, is_precision=True)

    def _simulate_precision_shot(self, sim, cue_id, target_id, other_ids,
                                  pocket_positions,
                                  cue_start, target_start, marker_pos,
                                  others_start, angle, speed, surface_z,
                                  max_steps=2000):
        """공에 속도 부여 → 정밀 정지 판정."""
        # 리셋
        for bid, pos in [(cue_id, cue_start), (target_id, target_start)]:
            p.resetBasePositionAndOrientation(bid, list(pos), [0,0,0,1],
                                              physicsClientId=sim)
            p.resetBaseVelocity(bid, [0,0,0], [0,0,0], physicsClientId=sim)
        for bid, pos in zip(other_ids, others_start):
            p.resetBasePositionAndOrientation(bid, list(pos), [0,0,0,1],
                                              physicsClientId=sim)
            p.resetBaseVelocity(bid, [0,0,0], [0,0,0], physicsClientId=sim)

        self._set_rolling_velocity(sim, cue_id, speed, angle)

        hit_target = False
        illegal_contact = False
        cue_scratched = False

        for step in range(max_steps):
            p.stepSimulation(physicsClientId=sim)

            cue_contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim)
            for c in cue_contacts:
                other_body = c[2]
                if other_body == target_id:
                    hit_target = True
                elif other_body in other_ids:
                    illegal_contact = True
                    break
            if illegal_contact:
                break

            if hit_target:
                target_contacts = p.getContactPoints(bodyA=target_id, physicsClientId=sim)
                for c in target_contacts:
                    if c[2] in other_ids:
                        illegal_contact = True
                        break
                if illegal_contact:
                    break

            cue_pos_now, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
            for pp in pocket_positions:
                if np.linalg.norm(np.array(cue_pos_now[:2]) - pp[:2]) < POCKET_CAPTURE_RADIUS:
                    cue_scratched = True
                    break
            if cue_scratched:
                break

            if step > 200 and step % 50 == 0:
                all_ids = [cue_id, target_id] + other_ids
                speeds_check = [np.linalg.norm(
                    p.getBaseVelocity(bid, physicsClientId=sim)[0][:2])
                    for bid in all_ids]
                if all(s < 0.005 for s in speeds_check):
                    break

        # 정지 위치 판정
        target_final, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=sim)
        cue_final, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
        distance = np.linalg.norm(np.array(target_final[:2]) - marker_pos[:2])

        score = 0
        if illegal_contact:
            score = -10000
        elif cue_scratched:
            score = -5000
        elif hit_target:
            if distance <= PRECISION_STOP_TOLERANCE:
                # 성공: 1cm 이내
                score = 100000 + int(1000 / (distance + 0.001))
            else:
                # 거리 역수 부분 점수 (상한 4000, 거리가 가까울수록 높음)
                score = min(4000, max(1, int(100 / (distance + 0.01))))
        else:
            score = 0

        info = {
            'hit_target': hit_target,
            'precision_distance': distance,
            'illegal_contact': illegal_contact,
            'cue_scratched': cue_scratched,
            'cue_final': [cue_final[0], cue_final[1]],
            'target_final': [target_final[0], target_final[1]],
        }
        return score, info

    # ================================================================
    # 공통 유틸
    # ================================================================

    def _add_robustness_bonus(self, results, threshold=3000):
        """인접 각도에서 복수 성공 시 bonus."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in results:
            key = round(np.degrees(r['angle']) * 4) / 4
            buckets[key].append(r)
        for key, group in buckets.items():
            n_success = sum(1 for r in group if r['score'] >= threshold)
            if n_success >= 2:
                for r in group:
                    if r['score'] >= threshold:
                        r['score'] += n_success * 1000
                        r['robust_count'] = n_success

    def _format_candidates(self, results, cue_pos, is_precision=False):
        """결과를 state_machine이 기대하는 후보 형식으로 변환."""
        SAFE_RADIUS = 0.65

        # 후보 필터링: score>0 후보를 우선하고, 없으면 legal 후보로 fallback.
        # score<=0 (빗나간/illegal/scratch)인 후보가 alignment만 좋아서
        # 선택되면 자살골이 발생하므로, score>0 대안이 있으면 제외한다.
        scored = [r for r in results if r['score'] > 0]
        if scored:
            positive = scored
        else:
            # score>0인 후보가 없으면 legal+non-scratch 후보로 fallback
            positive = [
                r for r in results
                if not r.get('illegal_contact', False)
                and not r.get('cue_hit_cushion_before_target', False)
                and not r.get('cue_scratched', False)
            ]
        if not positive:
            positive = sorted(results, key=lambda r: r['score'], reverse=True)[:7]

        # 도달가능성 + 다양성 필터: score 우선, alignment은 tiebreak
        positive.sort(
            key=lambda r: (
                1 if r.get('target_pocketed', False) else 0,
                r.get('score', -float('inf')),
                r.get('alignment_quality', 0.0),
            ),
            reverse=True
        )
        candidates = []
        seen_angles = set()

        for r in positive:
            angle = r['angle']
            angle_deg = np.degrees(angle) % 360
            strike_dir = np.array([np.cos(angle), np.sin(angle)])

            # 로봇 도달 가능 체크
            if abs(TOOL_YAW_OFFSET) > 1e-6:
                ee_y = np.array([strike_dir[1], -strike_dir[0]])
                tool_dir = strike_dir * np.cos(TOOL_YAW_OFFSET) + ee_y * np.sin(TOOL_YAW_OFFSET)
                ee_off = np.array([
                    -tool_dir[0] * TOOL_HORIZONTAL_EXT,
                    -tool_dir[1] * TOOL_HORIZONTAL_EXT,
                    TOOL_VERTICAL_DROP
                ])
            else:
                ee_off = np.array([
                    -strike_dir[0] * TOOL_HORIZONTAL_EXT,
                    -strike_dir[1] * TOOL_HORIZONTAL_EXT,
                    TOOL_VERTICAL_DROP
                ])

            ready = cue_pos[:2] + ee_off[:2] - strike_dir * STRIKE_APPROACH_DIST
            if np.linalg.norm(ready) > SAFE_RADIUS:
                continue

            # 벽 근접 타격 방지 — 벽에 수직으로 향하는 샷만 차단
            # 벽에 평행(따라치기)하거나 벽에서 멀어지는 방향은 허용
            sd2 = strike_dir[:2]
            cue2 = cue_pos[:2]
            wall_margin = 0.03
            wall_too_close = False
            dx_max = self.bounds['x_max'] - cue2[0]
            dx_min = cue2[0] - self.bounds['x_min']
            dy_max = self.bounds['y_max'] - cue2[1]
            dy_min = cue2[1] - self.bounds['y_min']
            # X축 벽: 큐볼이 벽에 3cm 이내이고, 타격 방향의 벽 수직 성분이 지배적(>70%)
            sd2_norm = np.linalg.norm(sd2)
            if sd2_norm > 1e-6:
                sd2_unit = sd2 / sd2_norm
                # +x 벽 근처에서 +x 방향으로 치는 경우
                if dx_max < wall_margin and sd2_unit[0] > 0.7:
                    wall_too_close = True
                # -x 벽 근처에서 -x 방향으로 치는 경우
                if dx_min < wall_margin and sd2_unit[0] < -0.7:
                    wall_too_close = True
                # +y 벽 근처에서 +y 방향으로 치는 경우
                if dy_max < wall_margin and sd2_unit[1] > 0.7:
                    wall_too_close = True
                # -y 벽 근처에서 -y 방향으로 치는 경우
                if dy_min < wall_margin and sd2_unit[1] < -0.7:
                    wall_too_close = True
            if wall_too_close:
                continue

            # approach 경로(ready→cue)에 다른 공/목적구가 있는지 체크
            # 직선 ready→cue 와 각 공 중심의 최소 거리 계산
            safe_clearance = MAZE_BALL_RADIUS * 2 + TOOL_TIP_RADIUS  # 공+툴 반경
            approach_blocked = False
            ready_2d = cue2 - sd2 * STRIKE_APPROACH_DIST
            approach_vec = cue2 - ready_2d
            approach_len = np.linalg.norm(approach_vec)
            if approach_len > 1e-6:
                approach_dir = approach_vec / approach_len
                # 목적구 + 다른 공 위치 (시뮬에서의 현재 위치)
                all_check_pos = []
                if hasattr(r, 'target_pos'):
                    all_check_pos.append(np.array(r['target_pos'][:2]) if 'target_pos' in r else None)
                # _format_candidates는 cue_pos 외에 다른 공 위치를 직접 갖고 있지 않으므로
                # self._other_balls_2d에 저장해둔 것을 사용
                for ob_pos in getattr(self, '_other_balls_2d', []):
                    # ready→cue 직선과 공 중심의 거리
                    to_ball = ob_pos - ready_2d
                    proj = np.dot(to_ball, approach_dir)
                    if proj < 0 or proj > approach_len:
                        continue  # 범위 밖
                    perp = np.linalg.norm(to_ball - proj * approach_dir)
                    if perp < safe_clearance:
                        approach_blocked = True
                        break
                # 목적구도 체크 (approach 경로에 목적구가 있으면 안됨 - 큐볼 앞에 있는 경우)
                if not approach_blocked and hasattr(self, '_target_ball_2d'):
                    tgt2d = self._target_ball_2d
                    to_tgt = tgt2d - ready_2d
                    proj = np.dot(to_tgt, approach_dir)
                    # 큐볼 바로 앞이 아닌 경우만 체크 (큐볼 앞에 있는 건 당연)
                    if 0 < proj < approach_len * 0.9:
                        perp = np.linalg.norm(to_tgt - proj * approach_dir)
                        if perp < safe_clearance:
                            approach_blocked = True
            if approach_blocked:
                continue

            # safe_approach_dist 동적 계산
            tip_margin = TOOL_TIP_RADIUS
            safe_approach = STRIKE_APPROACH_DIST
            for axis in [0, 1]:
                if abs(sd2[axis]) > 1e-6:
                    if sd2[axis] > 0:
                        max_a = (cue2[axis] - (self.bounds['x_min' if axis == 0 else 'y_min'] + tip_margin)) / sd2[axis]
                    else:
                        max_a = (cue2[axis] - (self.bounds['x_max' if axis == 0 else 'y_max'] - tip_margin)) / sd2[axis]
                    if max_a > 0:
                        safe_approach = min(safe_approach, max_a)
            # 벽 근처에서는 approach를 줄여야 하므로 최소 5mm까지 허용
            safe_approach = max(0.005, min(STRIKE_APPROACH_DIST, safe_approach))

            # 다양성: 1도 이내 중복 방지 (후보 5개 확보를 위해 기존 3도보다 완화)
            bucket = round(angle_deg)
            too_close = any(
                min(abs(angle_deg - s), 360 - abs(angle_deg - s)) < 1.0
                for s in seen_angles
            )
            if too_close:
                continue
            seen_angles.add(angle_deg)

            # tool_speed → ball_speed 변환 결정
            if is_precision:
                # Phase 2: 속도 가변 — tool_speed_for_ball_speed 역변환
                ball_speed = r['speed']
                tool_speed = ball_speed / BALL_SPEED_GAIN if BALL_SPEED_GAIN > 0 else MAX_TOOL_SPEED
                tool_speed = min(tool_speed, MAX_TOOL_SPEED)
            else:
                tool_speed = MAX_TOOL_SPEED

            candidates.append({
                'strike_dir': strike_dir,
                'strike_speed': tool_speed,
                'ball_speed': r['speed'],
                'score': r['score'],
                'angle_deg': angle_deg,
                'angle': angle,
                'safe_approach_dist': safe_approach,
                'cue_path': r.get('cue_path'),
                'target_path': r.get('target_path'),
                'tgt1_path': r.get('target_path'),  # state_machine 호환
                'tgt2_path': None,
                'hit_target': r.get('hit_target', False),
                'target_pocketed': r.get('target_pocketed', False),
                'pocket_idx': r.get('pocket_idx', -1),
                'alignment_quality': r.get('alignment_quality', 0.0),
                'alignment_error_deg': r.get('alignment_error_deg', 180.0),
                'alignment_pocket_idx': r.get('alignment_pocket_idx', -1),
                'target_min_pocket_dist': r.get('target_min_pocket_dist'),
                'target_closest_pocket_idx': r.get('target_closest_pocket_idx'),
                'precision_distance': r.get('precision_distance'),
                'illegal_contact': r.get('illegal_contact', False),
                'cue_hit_cushion_before_target': r.get(
                    'cue_hit_cushion_before_target', False),
                'cue_scratched': r.get('cue_scratched', False),
                'cue_final': r.get('cue_final'),
                'cushion_count': 1 if r.get('cue_hit_cushion_before_target') else 0,
                'hit_t1': r.get('hit_target', False),
                'hit_t2': False,
                'events': [],
            })

            if len(candidates) >= 25:
                break

        # 최종 정렬: 포켓 성공 > 정렬 품질 > score 순서
        candidates.sort(
            key=lambda c: (
                1 if c.get('target_pocketed', False) else 0,
                c.get('alignment_quality', 0.0),
                c.get('score', -float('inf')),
            ),
            reverse=True
        )

        n_success = sum(1 for c in candidates
                        if c.get('target_pocketed') or
                        (c.get('precision_distance') is not None and
                         c['precision_distance'] <= PRECISION_STOP_TOLERANCE))
        print(f"  [PocketPlanner] {len(candidates)} candidates, {n_success} successes")
        if candidates:
            top = candidates[0]
            print(f"  Top: angle={top['angle_deg']:.1f}deg, score={top['score']}, align={top.get('alignment_error_deg', 180.0):.1f}deg")

        return candidates

    # ================================================================
    # Phase 2 POSTECH: 트릭샷 탐색
    # ================================================================

    def plan_trick_shot(self, cue_pos, trick1_pos, trick2_pos,
                        target1_goal, target2_goal,
                        c_ball_positions):
        """트릭샷: 큐볼 한 번으로 trick ball 2개를 목표 위치로 보내는 탐색.

        Args:
            cue_pos: 큐볼 위치 [x,y,z]
            trick1_pos: trick ball 1 현재 위치 [x,y,z]
            trick2_pos: trick ball 2 현재 위치 [x,y,z]
            target1_goal: trick ball 1 목표 위치 [x,y,z]
            target2_goal: trick ball 2 목표 위치 [x,y,z]
            c_ball_positions: C형 공들 위치 [[x,y,z], ...]

        Returns:
            list of dicts sorted by score (best first)
        """
        cue_3d = np.array(cue_pos).flatten()
        t1_3d = np.array(trick1_pos).flatten()
        t2_3d = np.array(trick2_pos).flatten()
        g1 = np.array(target1_goal).flatten()[:2]
        g2 = np.array(target2_goal).flatten()[:2]
        c_balls_3d = [np.array(c).flatten() for c in c_ball_positions]

        # 전체 360° 탐색 (쿠션 바운스, 캐롬 포함)
        center_dir = (np.array(t1_3d[:2]) + np.array(t2_3d[:2])) / 2 - cue_3d[:2]
        center_angle = np.arctan2(center_dir[1], center_dir[0])
        
        # 전체 360°, 1° 단위 + 넓은 속도 범위
        speeds = np.arange(0.3, 2.0, 0.1)
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

        print(f"  [TrickSearch] center_angle={np.degrees(center_angle):.1f}°, "
              f"{len(angles)} angles × {len(speeds)} speeds = {len(angles)*len(speeds)} trials")

        sim, table_id, cushion_ids, pocket_positions = self._create_pocket_env()
        surface_z = self.bounds['z'] if 'z' in self.bounds else (
            MAZE_TABLE_SURFACE_HEIGHT + MAZE_TABLE_HEIGHT / 2)

        def _run_sim(sim, angle, speed):
            """단일 시뮬레이션 실행, 결과 반환."""
            cue_id = self._make_ball(sim, cue_3d)
            t1_id = self._make_ball(sim, t1_3d)
            t2_id = self._make_ball(sim, t2_3d)
            c_ids = [self._make_ball(sim, c) for c in c_balls_3d]
            self._set_rolling_velocity(sim, cue_id, speed, angle)

            for step in range(1500):
                p.stepSimulation(physicsClientId=sim)
                if step > 200 and step % 50 == 0:
                    vels = [np.linalg.norm(p.getBaseVelocity(bid, physicsClientId=sim)[0][:2])
                            for bid in [cue_id, t1_id, t2_id]]
                    if max(vels) < 0.005:
                        break

            t1_f = np.array(p.getBasePositionAndOrientation(t1_id, physicsClientId=sim)[0][:2])
            t2_f = np.array(p.getBasePositionAndOrientation(t2_id, physicsClientId=sim)[0][:2])

            dist_a = np.linalg.norm(t1_f - g1) + np.linalg.norm(t2_f - g2)
            dist_b = np.linalg.norm(t1_f - g2) + np.linalg.norm(t2_f - g1)
            total_dist = min(dist_a, dist_b)
            match = 'normal' if dist_a <= dist_b else 'swapped'

            t1_moved = np.linalg.norm(t1_f - t1_3d[:2]) > 0.03
            t2_moved = np.linalg.norm(t2_f - t2_3d[:2]) > 0.03
            cue_pocketed = self._is_pocketed(sim, cue_id, surface_z)
            t1_pocketed = self._is_pocketed(sim, t1_id, surface_z)
            t2_pocketed = self._is_pocketed(sim, t2_id, surface_z)

            c_ball_displaced = False
            for ci_idx, cid_check in enumerate(c_ids):
                c_pos, _ = p.getBasePositionAndOrientation(cid_check, physicsClientId=sim)
                c_disp = np.linalg.norm(np.array(c_pos[:2]) - c_balls_3d[ci_idx][:2])
                if c_disp > 0.01:
                    c_ball_displaced = True
                    break

            for bid in [cue_id, t1_id, t2_id] + c_ids:
                p.removeBody(bid, physicsClientId=sim)

            score = 0
            if t1_moved and t2_moved and not t1_pocketed and not t2_pocketed:
                dist_score = max(0, 300000 * (1 - total_dist / 0.3))
                score = int(dist_score)
                if c_ball_displaced:
                    score = max(score // 4, 1)
                if cue_pocketed:
                    score = max(score // 2, 1)
                if total_dist < 0.02:
                    score += 500000
                elif total_dist < 0.04:
                    score += 200000
                elif total_dist < 0.08:
                    score += 50000

            return {
                'angle': angle,
                'angle_deg': np.degrees(angle) % 360,
                'speed': speed,
                'score': score,
                'total_dist': total_dist,
                'match': match,
                'c_ball_hit': c_ball_displaced,
                'cue_scratched': cue_pocketed,
                't1_pocketed': t1_pocketed,
                't2_pocketed': t2_pocketed,
                't1_dist': min(np.linalg.norm(t1_f - g1), np.linalg.norm(t1_f - g2)),
                't2_dist': min(np.linalg.norm(t2_f - g2), np.linalg.norm(t2_f - g1)),
            }

        # ====== Stage 1: Coarse search (360°, 1° steps) ======
        results = []
        for angle in angles:
            for speed in speeds:
                r = _run_sim(sim, angle, speed)
                if r['score'] > 0:
                    results.append(r)

        results.sort(key=lambda r: -r['score'])
        n_coarse = sum(1 for r in results if r['total_dist'] < 0.06)
        print(f"  [Coarse] {len(results)} results, {n_coarse} within 6cm")
        if results:
            print(f"  [Coarse] Best: angle={results[0]['angle_deg']:.1f}°, "
                  f"speed={results[0]['speed']:.1f}m/s, dist={results[0]['total_dist']*100:.1f}cm")

        # ====== Stage 2: Fine search (top 3 근처 ±5°, ±0.2m/s) ======
        seen_centers = set()
        fine_results = []
        for coarse_r in results[:5]:
            ca = coarse_r['angle']
            cs = coarse_r['speed']
            bucket = (round(np.degrees(ca)), round(cs * 10))
            if any(abs(round(np.degrees(ca)) - s[0]) < 8 and abs(round(cs*10) - s[1]) < 3
                   for s in seen_centers):
                continue
            seen_centers.add(bucket)

            fine_angles = np.arange(ca - np.radians(5), ca + np.radians(5.1), np.radians(0.3))
            fine_speeds = np.arange(max(0.2, cs - 0.2), cs + 0.21, 0.02)

            for fa in fine_angles:
                for fs in fine_speeds:
                    r = _run_sim(sim, fa, fs)
                    if r['score'] > 0:
                        fine_results.append(r)

        p.disconnect(sim)

        # coarse + fine 병합
        all_results = results + fine_results
        all_results.sort(key=lambda r: -r['score'])

        # 중복 제거 (각도 1° 이내, 속도 0.05 이내)
        deduped = []
        for r in all_results:
            too_close = any(
                abs(r['angle_deg'] - d['angle_deg']) < 1 and abs(r['speed'] - d['speed']) < 0.05
                for d in deduped
            )
            if not too_close:
                deduped.append(r)
        results = deduped

        # 정렬
        results.sort(key=lambda r: -r['score'])

        n_good = sum(1 for r in results if r['total_dist'] < 0.04)
        print(f"  [TrickPlanner] {len(results)} results, {n_good} within 4cm")
        if results:
            top = results[0]
            print(f"  Top: angle={top['angle_deg']:.1f}°, speed={top['speed']:.1f}m/s, "
                  f"dist={top['total_dist']*100:.1f}cm, score={top['score']}")

        # format_candidates 스타일로 변환
        cue_pos_2d = cue_3d[:2]
        candidates = []
        seen_angles = set()

        for r in results[:50]:
            angle = r['angle']
            angle_deg = r['angle_deg']
            strike_dir = np.array([np.cos(angle), np.sin(angle)])

            # diversity
            bucket = round(angle_deg)
            too_close = any(
                min(abs(angle_deg - s), 360 - abs(angle_deg - s)) < 2
                for s in seen_angles
            )
            if too_close:
                continue
            seen_angles.add(angle_deg)

            # tool speed
            tool_speed = r['speed'] / BALL_SPEED_GAIN if BALL_SPEED_GAIN > 0 else MAX_TOOL_SPEED
            tool_speed = min(tool_speed, MAX_TOOL_SPEED)

            candidates.append({
                'strike_dir': strike_dir,
                'strike_speed': tool_speed,
                'ball_speed': r['speed'],
                'score': r['score'],
                'angle_deg': angle_deg,
                'angle': angle,
                'safe_approach_dist': STRIKE_APPROACH_DIST,
                'total_dist': r['total_dist'],
                'match': r['match'],
                'c_ball_hit': r['c_ball_hit'],
                'cue_scratched': r['cue_scratched'],
                'cushion_count': 0,
                'hit_t1': True,
                'hit_t2': True,
                'events': [],
                'cue_path': None,
                'tgt1_path': None,
                'tgt2_path': None,
            })

            if len(candidates) >= 15:
                break

        # approach 경로 체크용
        self._target_ball_2d = (t1_3d[:2] + t2_3d[:2]) / 2
        self._other_balls_2d = [c[:2] for c in c_balls_3d] + [t1_3d[:2], t2_3d[:2]]

        print(f"  [TrickPlanner] {len(candidates)} diverse candidates")
        return candidates

    def _detect_cue_wall_proximity(self, cue_pos):
        """큐볼이 테이블 벽에서 1cm 이하인지 확인하고 안쪽 방향을 반환한다.

        Returns:
            near_wall: bool
            inward_dir: np.array([dx, dy]) or None
            wall_info: dict
        """
        cue = np.array(cue_pos).flatten()
        cue_xy = cue[:2]
        b = self.bounds

        # 공 중심이 아니라 "공 표면과 벽 사이 간격" 기준.
        # 예: x_min 벽과의 실제 여유 = (공 중심 x - x_min) - 공 반지름
        distances = {
            "x_min": cue_xy[0] - b["x_min"] - self.ball_r,
            "x_max": b["x_max"] - cue_xy[0] - self.ball_r,
            "y_min": cue_xy[1] - b["y_min"] - self.ball_r,
            "y_max": b["y_max"] - cue_xy[1] - self.ball_r,
        }

        near_keys = [
            key for key, dist in distances.items()
            if dist <= ESCAPE_WALL_GAP_THRESHOLD
        ]

        if not near_keys:
            return False, None, {
                "distances": distances,
                "near_keys": [],
                "min_gap": min(distances.values()),
            }

        inward = np.zeros(2)

        # x_min 벽에 붙음 → +x 방향으로 탈출
        if "x_min" in near_keys:
            inward += np.array([1.0, 0.0])

        # x_max 벽에 붙음 → -x 방향으로 탈출
        if "x_max" in near_keys:
            inward += np.array([-1.0, 0.0])

        # y_min 벽에 붙음 → +y 방향으로 탈출
        if "y_min" in near_keys:
            inward += np.array([0.0, 1.0])

        # y_max 벽에 붙음 → -y 방향으로 탈출
        if "y_max" in near_keys:
            inward += np.array([0.0, -1.0])

        norm = np.linalg.norm(inward)
        if norm < 1e-9:
            return False, None, {
                "distances": distances,
                "near_keys": near_keys,
                "min_gap": min(distances.values()),
            }

        inward = inward / norm

        return True, inward, {
            "distances": distances,
            "near_keys": near_keys,
            "min_gap": min(distances.values()),
        }


    def _make_escape_candidates(self, cue_pos):
        """벽에 붙은 큐볼을 테이블 중앙 쪽으로 빼내는 escape shot 후보 생성."""
        near_wall, inward_dir, wall_info = self._detect_cue_wall_proximity(cue_pos)

        if not near_wall:
            return []

        cue = np.array(cue_pos).flatten()
        base_angle = np.arctan2(inward_dir[1], inward_dir[0])

        # 중앙 방향 기준으로 약간 좌우 후보도 생성
        angle_offsets_deg = [0.0, -8.0, 8.0, -15.0, 15.0]
        speed_list = [
            ESCAPE_BALL_SPEED,
            ESCAPE_BALL_SPEED * 0.85,
            ESCAPE_BALL_SPEED * 1.15,
        ]

        candidates = []

        print(
            f"  [EscapeShot] cue near wall: keys={wall_info['near_keys']}, "
            f"min_gap={wall_info['min_gap']*100:.1f}cm, "
            f"inward_dir=[{inward_dir[0]:.2f}, {inward_dir[1]:.2f}]"
        )

        for off_deg in angle_offsets_deg:
            angle = base_angle + np.radians(off_deg)
            strike_dir = np.array([np.cos(angle), np.sin(angle)])
            angle_deg = np.degrees(angle) % 360

            for spd in speed_list:
                # escape는 포켓 성공 후보가 아니므로 score는 일반 포켓 성공보다 낮게 둔다.
                # 단, 후보가 없을 때 fallback으로 선택될 수 있게 양수 점수를 준다.
                score = 5000 - abs(off_deg) * 50

                candidates.append({
                    "strike_dir": strike_dir,
                    "strike_speed": MAX_TOOL_SPEED,
                    "ball_speed": spd,
                    "score": score,
                    "angle_deg": angle_deg,
                    "angle": angle,

                    # escape 전용 필드
                    "is_escape_shot": True,
                    "escape_wall_keys": wall_info["near_keys"],
                    "escape_min_gap": wall_info["min_gap"],
                    "strike_height_offset": ESCAPE_STRIKE_HEIGHT_OFFSET,
                    "safe_approach_dist": ESCAPE_SAFE_APPROACH_DIST,
                    "follow_dist": ESCAPE_FOLLOW_DIST,

                    # pocket candidate 호환 필드
                    "target_pocketed": False,
                    "hit_target": False,
                    "illegal_contact": False,
                    "cue_scratched": False,
                    "pocket_idx": -1,
                    "cue_path": None,
                    "target_path": None,
                    "tgt1_path": None,
                    "tgt2_path": None,
                    "cushion_count": 0,
                    "hit_t1": False,
                    "hit_t2": False,
                    "events": [],
                })

        candidates.sort(key=lambda c: -c["score"])

        print(
            f"  [EscapeShot] generated {len(candidates)} escape candidates, "
            f"top_angle={candidates[0]['angle_deg']:.1f}deg"
        )

        return candidates
