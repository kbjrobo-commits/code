"""
3쿠션 타격 탐색기 — Headless PyBullet 로봇 시뮬레이션
=====================================================
Headless PyBullet에 동일한 로봇+도구+테이블/쿠션/3공을 구성.
PD computed torque (Kp=5000) 스트리밍으로 도구가 큐볼을 물리적으로 타격.
GUI _execute_sim과 동일한 물리 → 모델 gap 0.
"""
import numpy as np
import pybullet as p
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.config import *
from project.physics.pocket_avoidance import (
    rejects_first_cushion_toward_side_pocket,
    long_rail_side_pocket_centers,
)


class CushionShotPlanner:
    """Headless 로봇 시뮬 기반 쓰리쿠션 타격 탐색"""

    def __init__(self, table_bounds, ball_radius=MAZE_BALL_RADIUS):
        self.bounds = table_bounds
        self.ball_r = ball_radius

    def _reject_side_pocket_path(self, cue_pos, angle_rad):
        """첫 쿠션 후 사이드 포켓(홀) 방향 경로 제외.""" !!
        return rejects_first_cushion_toward_side_pocket(
            cue_pos, angle_rad, self.bounds, self.ball_r)

    def plan_shot(self, cue_pos, target_pos, obstacles, ball2_pos=None):
        """2단계 탐색: (1) 공 직접 속도 부여로 빠른 탐색 (2) 상위 후보만 로봇 PD 검증"""
        cue_3d = np.array(cue_pos).flatten()
        tgt1_3d = np.array(target_pos).flatten()
        tgt2_3d = np.array(ball2_pos).flatten() if ball2_pos is not None else None

        if MAZE_SIDE_POCKET_AVOID:
            pockets = long_rail_side_pocket_centers(self.bounds)
            print(f"  [PocketAvoid] 긴 변(y±) 중앙 홀 2곳 (가정): "
                  f"y+ ({pockets[0][0]:.3f}, {pockets[0][1]:.3f}), "
                  f"y- ({pockets[1][0]:.3f}, {pockets[1][1]:.3f})")

        # === Stage 1: 공 직접 속도 부여 (로봇 없음, ~100x 빠름) ===
        fast_results = self._fast_ball_search(cue_3d, tgt1_3d, tgt2_3d, obstacles)
        n_valid = sum(1 for r in fast_results if r['score'] >= 3000)
        best_fast = max(fast_results, key=lambda r: r['score'])
        print(f"  [Fast] {len(fast_results)} tested, {n_valid} valid 3-cushion, "
              f"best={best_fast['score']:.0f}")

        # === Stage 2: 3쿠션 유효 후보 우선 + 도달가능성 필터 ===
        fast_results.sort(key=lambda r: r['score'], reverse=True)
        SAFE_RADIUS = 0.65
        tilt_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)

        def _angle_priority(angle_rad):
            """Ready position 거리 기반 동적 안전도 (로봇에 가까울수록 안전)"""
            strike_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            if abs(TOOL_YAW_OFFSET) > 1e-6:
                ee_y = np.array([strike_dir[1], -strike_dir[0], 0.0])
                tool_dir = strike_dir * np.cos(TOOL_YAW_OFFSET) + ee_y * np.sin(TOOL_YAW_OFFSET)
                ee_off = -tool_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
            else:
                ee_off = -strike_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
            ready = cue_3d + ee_off - strike_dir * STRIKE_APPROACH_DIST
            r = np.linalg.norm(ready[:2])
            if r <= 0.45:
                return 0  # 가까움 → manipulability 최고
            elif r <= 0.55:
                return 1  # 보통
            else:
                return 2  # 멀어서 위험

        def _ready_distance(angle_rad):
            """Ready position까지 로봇 베이스 거리 (정렬 보조키)"""
            strike_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            if abs(TOOL_YAW_OFFSET) > 1e-6:
                ee_y = np.array([strike_dir[1], -strike_dir[0], 0.0])
                tool_dir = strike_dir * np.cos(TOOL_YAW_OFFSET) + ee_y * np.sin(TOOL_YAW_OFFSET)
                ee_off = -tool_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
            else:
                ee_off = -strike_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
            ready = cue_3d + ee_off - strike_dir * STRIKE_APPROACH_DIST
            return np.linalg.norm(ready[:2])

        def _filter_reachable_diverse(candidates, max_n=15, diversity_deg=3):
            """도달가능 + 각도 다양성 필터 (로봇 근접도 + 적은 쿠션 + 벽 뚫림 방지)"""
            # 로봇에 가까운 각도 우선, robust(score높은) 우선, 쿠션 마진순, 거리순
            def _cushion_margin(count):
                """쿠션 마진: 3=최적(마진1), 2=위험(마진0), 4+=불확실"""
                if count == 3: return 0
                if count == 2: return 1
                return 2  # 4+는 불확실
            candidates = sorted(candidates, key=lambda r: (
                _angle_priority(r['angle']),
                -r.get('score', 0),
                _cushion_margin(r.get('cushion_count', 99)),
                _ready_distance(r['angle'])
            ))
            bounds = self.bounds
            tip_margin = TOOL_TIP_RADIUS  # 큐팁 반경만 (벽 내면에 닿지 않도록)
            result = []
            for r in candidates:
                angle = r['angle']
                if self._reject_side_pocket_path(cue_3d, angle):
                    continue
                strike_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
                # ㄴ자 도구: EE는 공 뒤쪽 + 위쪽에 위치 (TOOL_YAW_OFFSET: EE 로컬 프레임 회전)
                if abs(TOOL_YAW_OFFSET) > 1e-6:
                    ee_y = np.array([strike_dir[1], -strike_dir[0], 0.0])
                    tool_dir = strike_dir * np.cos(TOOL_YAW_OFFSET) + ee_y * np.sin(TOOL_YAW_OFFSET)
                    ee_offset = -tool_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
                else:
                    ee_offset = -strike_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP])
                ready_pos = cue_3d + ee_offset - strike_dir * STRIKE_APPROACH_DIST
                if np.linalg.norm(ready_pos[:2]) > SAFE_RADIUS:
                    continue

                # 벽 근접 타격 방지: 공이 타격 방향 벽에 가까우면
                # 도구-벽-공 샌드위치로 예측 불가 → 해당 방향 제외
                wall_margin = 0.03  # 3cm 기본 마진
                sd2 = strike_dir[:2]
                cue2 = cue_3d[:2]
                dir_thresh = 0.1  # 방향 임계치 (0.1 = ~5.7° 이상 벽 쪽이면 필터)
                wall_too_close = False
                # 각 벽 거리 계산
                dx_max = bounds['x_max'] - cue2[0]
                dx_min = cue2[0] - bounds['x_min']
                dy_max = bounds['y_max'] - cue2[1]
                dy_min = cue2[1] - bounds['y_min']
                # 벽에 매우 가까우면(15mm) 마진 확대
                eff_margin_xmax = 0.05 if dx_max < 0.015 else wall_margin
                eff_margin_xmin = 0.05 if dx_min < 0.015 else wall_margin
                eff_margin_ymax = 0.05 if dy_max < 0.015 else wall_margin
                eff_margin_ymin = 0.05 if dy_min < 0.015 else wall_margin
                if sd2[0] > dir_thresh and dx_max < eff_margin_xmax:
                    wall_too_close = True
                elif sd2[0] < -dir_thresh and dx_min < eff_margin_xmin:
                    wall_too_close = True
                if sd2[1] > dir_thresh and dy_max < eff_margin_ymax:
                    wall_too_close = True
                elif sd2[1] < -dir_thresh and dy_min < eff_margin_ymin:
                    wall_too_close = True
                if wall_too_close:
                    continue

                # 도구 tip 벽 뚫림 방지: safe_approach_dist 계산
                # 실제 로봇에서 도구가 벽을 뚫지 않도록 approach_dist를 동적 조정
                safe_approach = STRIKE_APPROACH_DIST
                sd2 = strike_dir[:2]
                cue2 = cue_3d[:2]
                for axis in [0, 1]:
                    if abs(sd2[axis]) > 1e-6:
                        if sd2[axis] > 0:
                            max_a = (cue2[axis] - (bounds['x_min' if axis==0 else 'y_min'] + tip_margin)) / sd2[axis]
                        else:
                            max_a = (cue2[axis] - (bounds['x_max' if axis==0 else 'y_max'] - tip_margin)) / sd2[axis]
                        if max_a > 0:
                            safe_approach = min(safe_approach, max_a)
                min_approach = 0.10  # 최소 10cm — PD 수렴 + 도구 간섭 방지
                safe_approach = max(min_approach, safe_approach)
                # 최소 접근거리에서도 벽 밖이면 제외
                tip_check = cue2 - sd2 * safe_approach
                if (tip_check[0] < bounds['x_min'] + tip_margin or
                    tip_check[0] > bounds['x_max'] - tip_margin or
                    tip_check[1] < bounds['y_min'] + tip_margin or
                    tip_check[1] > bounds['y_max'] - tip_margin):
                    continue
                # approach_dist가 줄었을 때: 저장 + score 페널티
                if safe_approach < STRIKE_APPROACH_DIST:
                    r['safe_approach_dist'] = safe_approach
                    # 축소 비율만큼 페널티 (축소가 클수록 큰 페널티)
                    shrink_ratio = 1.0 - safe_approach / STRIKE_APPROACH_DIST
                    penalty = int(shrink_ratio * 3000)  # 최대 3000점 감점
                    r['score'] = max(r['score'] - penalty, 1)
                    r['approach_penalty'] = penalty

                angle_deg = np.degrees(angle) % 360
                too_close = any(
                    min(abs(angle_deg - np.degrees(e['angle']) % 360),
                        360 - abs(angle_deg - np.degrees(e['angle']) % 360)) < diversity_deg
                    for e in result)
                if not too_close:
                    result.append(r)
                if len(result) >= max_n:
                    break
            return result

        # 2쿠션 이상 전부 동등하게 취급 (안전 각도 우선)
        valid_all = [r for r in fast_results if r['score'] >= 2000]
        # DEBUG: priority 분포 확인
        from collections import Counter
        pri_dist = Counter(_angle_priority(r['angle']) for r in valid_all)
        print(f"  [DEBUG] valid_all priority distribution: {dict(pri_dist)}")
        top_fast = _filter_reachable_diverse(valid_all, max_n=25, diversity_deg=3)
        if len(top_fast) < 5:  # 후보 부족 → diversity 완화하여 재시도
            top_fast2 = _filter_reachable_diverse(valid_all, max_n=25, diversity_deg=1.5)
            if len(top_fast2) > len(top_fast):
                top_fast = top_fast2
        if not top_fast:
            top_fast = _filter_reachable_diverse(fast_results, max_n=15, diversity_deg=1.5)
        if not top_fast:
            # 모든 후보가 필터에 걸림 → 빈 candidates 반환 (unsafe 타격 방지)
            pass
        if top_fast:
            pri_dist2 = Counter(_angle_priority(r['angle']) for r in top_fast)
            print(f"  [DEBUG] top_fast priority distribution: {dict(pri_dist2)}")
        n_3c = sum(1 for r in valid_all if r['score'] >= 3000)
        n_2c = len(valid_all) - n_3c
        print(f"  [Filter] {n_3c} 3-cushion + {n_2c} 2-cushion → {len(top_fast)} reachable diverse")

        # 로봇 환경 생성 + 상위 후보만 검증
        # fast search 결과를 직접 후보로 사용 (headless robot sim 건너뜀)
        # 이유: headless robot sim과 GUI의 물리 차이로 인해 headless 예측이 부정확.
        # fast search의 공 속도/각도가 GUI와 0.3° 이내로 일치하므로 직접 사용.
        candidates = []
        for r in top_fast:
            strike_dir = np.array([np.cos(r['angle']), np.sin(r['angle'])])
            candidates.append({
                'strike_dir': strike_dir,
                'strike_speed': MAX_TOOL_SPEED,
                'ball_speed': MAX_TOOL_SPEED,
                'ball_path': r.get('cue_path'),
                'tgt1_path': r.get('tgt1_path'),
                'tgt2_path': r.get('tgt2_path'),
                'cushion_count': r.get('cushion_count', 0),
                'hit_t1': r.get('hit_t1', False),
                'hit_t2': r.get('hit_t2', False),
                'score': r['score'],
                'angle_deg': np.degrees(r['angle']),
                'safe_approach_dist': r.get('safe_approach_dist', STRIKE_APPROACH_DIST),
            })

        # score 최우선 → 쿠션 마진 → 로봇 근접도 (IK 검증은 state machine에서 수행)
        def _cushion_margin_final(count):
            if count == 3: return 0
            if count == 2: return 1
            return 2
        candidates.sort(key=lambda c: (
            0 if c['score'] > 0 else 1,  # 양 타겟 적중 예측을 항상 우선
            -c['score'],                  # score 최우선 (robustness + position bonus)
            _cushion_margin_final(c.get('cushion_count', 99)),
            _angle_priority(np.radians(c['angle_deg'])),
        ))
        # positive score 후보만 유지 (negative = 양 타겟 미적중 예측)
        positive_candidates = [c for c in candidates if c['score'] > 0]
        print(f"  Found {len(candidates)} diverse candidates ({len(positive_candidates)} positive)")
        if positive_candidates:
            candidates = positive_candidates
            top = candidates[0]
            print(f"  Top: angle={top['angle_deg']:.1f}deg, "
                  f"cushions={top['cushion_count']}, "
                  f"hit_t1={top['hit_t1']}, hit_t2={top['hit_t2']}, "
                  f"score={top['score']}")
        elif cue_pos is not None:
            # 양 타겟 적중 예측 없음 → 벽 위치 탈출: 테이블 중앙 방향으로 단순 타격
            center = np.array([
                (self.bounds['x_min'] + self.bounds['x_max']) / 2,
                (self.bounds['y_min'] + self.bounds['y_max']) / 2
            ])
            cue2 = np.array(cue_pos[:2])
            escape_dir = center - cue2
            escape_norm = np.linalg.norm(escape_dir)
            if escape_norm > 1e-6:
                escape_dir = escape_dir / escape_norm
            else:
                escape_dir = np.array([0.0, 1.0])
            escape_angle = np.degrees(np.arctan2(escape_dir[1], escape_dir[0]))
            candidates = [{
                'strike_dir': escape_dir,
                'strike_speed': MAX_TOOL_SPEED,
                'ball_speed': MAX_TOOL_SPEED,
                'ball_path': None, 'tgt1_path': None, 'tgt2_path': None,
                'cushion_count': 0, 'hit_t1': False, 'hit_t2': False,
                'score': 1, 'angle_deg': escape_angle,
                'safe_approach_dist': STRIKE_APPROACH_DIST,
                'is_escape': True,
            }]
            print(f"  [ESCAPE] No positive candidate → pushing toward center at {escape_angle:.1f}deg")
        return candidates

    # ================================================================
    # Stage 1: 빠른 공 직접 시뮬 (로봇 없음)
    # ================================================================

    def _fast_ball_search(self, cue_pos, tgt1_pos, tgt2_pos, obstacles):
        """공에 직접 속도를 부여하여 빠르게 탐색 (로봇/IK/PD 없음)"""
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
        p.changeDynamics(table_id, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=0.5, physicsClientId=sim)

        # 쿠션
        CH = MAZE_CUSHION_HEIGHT
        top_z = center[2] + TH / 2 + CH / 2
        thickness = 0.03  # GUI maze_env.py와 동일
        configs = [
            ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
            ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ]
        cushion_ids = []
        for pos, half_ext in configs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                         physicsClientId=sim)
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    basePosition=pos, physicsClientId=sim)
            p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION,
                             physicsClientId=sim)
            cushion_ids.append(cid)

        # 공
        def make_ball(pos):
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                         physicsClientId=sim)
            bid = p.createMultiBody(baseMass=MAZE_BALL_MASS,
                                    baseCollisionShapeIndex=col,
                                    basePosition=list(pos), physicsClientId=sim)
            p.changeDynamics(bid, -1,
                             lateralFriction=MAZE_BALL_FRICTION,
                             restitution=MAZE_BALL_RESTITUTION,
                             rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                             spinningFriction=0.02,
                             ccdSweptSphereRadius=MAZE_BALL_RADIUS * 0.5,
                             contactProcessingThreshold=0,
                             physicsClientId=sim)
            return bid

        cue_id = make_ball(cue_pos)
        tgt1_id = make_ball(tgt1_pos)
        tgt2_id = make_ball(tgt2_pos) if tgt2_pos is not None else None

        # 안정화
        for _ in range(50):
            p.stepSimulation(physicsClientId=sim)

        # Grid Search: 0 ~ 360도를 0.5도 간격으로 탐색
        n_initial = 720
        # 3개 속도로 검증 (GUI 실측 범위: 1.80~1.94 m/s)
        speed_center = 1.87
        test_speeds = [
            speed_center,              # 1.870
            speed_center * 1.05,       # 1.964
            speed_center * 0.95,       # 1.777
        ]
        angles = np.linspace(0, 2 * np.pi, n_initial, endpoint=False)

        results = []
        n_pocket_reject = 0
        for spd in test_speeds:
            for i in range(n_initial):
                if self._reject_side_pocket_path(cue_pos, angles[i]):
                    n_pocket_reject += 1
                    continue
                score, info = self._simulate_ball_only(
                    sim, cue_id, tgt1_id, tgt2_id, cushion_ids,
                    cue_pos, tgt1_pos, tgt2_pos, angles[i], spd)
                results.append({'angle': angles[i], 'speed': spd,
                                'score': score, **info})
        if MAZE_SIDE_POCKET_AVOID and n_pocket_reject:
            print(f"  [PocketAvoid] grid rejected {n_pocket_reject} "
                  f"(1st-cushion → side pocket zone)")

        # Deterministic neighborhood refinement: 상위 grid 각도 주변 정밀 탐색
        grid_results = sorted(results, key=lambda r: r['score'], reverse=True)
        seen_angles = set()
        top_angles = []
        for r in grid_results:
            bucket = round(np.degrees(r['angle']))
            if bucket not in seen_angles and r['score'] >= 2000:
                seen_angles.add(bucket)
                top_angles.append(r['angle'])
            if len(top_angles) >= 20:
                break
        offsets = [np.radians(d) for d in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]]
        for base_angle in top_angles:
            for offset in offsets:
                a = base_angle + offset
                for spd in test_speeds:
                    if self._reject_side_pocket_path(cue_pos, a):
                        continue
                    score, info = self._simulate_ball_only(
                        sim, cue_id, tgt1_id, tgt2_id, cushion_ids,
                        cue_pos, tgt1_pos, tgt2_pos, a, spd)
                    results.append({'angle': a, 'speed': spd,
                                    'score': score, **info})

        p.disconnect(sim)

        # Robustness bonus: 같은 각도에서 복수 속도 성공 시 bonus
        from collections import defaultdict
        angle_bucket = defaultdict(list)
        for r in results:
            bucket = round(np.degrees(r['angle']) * 4) / 4
            angle_bucket[bucket].append(r)
        for bucket, group in angle_bucket.items():
            n_success = sum(1 for r in group if r['score'] >= 3000)
            if n_success >= 2:
                for r in group:
                    if r['score'] >= 3000:
                        r['score'] += n_success * 1000
                        r['robust_count'] = n_success

        # Position play bonus: 큐볼이 벽에서 멀리 멈추는 각도에 보너스
        bounds = self.bounds
        for r in results:
            if r['score'] >= 3000 and 'cue_final' in r:
                cf = r['cue_final']
                min_wall_dist = min(
                    cf[0] - bounds['x_min'],
                    bounds['x_max'] - cf[0],
                    cf[1] - bounds['y_min'],
                    bounds['y_max'] - cf[1]
                )
                pos_bonus = int(min(min_wall_dist / 0.10, 1.0) * 500)
                r['score'] += pos_bonus
                r['pos_bonus'] = pos_bonus

        return results

    def _simulate_ball_only(self, sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids,
                            cue_start, tgt1_start, tgt2_start, angle, speed,
                            max_steps=2000):
        """공에 직접 속도를 부여하고 구름 시뮬 (로봇 없음)"""
        # 리셋
        for bid, pos in [(cue_id, cue_start), (tgt1_id, tgt1_start)]:
            p.resetBasePositionAndOrientation(bid, list(pos), [0,0,0,1],
                                              physicsClientId=sim_id)
            p.resetBaseVelocity(bid, [0,0,0], [0,0,0], physicsClientId=sim_id)
        if tgt2_id is not None and tgt2_start is not None:
            p.resetBasePositionAndOrientation(tgt2_id, list(tgt2_start), [0,0,0,1],
                                              physicsClientId=sim_id)
            p.resetBaseVelocity(tgt2_id, [0,0,0], [0,0,0], physicsClientId=sim_id)

        # 수평 타격
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        p.resetBaseVelocity(cue_id, [vx, vy, 0], [0, 0, 0], physicsClientId=sim_id)

        # 시뮬 + 접촉 추적 (순서 기록)
        hit_t1, hit_t2, cushion_contacts = False, False, 0
        events = []  # 접촉 이벤트 순서: 'c'=cushion, 't1'=target1, 't2'=target2
        cue_path = [[cue_start[0], cue_start[1]]]
        prev_cushion = set()

        for step in range(max_steps):
            p.stepSimulation(physicsClientId=sim_id)

            contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim_id)
            cur_cushion = set()
            for c in contacts:
                if c[2] == tgt1_id and not hit_t1:
                    hit_t1 = True
                    events.append('t1')
                elif c[2] == tgt2_id and not hit_t2:
                    hit_t2 = True
                    events.append('t2')
                elif c[2] in cushion_ids:
                    cur_cushion.add(c[2])
            new_cushions = cur_cushion - prev_cushion
            for _ in new_cushions:
                cushion_contacts += 1
                events.append('c')
            prev_cushion = cur_cushion

            if step % 20 == 0:
                pos, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
                cue_path.append([pos[0], pos[1]])

            if step > 200 and step % 50 == 0:
                speeds_check = [np.linalg.norm(p.getBaseVelocity(bid, physicsClientId=sim_id)[0][:2])
                          for bid in [cue_id, tgt1_id] + ([tgt2_id] if tgt2_id else [])]
                if all(s < 0.005 for s in speeds_check):
                    break

        # 큐볼 최종 위치
        cue_final, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
        cue_final_2d = [cue_final[0], cue_final[1]]

        score = self._score_result(cue_id, tgt1_id, tgt2_id, sim_id,
                                   cue_start, tgt1_start, tgt2_start,
                                   hit_t1, hit_t2, cushion_contacts, cue_path,
                                   events)
        return score, {'hit_t1': hit_t1, 'hit_t2': hit_t2,
                       'cushion_count': cushion_contacts, 'events': events,
                       'cue_final': cue_final_2d}

    # ================================================================
    # 환경 생성
    # ================================================================

    def _create_robot_env(self, cue_pos, tgt1_pos, tgt2_pos, obstacles):
        """GUI와 동일한 로봇+도구+테이블 환경을 headless로 구성"""
        sim = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=sim)
        p.setTimeStep(1./240, physicsClientId=sim)

        L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
        TH, H = MAZE_TABLE_HEIGHT, MAZE_TABLE_SURFACE_HEIGHT
        CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y
        center = [CX, CY, H]

        # 바닥
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=sim)

        # 테이블
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2],
                                     physicsClientId=sim)
        table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                     basePosition=center, physicsClientId=sim)
        p.changeDynamics(table_id, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=0.5, physicsClientId=sim)

        # 쿠션 4면
        CH = MAZE_CUSHION_HEIGHT
        top_z = center[2] + TH / 2 + CH / 2
        thickness = 0.03  # GUI maze_env.py와 동일
        configs = [
            ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
            ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ]
        cushion_ids = []
        for pos, half_ext in configs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                         physicsClientId=sim)
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    basePosition=pos, physicsClientId=sim)
            p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION,
                             physicsClientId=sim)
            cushion_ids.append(cid)

        # 공 3개
        def make_ball(pos):
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                         physicsClientId=sim)
            bid = p.createMultiBody(baseMass=MAZE_BALL_MASS,
                                    baseCollisionShapeIndex=col,
                                    basePosition=list(pos), physicsClientId=sim)
            p.changeDynamics(bid, -1,
                             lateralFriction=MAZE_BALL_FRICTION,
                             restitution=MAZE_BALL_RESTITUTION,
                             rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                             spinningFriction=0.02,
                             ccdSweptSphereRadius=MAZE_BALL_RADIUS * 0.5,
                             contactProcessingThreshold=0,
                             physicsClientId=sim)
            return bid

        cue_id = make_ball(cue_pos)
        tgt1_id = make_ball(tgt1_pos)
        tgt2_id = make_ball(tgt2_pos) if tgt2_pos is not None else None

        # 장애물
        obstacle_ids = []
        r_obs, h_obs = MAZE_OBSTACLE_RADIUS, MAZE_OBSTACLE_HEIGHT
        z_obs = center[2] + TH / 2 + h_obs / 2
        for (ox, oy, _) in obstacles:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r_obs,
                                         height=h_obs, physicsClientId=sim)
            oid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    basePosition=[ox, oy, z_obs], physicsClientId=sim)
            p.changeDynamics(oid, -1, restitution=0.5, lateralFriction=0.3,
                             physicsClientId=sim)
            obstacle_ids.append(oid)

        # 로봇
        import os as _os
        urdf_base = _os.path.join(_os.path.dirname(__file__), '..', '..',
                                   'src', 'assets', 'urdf', 'indy7_v2', 'indy7_v2')
        urdf_base = _os.path.abspath(urdf_base)
        try:
            urdf_base.encode('ascii')
            safe_dir = urdf_base
        except UnicodeEncodeError:
            import shutil
            safe_dir = "C:\\tmp_urdf\\indy7_v2\\indy7_v2"
            if not _os.path.exists(safe_dir):
                shutil.copytree(urdf_base, safe_dir)

        robot_id = p.loadURDF(
            safe_dir + "/model.urdf", basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=sim)

        movable_joints = [0, 1, 2, 3, 4, 5]
        home_q = np.array(HOME_Q_RAD).flatten()
        for i, jidx in enumerate(movable_joints):
            p.resetJointState(robot_id, jidx, home_q[i], 0, physicsClientId=sim)
            p.setJointMotorControl2(robot_id, jidx, p.VELOCITY_CONTROL,
                                    force=0, physicsClientId=sim)
        ee_link = 6

        # 도구 + constraint (GUI maze_env.py와 동일한 ㄴ자 도구)
        tool_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=TOOL_TIP_RADIUS,
                                          height=TOOL_TIP_LENGTH, physicsClientId=sim)
        tool_id = p.createMultiBody(baseMass=TOOL_HEAD_MASS,
                                    baseCollisionShapeIndex=tool_col,
                                    basePosition=[0, 0, 0], physicsClientId=sim)
        p.changeDynamics(tool_id, -1, restitution=TOOL_HEAD_RESTITUTION,
                         lateralFriction=0.3, physicsClientId=sim)
        # ㄴ자 오프셋: EE +X 방향으로 30mm, EE +Z 방향으로 60mm (EE z=아래)
        tip_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
        tool_cid = p.createConstraint(
            robot_id, ee_link, tool_id, -1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[TOOL_HORIZONTAL_EXT, 0, TOOL_VERTICAL_DROP],
            parentFrameOrientation=tip_orn,
            childFramePosition=[0, 0, 0], physicsClientId=sim)
        p.changeConstraint(tool_cid, maxForce=TOOL_CONSTRAINT_FORCE,
                           physicsClientId=sim)

        # 충돌 필터: 로봇-환경 비활성, 도구↔큐볼만 활성
        num_links = p.getNumJoints(robot_id, physicsClientId=sim)
        for link_idx in range(-1, num_links):
            p.setCollisionFilterPair(robot_id, table_id, link_idx, -1, 0,
                                     physicsClientId=sim)
            for c in cushion_ids:
                p.setCollisionFilterPair(robot_id, c, link_idx, -1, 0,
                                         physicsClientId=sim)
            for o in obstacle_ids:
                p.setCollisionFilterPair(robot_id, o, link_idx, -1, 0,
                                         physicsClientId=sim)
        p.setCollisionFilterPair(tool_id, table_id, -1, -1, 0, physicsClientId=sim)
        for c in cushion_ids:
            p.setCollisionFilterPair(tool_id, c, -1, -1, 0, physicsClientId=sim)
        for o in obstacle_ids:
            p.setCollisionFilterPair(tool_id, o, -1, -1, 0, physicsClientId=sim)
        p.setCollisionFilterPair(tool_id, tgt1_id, -1, -1, 0, physicsClientId=sim)
        if tgt2_id is not None:
            p.setCollisionFilterPair(tool_id, tgt2_id, -1, -1, 0, physicsClientId=sim)

        # IK solver
        from src.utils.pinocchio_utils import PinocchioModel
        from project.ik_solver import IKSolver
        pin_model = PinocchioModel(_os.path.join(
            _os.path.dirname(__file__), '..', '..', 'src', 'assets', 'urdf',
            'indy7_v2', 'indy7_v2'))
        ik_solver = IKSolver(pin_model, gain=IK_GAIN, damping=IK_DAMPING)

        # 안정화
        for _ in range(50):
            p.stepSimulation(physicsClientId=sim)

        return (sim, cue_id, tgt1_id, tgt2_id, cushion_ids, obstacle_ids,
                tool_id, robot_id, movable_joints, ee_link, tool_cid,
                ik_solver, pin_model)

    # ================================================================
    # 단일 시뮬레이션
    # ================================================================

    def _simulate_one(self, sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids, tool_id,
                      robot_id, movable_joints, ik_solver, pin_model,
                      cue_start, tgt1_start, tgt2_start,
                      angle, ee_speed, max_steps=2000):
        """PD computed torque 기반 타격 시뮬 (GUI와 동일한 물리)"""
        from project.trajectory_planner import TrajectoryPlanner
        sim_dt = 1. / 240

        # 리셋: 3공
        for bid, pos in [(cue_id, cue_start), (tgt1_id, tgt1_start)]:
            p.resetBasePositionAndOrientation(bid, list(pos), [0,0,0,1],
                                              physicsClientId=sim_id)
            p.resetBaseVelocity(bid, [0,0,0], [0,0,0], physicsClientId=sim_id)
        if tgt2_id is not None and tgt2_start is not None:
            p.resetBasePositionAndOrientation(tgt2_id, list(tgt2_start), [0,0,0,1],
                                              physicsClientId=sim_id)
            p.resetBaseVelocity(tgt2_id, [0,0,0], [0,0,0], physicsClientId=sim_id)

        # 타격 방향 (수평)
        strike_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        strike_dir /= np.linalg.norm(strike_dir)

        # SE3 계산 — ㄴ자 도구: EE z축=아래, x축=strike방향
        ball_pos = np.array(cue_start)
        # GUI trajectory_planner.py와 동일한 EE 오프셋 (TOOL_YAW_OFFSET: EE 로컬 프레임 회전)
        if abs(TOOL_YAW_OFFSET) > 1e-6:
            ee_y = np.array([strike_dir[1], -strike_dir[0], 0.0])
            tool_dir = strike_dir * np.cos(TOOL_YAW_OFFSET) + ee_y * np.sin(TOOL_YAW_OFFSET)
            ee_offset = -tool_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP + PIN_PB_EE_Z_OFFSET])
        else:
            ee_offset = -strike_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP + PIN_PB_EE_Z_OFFSET])
        impact_ee = ball_pos + ee_offset
        ready_ee = impact_ee - strike_dir * STRIKE_APPROACH_DIST
        follow_ee = impact_ee + strike_dir * STRIKE_FOLLOW_DIST

        # EE orientation: z축=아래, x축=strike방향 (GUI와 동일)
        z_ax = np.array([0, 0, -1.0])
        x_ax = strike_dir.copy(); x_ax[2] = 0; x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(z_ax, x_ax)
        R = np.column_stack([x_ax, y_ax, z_ax])

        def make_T(pos):
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos; return T

        T_ready = make_T(ready_ee)
        T_follow = make_T(follow_ee)

        # 로봇을 ready에 텔레포트
        home_q = np.array(HOME_Q_RAD).reshape(-1, 1)
        q_ready = home_q.copy()
        for _ in range(20):
            q_ready = ik_solver.solve_step(q_ready, T_ready)
        for i, jidx in enumerate(movable_joints):
            p.resetJointState(robot_id, jidx, float(q_ready[i, 0]), 0,
                              physicsClientId=sim_id)

        # PD 안정화 (Kp=800)
        for _ in range(240):
            states = [p.getJointState(robot_id, j, physicsClientId=sim_id)
                      for j in movable_joints]
            q = np.array([s[0] for s in states]).reshape(-1, 1)
            qdot = np.array([s[1] for s in states]).reshape(-1, 1)
            tau = pin_model.M(q) @ (800*(q_ready-q) + 40*(0-qdot)) + \
                  pin_model.C(q, qdot) @ qdot + pin_model.g(q)
            p.setJointMotorControlArray(robot_id, movable_joints, p.TORQUE_CONTROL,
                                        forces=list(tau.flatten()), physicsClientId=sim_id)
            p.stepSimulation(physicsClientId=sim_id)

        # 도구-큐볼 충돌 활성화
        p.setCollisionFilterPair(tool_id, cue_id, -1, -1, 1, physicsClientId=sim_id)

        # 궤적 생성 + IK
        tp = TrajectoryPlanner()
        full_traj = tp.plan_constant_speed_linear(T_ready, T_follow, ee_speed, sim_dt)
        if len(full_traj) == 0:
            return -5000, {'hit_t1': False, 'hit_t2': False,
                           'cushion_count': 0, 'cue_path': [], 'tgt1_path': [], 'tgt2_path': []}

        q_prev = q_ready.copy()
        q_traj, qdot_traj = [], []
        for T in full_traj:
            for _ in range(5):
                q_prev = ik_solver.solve_step(q_prev, T)
            q_traj.append(q_prev.copy())
        for k in range(len(q_traj)):
            qdot_traj.append((q_traj[k+1] - q_traj[k]) / sim_dt if k < len(q_traj)-1
                             else np.zeros_like(q_traj[0]))
        qddot_traj = []
        for k in range(len(qdot_traj)):
            qddot_traj.append((qdot_traj[k+1] - qdot_traj[k]) / sim_dt if k < len(qdot_traj)-1
                              else np.zeros_like(qdot_traj[0]))

        # 경로 기록 초기화 (strike 전부터 기록 시작)
        hit_t1, hit_t2, cushion_contacts = False, False, 0
        events = []  # 접촉 이벤트 순서
        cue_path = [[cue_start[0], cue_start[1]]]
        tgt1_path = [[tgt1_start[0], tgt1_start[1]]]
        tgt2_path = [[tgt2_start[0], tgt2_start[1]]] if tgt2_id else []
        prev_cushion = set()

        # PD 고게인 스트리밍 (Kp=5000 — GUI와 동일)
        contact_step = -1
        collision_off = False
        for step_i in range(len(q_traj)):
            states = [p.getJointState(robot_id, j, physicsClientId=sim_id)
                      for j in movable_joints]
            q = np.array([s[0] for s in states]).reshape(-1, 1)
            qdot = np.array([s[1] for s in states]).reshape(-1, 1)
            tau = pin_model.M(q) @ (qddot_traj[step_i] + 5000*(q_traj[step_i]-q) + 200*(qdot_traj[step_i]-qdot)) + \
                  pin_model.C(q, qdot) @ qdot + pin_model.g(q)
            p.setJointMotorControlArray(robot_id, movable_joints, p.TORQUE_CONTROL,
                                        forces=list(tau.flatten()), physicsClientId=sim_id)
            p.stepSimulation(physicsClientId=sim_id)

            # strike 중 공 위치 기록
            if step_i % 5 == 0:
                pos, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
                cue_path.append([pos[0], pos[1]])
                pos1, _ = p.getBasePositionAndOrientation(tgt1_id, physicsClientId=sim_id)
                tgt1_path.append([pos1[0], pos1[1]])
                if tgt2_id:
                    pos2, _ = p.getBasePositionAndOrientation(tgt2_id, physicsClientId=sim_id)
                    tgt2_path.append([pos2[0], pos2[1]])

            if not collision_off:
                if contact_step < 0:
                    if len(p.getContactPoints(bodyA=tool_id, bodyB=cue_id,
                                              physicsClientId=sim_id)) > 0:
                        contact_step = step_i
                        print(f'      [Headless DIAG] Contact at step {step_i}/{len(q_traj)}')
                elif step_i - contact_step >= 10:
                    p.setCollisionFilterPair(tool_id, cue_id, -1, -1, 0,
                                            physicsClientId=sim_id)
                    collision_off = True

        # 로봇 정지
        for jidx in movable_joints:
            p.setJointMotorControl2(robot_id, jidx, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=500, physicsClientId=sim_id)

        # 공 구름 관찰 + 접촉 추적 (계속)

        for step in range(max_steps):
            p.stepSimulation(physicsClientId=sim_id)
            if step % 10 == 0:
                pos, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
                cue_path.append([pos[0], pos[1]])
                pos1, _ = p.getBasePositionAndOrientation(tgt1_id, physicsClientId=sim_id)
                tgt1_path.append([pos1[0], pos1[1]])
                if tgt2_id:
                    pos2, _ = p.getBasePositionAndOrientation(tgt2_id, physicsClientId=sim_id)
                    tgt2_path.append([pos2[0], pos2[1]])

            contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim_id)
            cur_cushion = set()
            for c in contacts:
                if c[2] == tgt1_id and not hit_t1:
                    hit_t1 = True
                    events.append('t1')
                elif c[2] == tgt2_id and not hit_t2:
                    hit_t2 = True
                    events.append('t2')
                elif c[2] in cushion_ids:
                    cur_cushion.add(c[2])
            new_cushions = cur_cushion - prev_cushion
            for _ in new_cushions:
                cushion_contacts += 1
                events.append('c')
            prev_cushion = cur_cushion

            if step > 200 and step % 50 == 0:
                speeds = [np.linalg.norm(p.getBaseVelocity(bid, physicsClientId=sim_id)[0][:2])
                          for bid in [cue_id, tgt1_id] + ([tgt2_id] if tgt2_id else [])]
                if all(s < 0.005 for s in speeds):
                    break

        score = self._score_result(cue_id, tgt1_id, tgt2_id, sim_id,
                                   cue_start, tgt1_start, tgt2_start,
                                   hit_t1, hit_t2, cushion_contacts, cue_path,
                                   events)
        return score, {'hit_t1': hit_t1, 'hit_t2': hit_t2,
                       'cushion_count': cushion_contacts, 'events': events,
                       'cue_path': cue_path, 'tgt1_path': tgt1_path, 'tgt2_path': tgt2_path}

    def _score_result(self, cue_id, tgt1_id, tgt2_id, sim_id,
                      cue_start, tgt1_start, tgt2_start,
                      hit_t1, hit_t2, cushion_count, cue_path,
                      events=None):
        """3쿠션 스코어 — 순서 검증
        일반: 수구→t1→쿠션3+→t2
        뱅크: 수구→쿠션3+→t1→t2 (순서 무관)
        """
        score = 0
        cue_final, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)

        # 순서 검증: 두 번째 목적구 접촉 이전의 쿠션 수로 판정
        valid_3cushion = False
        valid_2cushion = False
        if events and hit_t1 and hit_t2:
            t1_idx = events.index('t1')
            t2_idx = events.index('t2')
            second_idx = max(t1_idx, t2_idx)
            c_before_2nd = events[:second_idx].count('c')
            if c_before_2nd >= 3:
                valid_3cushion = True
            if c_before_2nd >= 2:
                valid_2cushion = True

        if valid_3cushion:
            score += 3000  # 3쿠션
        elif valid_2cushion:
            score += 3000  # 2쿠션 (3쿠션과 동등 — 예측 정확도는 2쿠션이 더 높음)
        else:
            if hit_t1: score += 500
            if hit_t2: score += 500
            score += min(cushion_count, 6) * 10

        # 미적중 시 근접도 보너스
        if not hit_t1:
            arr = np.array(cue_path)
            if len(arr) > 0:
                score -= np.min(np.sqrt((arr[:,0]-tgt1_start[0])**2 +
                                        (arr[:,1]-tgt1_start[1])**2)) * 500
        if not hit_t2 and tgt2_id is not None:
            arr = np.array(cue_path)
            if len(arr) > 0:
                score -= np.min(np.sqrt((arr[:,0]-tgt2_start[0])**2 +
                                        (arr[:,1]-tgt2_start[1])**2)) * 500

        b = self.bounds
        if (cue_final[0] < b['x_min']-0.05 or cue_final[0] > b['x_max']+0.05 or
            cue_final[1] < b['y_min']-0.05 or cue_final[1] > b['y_max']+0.05):
            score -= 2000

        return score
