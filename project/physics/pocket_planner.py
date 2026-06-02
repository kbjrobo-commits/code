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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from project.config import *


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
        p.changeDynamics(table_id, -1, lateralFriction=POCKET_DEMO_FRICTION,
                         rollingFriction=POCKET_DEMO_ROLLING_FRICTION,
                         restitution=0.5, physicsClientId=sim)

        # 포켓 갭 쿠션 (maze_env._create_cushions_with_pockets와 동일)
        CH = MAZE_CUSHION_HEIGHT
        top_z = center[2] + TH / 2 + CH / 2
        thickness = 0.03
        gap = POCKET_RADIUS * 2

        x_min, x_max = CX - L / 2, CX + L / 2
        y_min, y_max = CY - W / 2, CY + W / 2

        cushion_ids = []

        def _add(pos, half_ext):
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                       physicsClientId=sim)
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                    basePosition=pos, physicsClientId=sim)
            p.changeDynamics(cid, -1, restitution=POCKET_DEMO_CUSHION_RESTITUTION,
                             physicsClientId=sim)
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
        p.changeDynamics(bid, -1,
                         lateralFriction=POCKET_DEMO_FRICTION,
                         restitution=POCKET_DEMO_BALL_RESTITUTION,
                         rollingFriction=POCKET_DEMO_ROLLING_FRICTION,
                         spinningFriction=0.01,
                         ccdSweptSphereRadius=MAZE_BALL_RADIUS * 0.5,
                         contactProcessingThreshold=0,
                         physicsClientId=sim)
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

        # 각 이상 각도 주변 ±15도를 0.5도 간격으로 탐색 + 글로벌 보충
        search_angles = set()
        for base in ideal_angles:
            for offset_deg in np.arange(-15, 15.5, 0.5):
                search_angles.add(base + np.radians(offset_deg))
        # 글로벌 보충: 전체를 2도 간격 (놓친 각도 커버)
        for a in np.linspace(0, 2 * np.pi, 180, endpoint=False):
            search_angles.add(a)
        search_angles = sorted(search_angles)

        speed_center = 1.87
        test_speeds = [speed_center, speed_center * 1.05, speed_center * 0.95]

        print(f"  [PocketSearch] {len(ideal_angles)} ideal angles, "
              f"{len(search_angles)} total search angles × {len(test_speeds)} speeds")

        results = []
        for spd in test_speeds:
            for angle in search_angles:
                score, info = self._simulate_pocket_shot(
                    sim, cue_id, target_id, other_ids, cushion_ids,
                    pocket_positions, cue_3d, target_3d, others_3d,
                    angle, spd, surface_z)
                results.append({
                    'angle': angle, 'speed': spd, 'score': score, **info
                })

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

        # Filter + format
        return self._format_candidates(results, cue_3d)

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

        # 큐볼 속도 부여 (순수 구름 조건)
        self._set_rolling_velocity(sim, cue_id, speed, angle)

        # 시뮬 + 접촉 추적
        hit_target = False
        illegal_contact = False
        cue_scratched = False
        target_min_pocket_dist = float('inf')  # 경로 상 최소 포켓 거리
        target_closest_pocket_idx = -1
        cue_path = [[cue_start[0], cue_start[1]]]
        target_path = [[target_start[0], target_start[1]]]

        for step in range(max_steps):
            p.stepSimulation(physicsClientId=sim)

            # 큐볼 접촉 체크
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

            # 목적구 접촉 체크 (목적구가 다른 공에 부딪히는 것도 금지)
            if hit_target:
                target_contacts = p.getContactPoints(bodyA=target_id, physicsClientId=sim)
                for c in target_contacts:
                    if c[2] in other_ids:
                        illegal_contact = True
                        break
                if illegal_contact:
                    break

            # 큐볼 포켓 스크래치 체크
            cue_pos_now, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)
            for pp in pocket_positions:
                if np.linalg.norm(np.array(cue_pos_now[:2]) - pp[:2]) < POCKET_RADIUS:
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
                if d < POCKET_RADIUS:
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
                if np.linalg.norm(np.array(cue_final[:2]) - pp[:2]) < POCKET_RADIUS:
                    cue_scratched = True
                    break

        # XY 기반 포켓 판정 (테이블에 물리적 구멍 없으므로 z 체크만으로 불충분)
        # 최종 위치 또는 경로 상 최근접 거리 중 하나라도 POCKET_RADIUS 이내면 성공
        if not target_pocketed and hit_target:
            # 최종 위치 체크
            for i, pp in enumerate(pocket_positions):
                dist_to_pocket = np.linalg.norm(
                    np.array(target_final[:2]) - pp[:2])
                if dist_to_pocket < POCKET_RADIUS:
                    target_pocketed = True
                    pocket_idx = i
                    break
        if not target_pocketed and hit_target:
            # 경로 상 최근접 체크 (시뮬 중 포켓을 스쳐 지나간 경우)
            if target_min_pocket_dist < POCKET_RADIUS:
                target_pocketed = True
                pocket_idx = target_closest_pocket_idx

        score = 0
        if illegal_contact:
            score = -10000
        elif cue_scratched:
            score = -5000
        elif target_pocketed:
            score = 100000  # 포켓 성공 — 항상 최고 우선
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
            'cue_scratched': cue_scratched,
            'cue_final': [cue_final[0], cue_final[1]],
            'target_final': [target_final[0], target_final[1]],
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
                if np.linalg.norm(np.array(cue_pos_now[:2]) - pp[:2]) < POCKET_RADIUS:
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

        # 양수 score만 필터
        positive = [r for r in results if r['score'] > 0]
        if not positive:
            positive = sorted(results, key=lambda r: r['score'], reverse=True)[:5]

        # 도달가능성 + 다양성 필터
        positive.sort(key=lambda r: r['score'], reverse=True)
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

            # 벽 근접 타격 방지 (cushion_planner에서 가져온 로직)
            sd2 = strike_dir[:2]
            cue2 = cue_pos[:2]
            dir_thresh = 0.1
            wall_margin = 0.03
            wall_too_close = False
            dx_max = self.bounds['x_max'] - cue2[0]
            dx_min = cue2[0] - self.bounds['x_min']
            dy_max = self.bounds['y_max'] - cue2[1]
            dy_min = cue2[1] - self.bounds['y_min']
            if sd2[0] > dir_thresh and dx_max < wall_margin:
                wall_too_close = True
            elif sd2[0] < -dir_thresh and dx_min < wall_margin:
                wall_too_close = True
            if sd2[1] > dir_thresh and dy_max < wall_margin:
                wall_too_close = True
            elif sd2[1] < -dir_thresh and dy_min < wall_margin:
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
            safe_approach = max(0.10, safe_approach)

            # 다양성: 3도 이내 중복 방지
            bucket = round(angle_deg)
            too_close = any(
                min(abs(angle_deg - s), 360 - abs(angle_deg - s)) < 3
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
                'precision_distance': r.get('precision_distance'),
                'illegal_contact': r.get('illegal_contact', False),
                'cue_scratched': r.get('cue_scratched', False),
                'cue_final': r.get('cue_final'),
                'cushion_count': 0,
                'hit_t1': r.get('hit_target', False),
                'hit_t2': False,
                'events': [],
            })

            if len(candidates) >= 25:
                break

        # 최종 정렬
        candidates.sort(key=lambda c: -c['score'])

        n_success = sum(1 for c in candidates
                        if c.get('target_pocketed') or
                        (c.get('precision_distance') is not None and
                         c['precision_distance'] <= PRECISION_STOP_TOLERANCE))
        print(f"  [PocketPlanner] {len(candidates)} candidates, {n_success} successes")
        if candidates:
            top = candidates[0]
            print(f"  Top: angle={top['angle_deg']:.1f}deg, score={top['score']}")

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

