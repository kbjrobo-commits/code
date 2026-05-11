"""
3쿠션 타격 탐색기 — Headless PyBullet 3D 시뮬레이션
====================================================
미니골프 Grid Search와 동일한 방식:
  별도 headless PyBullet에 동일한 테이블/쿠션/장애물/3공을 구성하고,
  큐볼에 수평 초기 속도를 부여하여 직접 굴려본다.
  → PyBullet이 계획에도 실행에도 쓰이므로 모델 gap이 0.

쓰리쿠션 판정:
  - 큐볼이 두 목표공 모두 접촉
  - 쿠션 반사 3회 이상
"""
import numpy as np
import pybullet as p
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project.config import *


def ball_speed_to_ee_speed(v_ball, m_tool=TOOL_HEAD_MASS,
                           m_ball=MAZE_BALL_MASS,
                           e_tool=TOOL_HEAD_RESTITUTION,
                           e_ball=MAZE_BALL_RESTITUTION):
    """운동량 보존으로 필요한 EE 속도 역산 (대각선 타격 보정 포함)"""
    e = np.sqrt(e_tool * e_ball)
    ratio = (1 + e) * m_tool / (m_tool + m_ball)
    if ratio < 1e-6:
        return v_ball
    ee_speed = v_ball / ratio
    cos_angle = np.cos(np.radians(MAZE_STRIKE_ANGLE_DEG))
    if cos_angle > 1e-6:
        ee_speed = ee_speed / cos_angle
    return ee_speed


class CushionShotPlanner:
    """Headless PyBullet 3D 시뮬레이션 기반 쓰리쿠션 타격 탐색"""

    def __init__(self, table_bounds, ball_radius=MAZE_BALL_RADIUS):
        self.bounds = table_bounds
        self.ball_r = ball_radius

    def plan_shot(self, cue_pos, target_pos, obstacles,
                  ball2_pos=None):
        """어닐링 탐색: headless PyBullet에서 직접 3공을 굴려보고 최적 타격 선택

        Returns:
            candidates: list of dicts (상위 다양 후보들)
        """
        cue_3d = np.array(cue_pos).flatten()
        tgt1_3d = np.array(target_pos).flatten()
        tgt2_3d = np.array(ball2_pos).flatten() if ball2_pos is not None else None

        # === Headless PyBullet 환경 구성 ===
        sim = self._create_headless_env(cue_3d, tgt1_3d, tgt2_3d, obstacles)
        sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids, obstacle_ids = sim

        # === 어닐링 탐색 ===
        n_initial = ANNEAL_N_INITIAL
        speed_lo, speed_hi = ANNEAL_SPEED_RANGE

        # Phase 1: 광역 샘플링
        angles = np.random.uniform(0, 2 * np.pi, n_initial)
        speeds = np.random.uniform(speed_lo, speed_hi, n_initial)

        results = []
        for i in range(n_initial):
            score, info = self._simulate_one(
                sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids,
                cue_3d, tgt1_3d, tgt2_3d,
                angles[i], speeds[i]
            )
            results.append({
                'angle': angles[i], 'speed': speeds[i],
                'score': score, **info
            })

        # Phase 2: 정밀화 라운드
        for rnd in range(ANNEAL_N_REFINE_ROUNDS):
            results.sort(key=lambda r: r['score'], reverse=True)
            n_top = max(int(len(results) * ANNEAL_TOP_RATIO), 5)
            top = results[:n_top]

            sigma_a = np.radians(ANNEAL_SIGMA_ANGLE[rnd])
            sigma_s = ANNEAL_SIGMA_SPEED[rnd]

            new_angles = []
            new_speeds = []
            for t in top:
                for _ in range(n_initial // n_top):
                    a = t['angle'] + np.random.normal(0, sigma_a)
                    s = np.clip(t['speed'] + np.random.normal(0, sigma_s),
                                speed_lo, speed_hi)
                    new_angles.append(a)
                    new_speeds.append(s)

            for i in range(len(new_angles)):
                score, info = self._simulate_one(
                    sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids,
                    cue_3d, tgt1_3d, tgt2_3d,
                    new_angles[i], new_speeds[i]
                )
                results.append({
                    'angle': new_angles[i], 'speed': new_speeds[i],
                    'score': score, **info
                })

        p.disconnect(sim_id)

        # === 상위 다양 후보 선택 ===
        results.sort(key=lambda r: r['score'], reverse=True)

        top_candidates = []
        for r in results:
            angle_deg = np.degrees(r['angle']) % 360
            too_close = False
            for existing in top_candidates:
                existing_deg = np.degrees(existing['angle']) % 360
                diff = abs(angle_deg - existing_deg)
                if diff > 180:
                    diff = 360 - diff
                if diff < 15:
                    too_close = True
                    break
            if not too_close:
                top_candidates.append(r)
            if len(top_candidates) >= 10:
                break

        if not top_candidates:
            top_candidates = [results[0]]

        # dict 변환
        candidates = []
        for r in top_candidates:
            strike_dir_2d = np.array([np.cos(r['angle']), np.sin(r['angle'])])
            ee_speed = ball_speed_to_ee_speed(r['speed'])
            ee_speed = min(ee_speed, MAX_TOOL_SPEED)
            candidates.append({
                'strike_dir': strike_dir_2d,
                'strike_speed': ee_speed,
                'ball_speed': r['speed'],
                'ball_path': r.get('cue_path'),
                'tgt1_path': r.get('tgt1_path'),
                'tgt2_path': r.get('tgt2_path'),
                'cushion_count': r.get('cushion_count', 0),
                'hit_t1': r.get('hit_t1', False),
                'hit_t2': r.get('hit_t2', False),
                'score': r['score'],
                'angle_deg': np.degrees(r['angle']),
            })

        return candidates

    # ================================================================
    # Headless PyBullet 환경 구성
    # ================================================================

    def _create_headless_env(self, cue_pos, tgt1_pos, tgt2_pos, obstacles):
        """GUI 환경과 동일한 테이블/쿠션/공/장애물을 headless로 구성"""
        sim = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=sim)
        p.setTimeStep(1./240, physicsClientId=sim)

        L = MAZE_TABLE_LENGTH
        W = MAZE_TABLE_WIDTH
        TH = MAZE_TABLE_HEIGHT
        H = MAZE_TABLE_SURFACE_HEIGHT
        CY = MAZE_TABLE_CENTER_Y
        center = [0.5, CY, H]

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
        thickness = 0.04
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
                                    basePosition=list(pos),
                                    physicsClientId=sim)
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
        r_obs = MAZE_OBSTACLE_RADIUS
        h_obs = MAZE_OBSTACLE_HEIGHT
        z_obs = center[2] + TH / 2 + h_obs / 2
        for (ox, oy, _) in obstacles:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r_obs,
                                         height=h_obs, physicsClientId=sim)
            oid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    basePosition=[ox, oy, z_obs],
                                    physicsClientId=sim)
            p.changeDynamics(oid, -1, restitution=0.5, lateralFriction=0.3,
                             physicsClientId=sim)
            obstacle_ids.append(oid)

        return sim, cue_id, tgt1_id, tgt2_id, cushion_ids, obstacle_ids

    # ================================================================
    # 단일 시뮬레이션 실행
    # ================================================================

    def _simulate_one(self, sim_id, cue_id, tgt1_id, tgt2_id, cushion_ids,
                      cue_start, tgt1_start, tgt2_start,
                      angle, speed, max_steps=2000):
        """하나의 (angle, speed) 조합을 headless PyBullet에서 시뮬

        Returns:
            score: float (높을수록 좋음)
            info: dict {hit_t1, hit_t2, cushion_count, cue_path}
        """
        # 공 위치/속도 리셋
        p.resetBasePositionAndOrientation(
            cue_id, list(cue_start), [0,0,0,1], physicsClientId=sim_id)
        p.resetBaseVelocity(cue_id, [0,0,0], [0,0,0], physicsClientId=sim_id)

        p.resetBasePositionAndOrientation(
            tgt1_id, list(tgt1_start), [0,0,0,1], physicsClientId=sim_id)
        p.resetBaseVelocity(tgt1_id, [0,0,0], [0,0,0], physicsClientId=sim_id)

        if tgt2_id is not None and tgt2_start is not None:
            p.resetBasePositionAndOrientation(
                tgt2_id, list(tgt2_start), [0,0,0,1], physicsClientId=sim_id)
            p.resetBaseVelocity(tgt2_id, [0,0,0], [0,0,0], physicsClientId=sim_id)

        # 큐볼에 수평 초기 속도 부여
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        p.resetBaseVelocity(cue_id, [vx, vy, 0], [0,0,0], physicsClientId=sim_id)

        # 시뮬 실행 + 접촉 감지
        hit_t1 = False
        hit_t2 = False
        cushion_set = set()  # 어떤 쿠션에 몇 번 닿았는지 (중복 카운트)
        cushion_contacts = 0
        cue_path = []
        tgt1_path = []
        tgt2_path = []

        # 이전 프레임 접촉 상태 (중복 카운트 방지)
        prev_cushion_contact = set()

        for step in range(max_steps):
            p.stepSimulation(physicsClientId=sim_id)

            # 3공 위치 기록 (매 10스텝)
            if step % 10 == 0:
                pos, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
                cue_path.append([pos[0], pos[1]])
                pos1, _ = p.getBasePositionAndOrientation(tgt1_id, physicsClientId=sim_id)
                tgt1_path.append([pos1[0], pos1[1]])
                if tgt2_id is not None:
                    pos2, _ = p.getBasePositionAndOrientation(tgt2_id, physicsClientId=sim_id)
                    tgt2_path.append([pos2[0], pos2[1]])

            # 접촉 판정
            contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim_id)
            current_cushion_contact = set()
            for c in contacts:
                other_id = c[2]  # bodyB
                if other_id == tgt1_id:
                    hit_t1 = True
                elif other_id == tgt2_id:
                    hit_t2 = True
                elif other_id in cushion_ids:
                    current_cushion_contact.add(other_id)

            # 새로운 쿠션 접촉만 카운트 (이전 프레임에 없던 것)
            new_contacts = current_cushion_contact - prev_cushion_contact
            cushion_contacts += len(new_contacts)
            prev_cushion_contact = current_cushion_contact

            # 조기 종료: 모든 공이 멈춤
            if step > 200 and step % 50 == 0:
                v_cue, _ = p.getBaseVelocity(cue_id, physicsClientId=sim_id)
                v_t1, _ = p.getBaseVelocity(tgt1_id, physicsClientId=sim_id)
                speed_cue = np.linalg.norm(v_cue[:2])
                speed_t1 = np.linalg.norm(v_t1[:2])
                speed_t2 = 0
                if tgt2_id is not None:
                    v_t2, _ = p.getBaseVelocity(tgt2_id, physicsClientId=sim_id)
                    speed_t2 = np.linalg.norm(v_t2[:2])
                if speed_cue < 0.005 and speed_t1 < 0.005 and speed_t2 < 0.005:
                    break

        # 스코어링
        score = self._score_result(
            cue_id, tgt1_id, tgt2_id, sim_id,
            cue_start, tgt1_start, tgt2_start,
            hit_t1, hit_t2, cushion_contacts, cue_path
        )

        return score, {
            'hit_t1': hit_t1, 'hit_t2': hit_t2,
            'cushion_count': cushion_contacts,
            'cue_path': cue_path,
            'tgt1_path': tgt1_path,
            'tgt2_path': tgt2_path,
        }

    def _score_result(self, cue_id, tgt1_id, tgt2_id, sim_id,
                      cue_start, tgt1_start, tgt2_start,
                      hit_t1, hit_t2, cushion_count, cue_path):
        """3쿠션 스코어: hit_t1 + hit_t2 + 쿠션≥3 = 성공"""
        score = 0

        # 큐볼 최종 위치
        cue_final, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim_id)
        tgt1_final, _ = p.getBasePositionAndOrientation(tgt1_id, physicsClientId=sim_id)

        # 3쿠션 조건
        if hit_t1:
            score += 1000
        if hit_t2:
            score += 1000
        if cushion_count >= 3:
            score += 1000
        score += min(cushion_count, 6) * 10

        # 미접촉 시: 최근접 거리로 부분 점수
        if not hit_t1:
            cue_arr = np.array(cue_path)
            if len(cue_arr) > 0:
                dists = np.sqrt((cue_arr[:,0] - tgt1_start[0])**2 +
                               (cue_arr[:,1] - tgt1_start[1])**2)
                score -= dists.min() * 500

        if not hit_t2 and tgt2_id is not None:
            cue_arr = np.array(cue_path)
            if len(cue_arr) > 0:
                dists = np.sqrt((cue_arr[:,0] - tgt2_start[0])**2 +
                               (cue_arr[:,1] - tgt2_start[1])**2)
                score -= dists.min() * 500

        # 큐볼이 테이블 밖으로 나가면 큰 패널티
        b = self.bounds
        cx, cy = cue_final[0], cue_final[1]
        if cx < b['x_min'] - 0.05 or cx > b['x_max'] + 0.05 or \
           cy < b['y_min'] - 0.05 or cy > b['y_max'] + 0.05:
            score -= 2000

        return score
