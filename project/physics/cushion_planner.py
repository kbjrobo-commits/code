"""
3쿠션 + 장애물 반사 타격 탐색기
================================
3공 동시 2D 충돌 시뮬레이션 + 어닐링 샘플링 최적화

물리 모델:
- 3공(큐, 황, 적) 동시 2D 위치/속도 추적
- 공-쿠션: 기하 반사 (반발계수 적용)
- 공-공: 2D 탄성 충돌 (운동량+에너지 보존)
- 공-장애물: 원형 장애물 반사 (법선 방향 반사)
- 마찰 감쇠: 매 스텝 속도 감쇠

쓰리쿠션 판정:
- 큐볼이 두 목표공 모두 접촉
- 쿠션 반사 3회 이상 (접촉 사이 or 접촉 전)
"""
import numpy as np
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
    # 대각선 타격 보정: EE의 수평 성분만 공에 전달
    cos_angle = np.cos(np.radians(MAZE_STRIKE_ANGLE_DEG))
    if cos_angle > 1e-6:
        ee_speed = ee_speed / cos_angle
    return ee_speed


class CushionShotPlanner:
    """3공 동시 시뮬레이션 기반 쓰리쿠션 타격 탐색"""

    def __init__(self, table_bounds, ball_radius=MAZE_BALL_RADIUS):
        self.bounds = table_bounds
        self.ball_r = ball_radius

    def plan_shot(self, cue_pos, target_pos, obstacles,
                  ball2_pos=None,
                  n_initial=None, n_refine=None, max_cushions=None):
        """최적 타격 방향/속도 탐색

        Args:
            cue_pos: 큐볼 3D 위치
            target_pos: 목표공1(황) 3D 위치
            obstacles: [(x, y, radius), ...] 장애물 리스트
            ball2_pos: 목표공2(적) 3D 위치 (None이면 무시)

        Returns:
            dict: strike_dir, strike_speed, ball_path, cushion_count, score, ...
        """
        if n_initial is None: n_initial = ANNEAL_N_INITIAL
        if n_refine is None: n_refine = ANNEAL_N_REFINE_ROUNDS
        if max_cushions is None: max_cushions = ANNEAL_MAX_CUSHIONS

        cue_2d = np.array(cue_pos[:2])
        tgt1_2d = np.array(target_pos[:2])
        tgt2_2d = np.array(ball2_pos[:2]) if ball2_pos is not None else None
        obs_2d = [(o[0], o[1], o[2]) for o in obstacles]

        speed_lo, speed_hi = ANNEAL_SPEED_RANGE

        # Phase 1: 광역 탐색
        angles = np.random.uniform(0, 2 * np.pi, n_initial)
        speeds = np.random.uniform(speed_lo, speed_hi, n_initial)
        results = self._evaluate_batch(
            cue_2d, tgt1_2d, tgt2_2d, obs_2d, angles, speeds, max_cushions)

        # Phase 2: 어닐링 정밀화
        for rnd in range(n_refine):
            results.sort(key=lambda r: r['score'], reverse=True)
            n_top = max(int(n_initial * ANNEAL_TOP_RATIO), 5)
            top = results[:n_top]

            sigma_a = np.radians(ANNEAL_SIGMA_ANGLE[min(rnd, len(ANNEAL_SIGMA_ANGLE)-1)])
            sigma_s = ANNEAL_SIGMA_SPEED[min(rnd, len(ANNEAL_SIGMA_SPEED)-1)]

            new_angles = []
            new_speeds = []
            n_per = 5
            for t in top:
                a = np.random.normal(t['angle'], sigma_a, n_per)
                s = np.random.normal(t['speed'], sigma_s, n_per)
                s = np.clip(s, speed_lo, speed_hi)
                new_angles.extend(a)
                new_speeds.extend(s)

            new_results = self._evaluate_batch(
                cue_2d, tgt1_2d, tgt2_2d, obs_2d,
                np.array(new_angles), np.array(new_speeds), max_cushions
            )
            results = results[:n_top] + new_results

        # 최적 선택 — 상위 후보들 리턴 (IK 검증에서 걸러질 수 있으므로)
        results.sort(key=lambda r: r['score'], reverse=True)

        # 각도가 유사한 후보 제거 (다양한 방향 확보)
        top_candidates = []
        for r in results:
            if r['score'] < -1e5:  # 장애물 충돌 후보 제외
                continue
            angle_deg = np.degrees(r['angle']) % 360
            too_close = False
            for existing in top_candidates:
                existing_deg = np.degrees(existing['angle']) % 360
                diff = abs(angle_deg - existing_deg)
                if diff > 180:
                    diff = 360 - diff
                if diff < 15:  # 15도 이내면 유사 후보
                    too_close = True
                    break
            if not too_close:
                top_candidates.append(r)
            if len(top_candidates) >= 10:  # 최대 10개
                break

        if not top_candidates:
            top_candidates = [results[0]]  # fallback

        # 각 후보를 dict으로 변환
        candidates = []
        for r in top_candidates:
            strike_dir_2d = np.array([np.cos(r['angle']), np.sin(r['angle'])])
            ee_speed = ball_speed_to_ee_speed(r['speed'])
            ee_speed = min(ee_speed, MAX_TOOL_SPEED)
            candidates.append({
                'strike_dir': strike_dir_2d,
                'strike_speed': ee_speed,
                'ball_speed': r['speed'],
                'ball_path': r['path'],
                'cushion_count': r['cushions'],
                'hit_t1': r.get('hit_t1', False),
                'hit_t2': r.get('hit_t2', False),
                'score': r['score'],
                'angle_deg': np.degrees(r['angle']),
            })

        return candidates

    def _evaluate_batch(self, cue_2d, tgt1_2d, tgt2_2d, obs_2d,
                        angles, speeds, max_cushions):
        results = []
        for angle, speed in zip(angles, speeds):
            direction = np.array([np.cos(angle), np.sin(angle)])
            sim = self.simulate_3ball(
                cue_2d, tgt1_2d, tgt2_2d, direction, speed,
                obs_2d, max_cushions
            )
            score = self._score_3cushion(sim)
            results.append({
                'angle': angle, 'speed': speed,
                'path': sim['cue_path'],
                'hit_t1': sim['hit_t1'], 'hit_t2': sim['hit_t2'],
                'cushions': sim['cushion_count'], 'score': score
            })
        return results

    # ================================================================
    # 3공 동시 2D 충돌 시뮬레이션
    # ================================================================

    def simulate_3ball(self, cue_start, tgt1_start, tgt2_start,
                       direction, speed, obstacles, max_cushions,
                       dt=0.005, max_steps=1000):
        """3공 동시 2D 물리 시뮬레이션

        Args:
            cue_start: 큐볼 초기 위치 [x,y]
            tgt1_start: 황구 초기 위치 [x,y]
            tgt2_start: 적구 초기 위치 [x,y] (None 가능)
            direction: 큐볼 초기 방향 (단위벡터)
            speed: 큐볼 초기 속력
            obstacles: [(x, y, radius), ...]
            max_cushions: 쿠션 횟수 상한

        Returns:
            dict: cue_path, t1_path, t2_path, hit_t1, hit_t2,
                  cushion_count, obstacle_bounces
        """
        r = self.ball_r
        # PyBullet 반발계수 공식: e_combined = max(e1, e2)
        e_cushion = max(MAZE_BALL_RESTITUTION, MAZE_CUSHION_RESTITUTION)  # 0.85
        e_ball = max(MAZE_BALL_RESTITUTION, MAZE_BALL_RESTITUTION)        # 0.85
        # 마찰 감쇠: rollingFriction + lateralFriction 근사
        decay = 1 - (ANNEAL_ROLLING_FRICTION + MAZE_BALL_FRICTION * 0.05) * dt

        # 테이블 경계 (공 반지름 보정)
        xmin = self.bounds['x_min'] + r
        xmax = self.bounds['x_max'] - r
        ymin = self.bounds['y_min'] + r
        ymax = self.bounds['y_max'] - r

        # 공 상태: [px, py, vx, vy]
        balls = []
        # 0: 큐볼
        balls.append([float(cue_start[0]), float(cue_start[1]),
                      float(direction[0] * speed), float(direction[1] * speed)])
        # 1: 황구 (target1)
        balls.append([float(tgt1_start[0]), float(tgt1_start[1]), 0.0, 0.0])
        # 2: 적구 (target2)
        if tgt2_start is not None:
            balls.append([float(tgt2_start[0]), float(tgt2_start[1]), 0.0, 0.0])

        n_balls = len(balls)
        contact_r = 2 * r  # 공-공 접촉 거리
        contact_r2 = contact_r ** 2

        # 장애물 사전 변환
        obs_list = [(o[0], o[1], o[2] + r) for o in obstacles]  # 반지름에 공 반지름 추가

        # 추적 변수
        cushion_count = 0
        obstacle_bounces = 0
        hit_t1 = False
        hit_t2 = False
        cue_path = [(balls[0][0], balls[0][1])]
        t1_path = [(balls[1][0], balls[1][1])]
        t2_path = [(balls[2][0], balls[2][1])] if n_balls > 2 else []

        # 충돌 쿨다운 (같은 쌍이 연속 충돌 방지)
        cooldown = {}

        for step in range(max_steps):
            # --- 전체 속도 체크: 모든 공이 멈추면 종료 ---
            total_v2 = 0
            for b in balls:
                total_v2 += b[2] ** 2 + b[3] ** 2
            if total_v2 < 1e-4:  # ~0.01 m/s 전체
                break

            # --- 속도 감쇠 (마찰) ---
            for b in balls:
                b[2] *= decay
                b[3] *= decay

            # --- 위치 업데이트 ---
            for b in balls:
                b[0] += b[2] * dt
                b[1] += b[3] * dt

            # --- 쿠션 반사 (각 공) ---
            for bi, b in enumerate(balls):
                reflected = False
                if b[0] <= xmin:
                    b[0] = xmin
                    b[2] = abs(b[2]) * e_cushion
                    reflected = True
                elif b[0] >= xmax:
                    b[0] = xmax
                    b[2] = -abs(b[2]) * e_cushion
                    reflected = True
                if b[1] <= ymin:
                    b[1] = ymin
                    b[3] = abs(b[3]) * e_cushion
                    reflected = True
                elif b[1] >= ymax:
                    b[1] = ymax
                    b[3] = -abs(b[3]) * e_cushion
                    reflected = True

                if reflected and bi == 0:  # 큐볼 쿠션만 카운트
                    cushion_count += 1
                    if cushion_count > max_cushions + 2:
                        break

            # --- 공-공 충돌 ---
            for i in range(n_balls):
                for j in range(i + 1, n_balls):
                    pair_key = (i, j)
                    # 쿨다운 체크
                    if pair_key in cooldown and cooldown[pair_key] > 0:
                        cooldown[pair_key] -= 1
                        continue

                    dx = balls[i][0] - balls[j][0]
                    dy = balls[i][1] - balls[j][1]
                    dist2 = dx * dx + dy * dy

                    if dist2 < contact_r2 and dist2 > 1e-10:
                        dist = np.sqrt(dist2)
                        # 법선 벡터 (i→j)
                        nx = dx / dist
                        ny = dy / dist

                        # 상대 속도의 법선 성분
                        dvx = balls[i][2] - balls[j][2]
                        dvy = balls[i][3] - balls[j][3]
                        dvn = dvx * nx + dvy * ny

                        # 접근 중일 때만 충돌 (이미 분리 중이면 무시)
                        if dvn > 0:
                            continue

                        # 동일 질량 2D 탄성 충돌:
                        # 법선 방향 속도 성분만 교환
                        impulse = -(1 + e_ball) * dvn / 2.0
                        balls[i][2] += impulse * nx
                        balls[i][3] += impulse * ny
                        balls[j][2] -= impulse * nx
                        balls[j][3] -= impulse * ny

                        # 겹침 해제
                        overlap = contact_r - dist
                        if overlap > 0:
                            balls[i][0] += nx * overlap * 0.5
                            balls[i][1] += ny * overlap * 0.5
                            balls[j][0] -= nx * overlap * 0.5
                            balls[j][1] -= ny * overlap * 0.5

                        # 쿨다운 설정 (5 스텝)
                        cooldown[pair_key] = 5

                        # 접촉 기록
                        if i == 0 and j == 1:
                            hit_t1 = True
                        elif i == 0 and j == 2:
                            hit_t2 = True

            # --- 장애물 충돌 (반사) ---
            for bi, b in enumerate(balls):
                for ox, oy, combined_r in obs_list:
                    dx = b[0] - ox
                    dy = b[1] - oy
                    dist2 = dx * dx + dy * dy
                    cr2 = combined_r * combined_r

                    if dist2 < cr2 and dist2 > 1e-10:
                        dist = np.sqrt(dist2)
                        # 법선 (장애물 중심 → 공 중심)
                        nx = dx / dist
                        ny = dy / dist

                        # 법선 방향 속도 성분
                        vn = b[2] * nx + b[3] * ny
                        if vn < 0:  # 접근 중
                            # 반사: 법선 방향 속도 반전
                            b[2] -= 2 * vn * nx * e_cushion
                            b[3] -= 2 * vn * ny * e_cushion

                            # 겹침 해제
                            overlap = combined_r - dist
                            if overlap > 0:
                                b[0] += nx * overlap
                                b[1] += ny * overlap

                            obstacle_bounces += 1

            # --- 경로 기록 (매 5스텝) ---
            if step % 5 == 0:
                cue_path.append((balls[0][0], balls[0][1]))
                t1_path.append((balls[1][0], balls[1][1]))
                if n_balls > 2:
                    t2_path.append((balls[2][0], balls[2][1]))

            # 쿨다운 감소 (이미 위에서 처리)

        return {
            'cue_path': cue_path,
            't1_path': t1_path,
            't2_path': t2_path,
            'hit_t1': hit_t1,
            'hit_t2': hit_t2,
            'cushion_count': cushion_count,
            'obstacle_bounces': obstacle_bounces,
        }

    def _score_3cushion(self, sim):
        """쓰리쿠션 스코어링

        득점 조건: 큐볼이 두 목표공 모두 접촉 + 쿠션 3회 이상
        """
        hit_t1 = sim['hit_t1']
        hit_t2 = sim['hit_t2']
        cushions = sim['cushion_count']

        # 쓰리쿠션 완전 득점
        if hit_t1 and hit_t2 and cushions >= 3:
            return 3000 + cushions * 10  # 최고 점수

        # 두 공 접촉했지만 쿠션 부족
        if hit_t1 and hit_t2:
            return 2000 + cushions * 30  # 쿠션 늘리면 좋음

        # 한 공만 접촉
        score = 0
        if hit_t1:
            score += 1000
        if hit_t2:
            score += 1000

        # 쿠션 보너스
        score += min(cushions, 3) * 30

        # 미접촉 공까지의 최소 거리 페널티
        cue_path = np.array(sim['cue_path'])
        if not hit_t1 and len(sim['t1_path']) > 0:
            t1_last = np.array(sim['t1_path'][-1])
            # 큐볼 궤적에서 t1까지 최소 거리
            dists = np.sqrt((cue_path[:, 0] - t1_last[0])**2 +
                           (cue_path[:, 1] - t1_last[1])**2)
            score -= dists.min() * 500

        if not hit_t2 and len(sim['t2_path']) > 0:
            t2_last = np.array(sim['t2_path'][-1])
            dists = np.sqrt((cue_path[:, 0] - t2_last[0])**2 +
                           (cue_path[:, 1] - t2_last[1])**2)
            score -= dists.min() * 500

        return score
