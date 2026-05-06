"""
3쿠션 + 장애물 회피 타격 탐색기
================================
어닐링 샘플링 기반 다중 반사 궤적 최적화
기획서 3.1절: DIAL-MPC 탐색
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
    """운동량 보존으로 필요한 EE 속도 역산"""
    e = np.sqrt(e_tool * e_ball)
    ratio = (1 + e) * m_tool / (m_tool + m_ball)
    if ratio < 1e-6:
        return v_ball
    return v_ball / ratio


class CushionShotPlanner:
    """어닐링 기반 다중 쿠션 반사 + 장애물 회피 최적 타격 탐색"""

    def __init__(self, table_bounds, ball_radius=MAZE_BALL_RADIUS):
        """
        Args:
            table_bounds: dict with x_min, x_max, y_min, y_max
            ball_radius: 공 반지름
        """
        self.bounds = table_bounds
        self.ball_r = ball_radius

    def plan_shot(self, cue_pos, target_pos, obstacles,
                  n_initial=None, n_refine=None, max_cushions=None):
        """최적 타격 방향/속도 탐색

        Args:
            cue_pos: 큐볼 3D 위치
            target_pos: 목표공 3D 위치
            obstacles: [(x, y, radius), ...] 장애물 리스트

        Returns:
            dict: strike_dir, strike_speed, ball_path, cushion_count, score
        """
        if n_initial is None: n_initial = ANNEAL_N_INITIAL
        if n_refine is None: n_refine = ANNEAL_N_REFINE_ROUNDS
        if max_cushions is None: max_cushions = ANNEAL_MAX_CUSHIONS

        cue_2d = np.array(cue_pos[:2])
        tgt_2d = np.array(target_pos[:2])
        obs_2d = [(o[0], o[1], o[2]) for o in obstacles]

        speed_lo, speed_hi = ANNEAL_SPEED_RANGE

        # Phase 1: 광역 탐색
        angles = np.random.uniform(0, 2 * np.pi, n_initial)
        speeds = np.random.uniform(speed_lo, speed_hi, n_initial)
        results = self._evaluate_batch(cue_2d, tgt_2d, obs_2d, angles, speeds, max_cushions)

        # Phase 2: 어닐링 정밀화
        for rnd in range(n_refine):
            n_top = max(10, int(len(results) * ANNEAL_TOP_RATIO))
            results.sort(key=lambda r: r['score'], reverse=True)
            top = results[:n_top]

            sigma_a = np.radians(ANNEAL_SIGMA_ANGLE[min(rnd, len(ANNEAL_SIGMA_ANGLE)-1)])
            sigma_s = ANNEAL_SIGMA_SPEED[min(rnd, len(ANNEAL_SIGMA_SPEED)-1)]

            new_angles = []
            new_speeds = []
            n_per = max(10, n_initial // (5 * (rnd + 1)))
            for t in top:
                a = np.random.normal(t['angle'], sigma_a, n_per)
                s = np.random.normal(t['speed'], sigma_s, n_per)
                s = np.clip(s, speed_lo, speed_hi)
                new_angles.extend(a)
                new_speeds.extend(s)

            new_results = self._evaluate_batch(
                cue_2d, tgt_2d, obs_2d,
                np.array(new_angles), np.array(new_speeds), max_cushions
            )
            results = results[:n_top] + new_results

        # 최적 선택
        results.sort(key=lambda r: r['score'], reverse=True)
        best = results[0]

        # 2D 타격 방향 → EE 속도 역산
        strike_dir_2d = np.array([np.cos(best['angle']), np.sin(best['angle'])])
        ee_speed = ball_speed_to_ee_speed(best['speed'])
        ee_speed = min(ee_speed, MAX_TOOL_SPEED)

        return {
            'strike_dir': strike_dir_2d,
            'strike_speed': ee_speed,
            'ball_speed': best['speed'],
            'ball_path': best['path'],
            'cushion_count': best['cushions'],
            'score': best['score'],
            'angle_deg': np.degrees(best['angle']),
        }

    def _evaluate_batch(self, cue_2d, tgt_2d, obs_2d, angles, speeds, max_cushions):
        """일괄 평가"""
        results = []
        for angle, speed in zip(angles, speeds):
            direction = np.array([np.cos(angle), np.sin(angle)])
            path, hit, cushions, collided = self.simulate_ball_path(
                cue_2d, direction, speed, obs_2d, max_cushions
            )
            score = self._score(path, tgt_2d, hit, cushions, collided)
            results.append({
                'angle': angle, 'speed': speed, 'path': path,
                'hit': hit, 'cushions': cushions, 'score': score
            })
        return results

    def simulate_ball_path(self, start, direction, speed, obstacles, max_cushions,
                           dt=0.002, max_steps=3000):
        """2D 공 궤적 시뮬레이션 (기하학적 반사)

        Returns:
            path: [(x,y), ...] 궤적 포인트
            hit_target: bool — 목표공에 도달했는지는 외부에서 판정
            cushion_count: int
            collided_obstacle: bool
        """
        pos = np.array(start, dtype=float)
        vel = direction * speed
        path = [pos.copy()]
        cushion_count = 0
        friction = ANNEAL_ROLLING_FRICTION

        b = self.bounds
        ball_r = self.ball_r

        for _ in range(max_steps):
            # 마찰 감속
            spd = np.linalg.norm(vel)
            if spd < 0.005:
                break
            vel *= max(0, 1 - friction * dt)

            pos = pos + vel * dt
            path.append(pos.copy())

            # 쿠션 반사
            reflected = False
            if pos[0] - ball_r <= b['x_min']:
                pos[0] = b['x_min'] + ball_r
                vel[0] = abs(vel[0]) * MAZE_CUSHION_RESTITUTION
                reflected = True
            elif pos[0] + ball_r >= b['x_max']:
                pos[0] = b['x_max'] - ball_r
                vel[0] = -abs(vel[0]) * MAZE_CUSHION_RESTITUTION
                reflected = True
            if pos[1] - ball_r <= b['y_min']:
                pos[1] = b['y_min'] + ball_r
                vel[1] = abs(vel[1]) * MAZE_CUSHION_RESTITUTION
                reflected = True
            elif pos[1] + ball_r >= b['y_max']:
                pos[1] = b['y_max'] - ball_r
                vel[1] = -abs(vel[1]) * MAZE_CUSHION_RESTITUTION
                reflected = True

            if reflected:
                cushion_count += 1
                if cushion_count > max_cushions + 1:
                    break

            # 장애물 충돌 판정
            for ox, oy, o_r in obstacles:
                d = np.linalg.norm(pos - np.array([ox, oy]))
                if d < ball_r + o_r:
                    return path, False, cushion_count, True

        return path, False, cushion_count, False

    def _score(self, path, target_2d, hit, cushions, collided):
        """스코어링 함수"""
        if collided:
            return -1e6  # 장애물 충돌 → 즉시 기각

        # 경로 중 목표공에 가장 가까운 거리
        min_dist = float('inf')
        for pt in path:
            d = np.linalg.norm(np.array(pt) - target_2d)
            if d < min_dist:
                min_dist = d

        # 명중 판정 (2r 이내)
        hit_threshold = self.ball_r * 2 + 0.005
        hit_score = 1000 if min_dist < hit_threshold else 0

        # 거리 페널티
        dist_penalty = -min_dist * 500

        # 쿠션 보너스 (3쿠션이면 추가 점수)
        cushion_bonus = min(cushions, 3) * 30

        return hit_score + dist_penalty + cushion_bonus

    def find_best_shot(self, cue_pos, target_pos, obstacles):
        """plan_shot의 편의 래퍼 — state_machine에서 호출용"""
        return self.plan_shot(cue_pos, target_pos, obstacles)
