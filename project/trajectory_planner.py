"""
타격 궤적 생성기
==================
미니골프/포켓볼 공통 타격 궤적을 3단계로 생성:
  1. Approach: 현재 위치 → 타격 준비 위치 (cubic time scaling)
  2. Strike: 준비 위치 → 공 (일정 속도 직선)
  3. Follow-through: 공 관통 후 감속
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    xyzeul2SE3, Rot2Vec, Vec2Rot, eul2Rot, Rot2eul
)
from project.config import TOOL_VERTICAL_DROP, TOOL_HORIZONTAL_EXT, PIN_PB_EE_Z_OFFSET


def _cubic_time_scaling(T_total, num_points):
    """Cubic polynomial time scaling: s(0)=0, s(T)=1, s'(0)=s'(T)=0

    Returns:
        s_array: (num_points,) array of scaling values [0, 1]
    """
    a = np.linalg.solve(
        [[1, 0, 0, 0],
         [1, T_total, T_total**2, T_total**3],
         [0, 1, 0, 0],
         [0, 1, 2*T_total, 3*T_total**2]],
        [0, 1, 0, 0]
    )
    s_array = np.zeros(num_points)
    for i in range(num_points):
        t = T_total * (i + 1) / num_points
        s_array[i] = np.dot(a, [1, t, t**2, t**3])
    return s_array


def interpolate_SE3_decoupled(T_start, T_end, s):
    """Decoupled position/orientation interpolation"""
    p_start = T_start[0:3, [3]]
    p_end = T_end[0:3, [3]]
    R_start = T_start[0:3, 0:3]
    R_end = T_end[0:3, 0:3]

    p_goal = p_start + s * (p_end - p_start)
    R_goal = R_start @ Vec2Rot(Rot2Vec(R_start.T @ R_end) * s)

    T_interp = np.eye(4)
    T_interp[0:3, [3]] = p_goal
    T_interp[0:3, 0:3] = R_goal
    return T_interp


class TrajectoryPlanner:
    """궤적 생성기 — 직선, 원형 등 기본 궤적 지원"""

    @staticmethod
    def plan_linear(T_start, T_end, duration, dt):
        """직선 경로 (cubic time scaling)"""
        num_points = int(duration / dt)
        s_array = _cubic_time_scaling(duration, num_points)
        trajectory = []
        for s in s_array:
            T = interpolate_SE3_decoupled(T_start, T_end, s)
            trajectory.append(T)
        return trajectory

    @staticmethod
    def plan_constant_speed_linear(T_start, T_end, speed, dt):
        """일정 속도 직선 경로"""
        p_start = T_start[0:3, 3]
        p_end = T_end[0:3, 3]
        distance = np.linalg.norm(p_end - p_start)
        if distance < 1e-6:
            return [T_end.copy()]

        duration = distance / speed
        num_points = max(int(duration / dt), 1)

        trajectory = []
        for i in range(num_points):
            s = (i + 1) / num_points
            T = interpolate_SE3_decoupled(T_start, T_end, s)
            trajectory.append(T)
        return trajectory


class StrikeTrajectoryPlanner:
    """타격 궤적 생성기 — Approach → Strike → Follow-through"""

    def __init__(self, approach_duration=3.0, dt=0.001):
        self.approach_duration = approach_duration
        self.dt = dt
        self.traj_planner = TrajectoryPlanner()

    def compute_strike_orientation(self, strike_direction, tool_rotation=0.0):
        """ㄴ자 도구용 엔드이펙터 자세 (Rotation matrix) 계산

        EE의 z축이 아래를 향하고 (도구의 수직 부분이 내려감),
        EE의 x축이 타격 방향을 향하도록 설정 (도구의 수평 부분이 공을 향함).

        ㄴ자 도구 구조:
          EE (z축 = 아래)
           |  ← EE z축 방향
           |
           └──● ← EE x축 방향 (strike_dir)

        tool_rotation(φ): z축 주위 회전. 자유도 활용.
        """
        strike_dir = np.array(strike_direction).flatten()
        # 수평 성분만 사용 (ㄴ자 도구는 수평 타격)
        strike_dir[2] = 0
        strike_dir = strike_dir / np.linalg.norm(strike_dir)

        # z축: 아래 (도구의 수직 부분이 내려감)
        z_axis = np.array([0.0, 0.0, -1.0])

        # x축: 타격 방향 (도구의 수평 부분이 공을 향함)
        x_axis = strike_dir.copy()

        # y축: z × x (오른손 좌표계 완성)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])

        # φ 회전: z축 주위 회전 (도구 원통 대칭 활용)
        if abs(tool_rotation) > 1e-6:
            c, s = np.cos(tool_rotation), np.sin(tool_rotation)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            R = R @ Rz

        return R

    def plan_strike(self, T_current, ball_pos, strike_direction,
                    strike_speed=0.5, approach_dist=0.08,
                    follow_dist=0.10, strike_height=None,
                    tool_offset=0.0, tool_rotation=0.0,
                    table_bounds=None):
        """완전한 타격 궤적 생성 (ㄴ자 도구 수평 타격)

        Args:
            T_current: 현재 엔드이펙터 SE3 (4,4)
            ball_pos: 공 위치 [x, y, z]
            strike_direction: 타격 방향 벡터 [dx, dy, dz] (수평일 때 dz≈0)
            strike_speed: 타격 속도 (m/s)
            approach_dist: 접근 거리 (m)
            follow_dist: Follow-through 거리 (m)
            strike_height: 미사용 (하위 호환용으로 유지)
            tool_offset: 미사용 (ㄴ자 도구에서는 자동 계산)
            tool_rotation: 도구 축(z) 주위 회전 φ (rad) — 특이점/관절한계 회피
            table_bounds: dict {'x_min','x_max','y_min','y_max'} 테이블 범위

        Returns:
            trajectory: SE3 리스트
            phase_indices: 각 단계의 인덱스 경계

        ㄴ자 도구 형상:
          EE
           |  (TOOL_VERTICAL_DROP)
           |
           └──● (TOOL_HORIZONTAL_EXT, 큐팁이 공에 닿음)

        EE 목표 = 공 위치 - strike_dir * TOOL_HORIZONTAL_EXT + [0,0, TOOL_VERTICAL_DROP]
        """
        ball_pos = np.array(ball_pos).flatten()
        strike_dir = np.array(strike_direction).flatten()
        strike_dir = strike_dir / np.linalg.norm(strike_dir)

        # 타격 자세 (φ 회전 적용)
        R_strike = self.compute_strike_orientation(strike_dir, tool_rotation)

        # ㄴ자 도구 오프셋: 큐팁이 ball_pos에 도달하려면
        # EE는 공 뒤쪽(수평) + 공 위(수직)에 위치해야 함
        # PIN_PB_EE_Z_OFFSET: Pinocchio FK가 PyBullet EE보다 62mm 높으므로 보정
        ee_offset = -strike_dir * TOOL_HORIZONTAL_EXT + np.array([0, 0, TOOL_VERTICAL_DROP + PIN_PB_EE_Z_OFFSET])

        # 1. 준비 위치: 공 뒤쪽 approach_dist만큼 (+ ㄴ자 오프셋)
        ready_pos = ball_pos - strike_dir * approach_dist + ee_offset
        T_ready = np.eye(4)
        T_ready[0:3, 0:3] = R_strike
        T_ready[0:3, 3] = ready_pos

        # 2. 임팩트 위치: 큐팁이 공 표면에 닿는 지점
        impact_pos = ball_pos + ee_offset
        T_impact = np.eye(4)
        T_impact[0:3, 0:3] = R_strike
        T_impact[0:3, 3] = impact_pos

        # 3. Follow-through 위치: 임팩트 후 계속 전진 (수평이므로 Z클램핑 불필요)
        follow_pos = impact_pos + strike_dir * follow_dist
        T_follow = np.eye(4)
        T_follow[0:3, 0:3] = R_strike
        T_follow[0:3, 3] = follow_pos

        # 궤적 생성
        # Phase 1: Approach — 2단계 안전 접근
        safe_height = 0.25  # 공 위 25cm 상공 경유 (장애물 충분히 넘김)
        above_pos = ready_pos.copy()
        above_pos[2] = max(ready_pos[2] + safe_height, ball_pos[2] + safe_height)
        T_above = np.eye(4)
        T_above[0:3, 0:3] = R_strike
        T_above[0:3, 3] = above_pos

        # Stage 1: Home → 상공 (팔이 테이블 위 공중에서 이동)
        rise_duration = self.approach_duration * 0.6
        rise_traj = self.traj_planner.plan_linear(
            T_current, T_above, rise_duration, self.dt
        )
        # Stage 2: 상공 → Ready (공 뒤로 하강)
        descend_duration = self.approach_duration * 0.4
        descend_traj = self.traj_planner.plan_linear(
            T_above, T_ready, descend_duration, self.dt
        )
        approach_traj = rise_traj + descend_traj

        # Phase 2: Strike (일정 속도)
        strike_traj = self.traj_planner.plan_constant_speed_linear(
            T_ready, T_impact, strike_speed, self.dt
        )

        # Phase 3: Follow-through (감속)
        follow_duration = follow_dist / (strike_speed * 0.5)  # 감속
        follow_traj = self.traj_planner.plan_linear(
            T_impact, T_follow, max(follow_duration, 0.1), self.dt
        )

        trajectory = approach_traj + strike_traj + follow_traj

        phase_indices = {
            'approach': (0, len(approach_traj)),
            'strike': (len(approach_traj), len(approach_traj) + len(strike_traj)),
            'follow': (len(approach_traj) + len(strike_traj), len(trajectory)),
        }

        return trajectory, phase_indices
