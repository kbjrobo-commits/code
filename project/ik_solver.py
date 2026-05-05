"""
Numerical Inverse Kinematics 솔버
===================================
Damped Least Squares (DLS) 방식의 IK
기존 예시 코드의 패턴을 모듈화
"""
import numpy as np
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import Rot2Vec, xyzeul2SE3


class IKSolver:
    """Numerical IK Solver using Damped Least Squares"""

    def __init__(self, pinocchio_model, gain=1.0, damping=1e-3):
        """
        Args:
            pinocchio_model: PinocchioModel 인스턴스
            gain: IK 스텝 크기 (0~1)
            damping: DLS 감쇠 계수
        """
        self.pin = pinocchio_model
        self.gain = gain
        self.damping = damping

    def _compute_jacobian(self, q):
        """변환된 Jacobian (position + orientation) 계산"""
        T_i = self.pin.FK(q)
        Jb_i = self.pin.Jb(q)

        R_i = T_i[0:3, 0:3]
        A_upper = np.concatenate((np.zeros([3, 3]), R_i), axis=1)
        A_lower = np.concatenate((np.eye(3), np.zeros([3, 3])), axis=1)
        A = np.concatenate((A_upper, A_lower), axis=0)

        Jv_i = A @ Jb_i
        return Jv_i, T_i

    def _compute_error(self, T_current, T_goal):
        """태스크 공간 오차 계산 [position_error; orientation_error]"""
        r_err = T_goal[0:3, [3]] - T_current[0:3, [3]]
        R_err = T_current[0:3, 0:3].T @ T_goal[0:3, 0:3]
        xi_err = Rot2Vec(R_err)
        p_err = np.concatenate((r_err, xi_err), axis=0)
        return p_err

    def solve_step(self, q_current, T_goal):
        """단일 IK 스텝

        Args:
            q_current: 현재 관절각 (n,1) ndarray (rad)
            T_goal: 목표 SE3 (4,4) ndarray

        Returns:
            q_new: 업데이트된 관절각 (n,1) ndarray (rad)
        """
        q = np.asarray(q_current).reshape(-1, 1)
        Jv, T_i = self._compute_jacobian(q)
        p_err = self._compute_error(T_i, T_goal)

        # Damped Least Squares
        JJT = Jv @ Jv.T + self.damping * np.eye(6)
        q_new = q + self.gain * Jv.T @ np.linalg.solve(JJT, p_err)
        return q_new

    def solve_to_target(self, q_init, T_goal, max_iter=100, tol=1e-4):
        """목표 SE3까지 IK 반복 풀이

        Args:
            q_init: 초기 관절각
            T_goal: 목표 SE3
            max_iter: 최대 반복 횟수
            tol: 수렴 허용 오차 (m)

        Returns:
            q_solution: 풀이된 관절각
            success: 수렴 여부
            error: 최종 위치 오차
        """
        q = np.asarray(q_init).reshape(-1, 1)
        for i in range(max_iter):
            Jv, T_i = self._compute_jacobian(q)
            p_err = self._compute_error(T_i, T_goal)

            pos_err = np.linalg.norm(p_err[0:3])
            if pos_err < tol:
                return q, True, pos_err

            JJT = Jv @ Jv.T + self.damping * np.eye(6)
            q = q + self.gain * Jv.T @ np.linalg.solve(JJT, p_err)

        # 최종 오차 계산
        T_final = self.pin.FK(q)
        final_err = np.linalg.norm(T_goal[0:3, 3] - T_final[0:3, 3])
        return q, final_err < tol * 10, final_err

    def solve_trajectory(self, q_init, trajectory_SE3):
        """궤적 전체에 대한 IK 풀이

        Args:
            q_init: 초기 관절각 (n,1)
            trajectory_SE3: SE3 리스트

        Returns:
            q_trajectory: 관절각 리스트 [(n,1), ...]
        """
        q_trajectory = []
        q_i = np.asarray(q_init).reshape(-1, 1)
        for T_goal in trajectory_SE3:
            q_i = self.solve_step(q_i, T_goal)
            q_trajectory.append(q_i.copy())
        return q_trajectory
