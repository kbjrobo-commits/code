"""
Numerical Inverse Kinematics 솔버
===================================
Damped Least Squares (DLS) 방식의 IK
- 적응형 damping (manipulability 기반)
- 관절 한계 검증
- 궤적 사전검증 파이프라인
"""
import numpy as np
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import Rot2Vec


class IKSolver:
    """Numerical IK Solver using Damped Least Squares

    Features:
        - 적응형 DLS damping (manipulability 기반 안전망)
        - 관절 한계 검증
        - 궤적 사전검증 (실행 전 전체 IK 풀이 + 검증)
    """

    # Indy7 URDF 관절 한계 (rad)
    INDY7_Q_LOWER = np.array([-3.054, -3.054, -3.054, -3.054, -3.054, -3.752]).reshape(-1, 1)
    INDY7_Q_UPPER = np.array([ 3.054,  3.054,  3.054,  3.054,  3.054,  3.752]).reshape(-1, 1)

    def __init__(self, pinocchio_model, gain=1.0, damping=1e-3,
                 q_lower=None, q_upper=None):
        """
        Args:
            pinocchio_model: PinocchioModel 인스턴스
            gain: IK 스텝 크기 (0~1)
            damping: DLS 기본 감쇠 계수
            q_lower: 관절 하한 (n,1) ndarray — None이면 Indy7 기본값 사용
            q_upper: 관절 상한 (n,1) ndarray — None이면 Indy7 기본값 사용
        """
        self.pin = pinocchio_model
        self.gain = gain
        self.damping = damping
        self.q_lower = q_lower if q_lower is not None else self.INDY7_Q_LOWER
        self.q_upper = q_upper if q_upper is not None else self.INDY7_Q_UPPER
        # 적응형 damping 파라미터
        self._w_thresh = 0.005   # manipulability 임계치
        self._damping_max = 0.05 # 특이점 근방 최대 damping

    # ─── 핵심 도구 ─────────────────────────────────────

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

    def manipulability(self, q):
        """Manipulability index: w = sqrt(det(J * J^T))

        Returns:
            w: manipulability (0에 가까울수록 특이점)
        """
        Jv, _ = self._compute_jacobian(q)
        return np.sqrt(max(np.linalg.det(Jv @ Jv.T), 0))

    def check_joint_limits(self, q):
        """관절 한계 위반 여부 확인

        Returns:
            valid: 모든 관절이 한계 내이면 True
            violations: 위반 관절 인덱스 리스트
        """
        if self.q_lower is None or self.q_upper is None:
            return True, []
        q = np.asarray(q).flatten()
        q_lo = np.asarray(self.q_lower).flatten()
        q_hi = np.asarray(self.q_upper).flatten()
        violations = []
        for i in range(len(q)):
            if q[i] < q_lo[i] or q[i] > q_hi[i]:
                violations.append(i)
        return len(violations) == 0, violations

    # ─── IK 풀이 ──────────────────────────────────────

    def solve_step(self, q_current, T_goal):
        """단일 IK 스텝 (적응형 DLS damping + 특이점 회피 포함)

        Args:
            q_current: 현재 관절각 (n,1) ndarray (rad)
            T_goal: 목표 SE3 (4,4) ndarray

        Returns:
            q_new: 업데이트된 관절각 (n,1) ndarray (rad)
        """
        q = np.asarray(q_current).reshape(-1, 1)
        Jv, T_i = self._compute_jacobian(q)
        p_err = self._compute_error(T_i, T_goal)

        # 적응형 damping: 특이점 접근 시 자동 증가 (안전망)
        w = np.sqrt(max(np.linalg.det(Jv @ Jv.T), 0))
        if w < self._w_thresh:
            lam = self._damping_max * (1 - (w / self._w_thresh)**2)
        else:
            lam = self.damping
        lam = max(lam, self.damping)

        # Damped Least Squares
        JJT = Jv @ Jv.T + lam * np.eye(6)
        q_new = q + self.gain * Jv.T @ np.linalg.solve(JJT, p_err)

        # 특이점 회피: J3·J5 월드 축 외적이 작으면 perturbation
        import pinocchio as pin
        pin.forwardKinematics(self.pin.pinModel, self.pin.pinData, q_new)
        if self.pin.pinModel.njoints >= 6:
            # oMi: Pinocchio index = URDF joint index + 1 (universe joint가 0)
            # joint3 = oMi[4], joint5 = oMi[6]
            j3_axis = self.pin.pinData.oMi[4].rotation @ np.array([0, 0, 1])
            j5_axis = self.pin.pinData.oMi[6].rotation @ np.array([0, 0, 1])
            cross_mag = np.linalg.norm(np.cross(j3_axis, j5_axis))
            if cross_mag < 0.15:
                # joint4 (q[4], oMi[5]) = J3-J5 사이의 꺾임 관절
                # 이걸 움직여야 두 축 사이 각도가 변함
                j4_val = float(q_new[4, 0]) if q_new.ndim > 1 else float(q_new[4])
                push_dir = 1.0 if j4_val >= 0 else -1.0
                if abs(j4_val) < 0.01:
                    push_dir = 1.0
                q_new[4] += push_dir * 0.10  # ~5.7° perturbation

        # 관절 한계 클램핑
        q_new = np.clip(q_new, self.q_lower, self.q_upper)
        return q_new

    # ── solve_to_target: 현재 미사용, 향후 사전검증 파이프라인에서 활용 예정 ──
    # def solve_to_target(self, q_init, T_goal, max_iter=100, tol=1e-4):
    #     """목표 SE3까지 IK 반복 풀이"""
    #     q = np.asarray(q_init).reshape(-1, 1)
    #     for i in range(max_iter):
    #         Jv, T_i = self._compute_jacobian(q)
    #         p_err = self._compute_error(T_i, T_goal)
    #         pos_err = np.linalg.norm(p_err[0:3])
    #         if pos_err < tol:
    #             return q, True, pos_err
    #         JJT = Jv @ Jv.T + self.damping * np.eye(6)
    #         q = q + self.gain * Jv.T @ np.linalg.solve(JJT, p_err)
    #     T_final = self.pin.FK(q)
    #     final_err = np.linalg.norm(T_goal[0:3, 3] - T_final[0:3, 3])
    #     return q, final_err < tol * 10, final_err

    # ─── 궤적 풀이 ────────────────────────────────────

    def solve_trajectory(self, q_init, trajectory_SE3):
        """궤적 전체에 대한 IK 풀이 (기존 호환)

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

    def solve_trajectory_validated(self, q_init, trajectory_SE3,
                                    w_threshold=0.002, dq_max=0.35,
                                    validate_from=0, table_bounds=None):
        """궤적 전체 IK 사전풀이 + 검증

        실행 전에 전체 궤적을 풀어서 관절한계/특이점/급격한 점프를 확인.
        검증 실패 시 어느 지점에서 문제가 발생했는지 상세 보고.

        Args:
            q_init: 초기 관절각 (n,1)
            trajectory_SE3: SE3 리스트
            w_threshold: manipulability 경고 임계치
            dq_max: 연속 점 간 최대 허용 관절 변화 (rad)
            validate_from: 이 인덱스부터 검증 시작 (이전은 IK만 풀이)
            table_bounds: dict {'x_min','x_max','y_min','y_max'} — 도구 팁 범위 검사용

        Returns:
            result: dict {
                'q_trajectory': [(n,1), ...],
                'valid': bool,
                'issues': [str, ...],    # 문제 설명 리스트
                'manipulability': [float, ...],
                'min_manipulability': float,
                'joint_limit_violations': [(index, joint_id), ...],
                'min_singularity_margin': float,  # J3·J5 외적 최소값
            }
        """
        from project.config import (TOOL_HORIZONTAL_EXT, TOOL_VERTICAL_DROP,
                                     MAZE_TABLE_SURFACE_HEIGHT, MAZE_TABLE_HEIGHT)

        q_trajectory = []
        manipulability_list = []
        singularity_margins = []
        issues = []
        joint_violations = []

        q_i = np.asarray(q_init).reshape(-1, 1)
        q_prev = q_i.copy()

        for idx, T_goal in enumerate(trajectory_SE3):
            for _ in range(5):
                q_i = self.solve_step(q_i, T_goal)
            q_trajectory.append(q_i.copy())

            # validate_from 이전은 IK만 풀고 검증 건너뜀 (고공 접근구간 등)
            if idx < validate_from:
                q_prev = q_i.copy()
                continue

            # 1. Manipulability 추적 (soft warning — 순위 정렬에만 사용, blocking 아님)
            w = self.manipulability(q_i)
            manipulability_list.append(w)

            # 2. 관절 한계 검사 (blocking)
            valid, viols = self.check_joint_limits(q_i)
            if not valid:
                joint_violations.append((idx, viols))
                issues.append(
                    f"[pt {idx}] 관절 한계 초과: joints {viols}")

            # 3. 바디 뚫림 검사 (blocking, 테이블 높이 기반 동적 임계치)
            body_z_min = MAZE_TABLE_SURFACE_HEIGHT + MAZE_TABLE_HEIGHT / 2 + 0.03
            import pinocchio as pin
            pin.forwardKinematics(self.pin.pinModel, self.pin.pinData, q_i)
            for j_id in range(3, self.pin.pinModel.njoints):
                pos = self.pin.pinData.oMi[j_id].translation
                if pos[1] > 0.05 and pos[2] < body_z_min:
                    issues.append(f"[pt {idx}] 바디 뚫림: joint {j_id} at z={pos[2]:.3f} < {body_z_min:.3f}m")
                    break

            # 4. 급격한 관절 점프 검사 (blocking)
            dq = np.max(np.abs(q_i - q_prev))
            if dq > dq_max:
                issues.append(
                    f"[pt {idx}] joint jump: max dq={np.degrees(dq):.1f}deg > {np.degrees(dq_max):.1f}deg")

            # 5. 도구 팁이 테이블 범위 안에 있는지 검사 (blocking)
            #    도구 팁 = EE_pos + R_ee @ [TOOL_HORIZONTAL_EXT, 0, -TOOL_VERTICAL_DROP]
            if table_bounds is not None:
                R_ee = T_goal[:3, :3]
                ee_pos = T_goal[:3, 3]
                tool_tip = ee_pos + R_ee @ np.array([TOOL_HORIZONTAL_EXT, 0, -TOOL_VERTICAL_DROP])
                margin = 0.005  # 5mm 마진
                if (tool_tip[0] < table_bounds['x_min'] - margin or
                    tool_tip[0] > table_bounds['x_max'] + margin or
                    tool_tip[1] < table_bounds['y_min'] - margin or
                    tool_tip[1] > table_bounds['y_max'] + margin):
                    issues.append(
                        f"[pt {idx}] 도구팁 테이블밖: tip=({tool_tip[0]:.3f},{tool_tip[1]:.3f})")

            # 6. J3·J5 손목 특이점 검사 (blocking — FK 기반 축 외적)
            if self.pin.pinModel.njoints >= 6:
                j3_axis = self.pin.pinData.oMi[4].rotation @ np.array([0, 0, 1])
                j5_axis = self.pin.pinData.oMi[6].rotation @ np.array([0, 0, 1])
                cross_mag = np.linalg.norm(np.cross(j3_axis, j5_axis))
                singularity_margins.append(cross_mag)
                if cross_mag < 0.1:
                    issues.append(
                        f"[pt {idx}] 손목 특이점: cross(J3,J5)={cross_mag:.4f} < 0.1")

            q_prev = q_i.copy()

        min_w = min(manipulability_list) if manipulability_list else 0
        min_sm = min(singularity_margins) if singularity_margins else 1.0

        return {
            'q_trajectory': q_trajectory,
            'valid': len(issues) == 0,
            'issues': issues,
            'manipulability': manipulability_list,
            'min_manipulability': min_w,
            'joint_limit_violations': joint_violations,
            'min_singularity_margin': min_sm,
        }

