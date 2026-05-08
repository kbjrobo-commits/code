"""
SimMode / RealMode 통합 로봇 컨트롤러
=======================================
동일한 인터페이스로 PyBullet 시뮬레이션과 실제 Indy7 로봇을 제어
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.pybullet_core import PybulletCore
from src.utils import Rot2eul
from project.ik_solver import IKSolver
from project.config import *


class RobotController:
    """통합 로봇 컨트롤러 — Sim / Real 모드 자동 전환"""

    def __init__(self, mode='sim', robot_ip=None, headless=False):
        self.mode = mode
        self.headless = headless
        self.pb = None
        self.indy = None
        self.ik = None
        self._connected = False
        self._pinModel = None
        self._q_current = None
        self._client_id = None

        if mode == 'real' and robot_ip is None:
            robot_ip = ROBOT_IP
        self._robot_ip = robot_ip

    def connect(self):
        if self.headless:
            self._connect_headless()
        else:
            self._connect_gui()

    def _connect_gui(self):
        self.pb = PybulletCore()
        self.pb.connect(
            robot_name="indy7_v2",
            joint_limit=True,
            constraint_visualization=False
        )
        # 관절 한계를 프레임워크에서 가져와 IK에 전달
        q_lo = np.array(self.pb.my_robot.q_lower).reshape(-1, 1)
        q_hi = np.array(self.pb.my_robot.q_upper).reshape(-1, 1)
        self.ik = IKSolver(
            self.pb.my_robot.pinModel,
            gain=IK_GAIN,
            damping=IK_DAMPING,
            q_lower=q_lo,
            q_upper=q_hi
        )
        if self.mode == 'real':
            try:
                from neuromeka import IndyDCP3
                self.indy = IndyDCP3(robot_ip=self._robot_ip, index=0)
                print(f"[RobotController] Real robot connected at {self._robot_ip}")
            except Exception as e:
                print(f"[RobotController] Warning: Could not connect to real robot: {e}")
                self.mode = 'sim'

        self._connected = True
        self._pinModel = self.pb.my_robot.pinModel
        self._client_id = self.pb.ClientId
        self._q_current = self.pb.my_robot.q.copy()
        print(f"[RobotController] Connected in '{self.mode}' mode (GUI)")

    def _connect_headless(self):
        import pybullet as p
        import pybullet_data
        from src.utils.pinocchio_utils import PinocchioModel

        self._client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240)

        p.loadURDF("plane.urdf", physicsClientId=self._client_id)

        urdf_path = "src/assets/urdf/indy7_v2/indy7_v2/model.urdf"
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self._robot_id = p.loadURDF(
            urdf_path, basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            flags=flags, physicsClientId=self._client_id
        )

        self._pinModel = PinocchioModel('src/assets/urdf/indy7_v2/indy7_v2')
        # Indy7 URDF 관절 한계 (rad) × SAFETY_FACTOR=0.95
        _jl = 3.05432619099 * 0.95  # J0-J4
        _j5 = 3.75245789179 * 0.95  # J5
        q_lo = np.array([-_jl, -_jl, -_jl, -_jl, -_jl, -_j5]).reshape(-1, 1)
        q_hi = np.array([ _jl,  _jl,  _jl,  _jl,  _jl,  _j5]).reshape(-1, 1)
        self.ik = IKSolver(self._pinModel, gain=IK_GAIN, damping=IK_DAMPING,
                           q_lower=q_lo, q_upper=q_hi)

        self._movable_joints = [0, 1, 2, 3, 4, 5]
        self._q_current = np.array(HOME_Q_RAD).reshape(-1, 1)

        for i, idx in enumerate(self._movable_joints):
            p.resetJointState(self._robot_id, idx,
                              self._q_current[i, 0],
                              physicsClientId=self._client_id)

        self._connected = True
        print(f"[RobotController] Connected in '{self.mode}' mode (HEADLESS)")

    def disconnect(self):
        if self.headless:
            import pybullet as p
            p.disconnect(physicsClientId=self._client_id)
        elif self.pb is not None:
            self.pb.disconnect()
        self._connected = False

    def move_home(self, safe_retract=True):
        """홈 복귀 — safe_retract=True면 먼저 팔을 들어올린 후 복귀"""
        if safe_retract:
            # 현재 자세에서 Joint 2,3만 들어올려 테이블 위 상공 확보
            RETRACT_Q_DEG = [0, -15, -45, 0, -90, 0]
            self.movej(RETRACT_Q_DEG, wait=True)
        self.movej(HOME_Q_DEG, wait=True)
        print("[RobotController] Moved to home position")

    def movej(self, q_deg, wait=True):
        q_deg = list(np.asarray(q_deg).flatten())
        q_rad = [d * np.pi / 180 for d in q_deg]

        if self.headless:
            import pybullet as p
            self._q_current = np.array(q_rad).reshape(-1, 1)
            for i, idx in enumerate(self._movable_joints):
                p.resetJointState(self._robot_id, idx, q_rad[i],
                                  physicsClientId=self._client_id)
            for _ in range(240):
                p.stepSimulation(physicsClientId=self._client_id)
        elif self.mode == 'sim':
            self.pb.MoveRobot(q_deg, degree=True)
            if wait:
                time.sleep(1.0)
        elif self.mode == 'real':
            self.indy.movej(q_deg)
            self.pb.MoveRobot(q_deg, degree=True)
            if wait:
                self._wait_indy()

    def get_current_q(self):
        if self.headless:
            return self._q_current.copy()
        elif self.mode == 'sim':
            return self.pb.my_robot.q.copy()
        elif self.mode == 'real':
            q_deg = self.indy.get_control_data()['q']
            return np.array(q_deg).reshape(-1, 1) * np.pi / 180

    def get_current_T(self):
        q = self.get_current_q()
        return self._pinModel.FK(q)

    def get_FK(self, q_rad):
        return self._pinModel.FK(q_rad)

    def boost_pd_gains(self, kp=800, kd=40):
        """PD 제어 게인 강화 — 도구 장착 시 안정성 향상

        PybulletRobot의 _compute_torque_input 메서드를 직접 교체하여
        더 강한 Kp/Kd로 위치 추적 정밀도 향상
        """
        if self.headless or self.pb is None:
            return

        robot = self.pb.my_robot

        def _boosted_torque_input():
            qddot = robot._qddot_des + \
                     kp * (robot._q_des - robot._q) + \
                     kd * (robot._qdot_des - robot._qdot)
            robot._tau = robot._M @ qddot + robot._c + robot._g

        robot._compute_torque_input = _boosted_torque_input
        print(f"[RobotController] PD gains boosted: Kp={kp}, Kd={kd}")

    def execute_trajectory(self, trajectory_SE3, dt=None, visualize=True,
                           phase_indices=None, strike_speed=None):
        """SE3 궤적 실행

        phase_indices가 있으면:
        - Approach: 정밀 접근 (매 포인트 time.sleep)
        - Strike+Follow: 속도 제어 포인트-by-포인트 추적 (strike_speed 기반)
        """
        if dt is None:
            dt = TRAJECTORY_DT

        if self.headless:
            self._execute_headless(trajectory_SE3, dt)
        elif self.mode == 'sim':
            self._execute_sim(trajectory_SE3, dt, visualize, phase_indices,
                              strike_speed=strike_speed)
        elif self.mode == 'real':
            self._execute_real(trajectory_SE3, dt, phase_indices=phase_indices)

    def _execute_headless(self, trajectory, dt):
        import pybullet as p
        q_i = self.get_current_q()
        steps_per_point = max(int(dt / (1./240)), 1)

        for T_goal in trajectory:
            q_i = self.ik.solve_step(q_i, T_goal)
            for i, idx in enumerate(self._movable_joints):
                p.resetJointState(self._robot_id, idx, q_i[i, 0],
                                  physicsClientId=self._client_id)
            for _ in range(steps_per_point):
                p.stepSimulation(physicsClientId=self._client_id)

        self._q_current = q_i.copy()

    def _execute_sim(self, trajectory, dt, visualize=True, phase_indices=None,
                      strike_speed=None):
        """시뮬레이션 모드 궤적 실행 — 타격 후 즉시 후퇴"""
        q_i = self.get_current_q()

        if visualize and self.pb is not None:
            vis_step = max(len(trajectory) // 20, 1)
            vis_frames = [trajectory[i] for i in range(0, len(trajectory), vis_step)]
            self.pb.add_debug_frames(vis_frames)

        if phase_indices is None:
            for T_goal in trajectory:
                q_i = self.ik.solve_step(q_i, T_goal)
                self.pb.MoveRobot(q_i, degree=False)
                time.sleep(dt)
            self._q_current = q_i.copy()
            return

        # === Phase-aware 실행 ===
        approach_range = phase_indices.get('approach', (0, 0))
        strike_range = phase_indices.get('strike', (0, 0))

        # Phase 1: Approach (정밀 접근 — cubic time scaling)
        print(f"    [Approach] {approach_range[1] - approach_range[0]} pts")
        for i in range(approach_range[0], approach_range[1]):
            q_i = self.ik.solve_step(q_i, trajectory[i])
            self.pb.MoveRobot(q_i, degree=False)
            time.sleep(dt)

        # Approach 끝 = 타격 준비 위치 (공 뒤 approach_dist)
        q_ready = q_i.copy()
        T_ready = trajectory[approach_range[1] - 1] if approach_range[1] > 0 else trajectory[0]

        # Approach 끝에서 안정화 대기
        time.sleep(0.5)

        # Phase 2: Strike — 실제 임펄스 타격 (치기, 밀기 아님)
        # 핵심 아이디어: 접근 단계 마지막 부분에서 이미 가속을 시작하여
        # 공에 닿는 순간 EE가 이미 목표 속도에 도달해 있도록 함
        strike_start = strike_range[0]
        strike_end = strike_range[1]

        if strike_end <= strike_start:
            self._q_current = q_i.copy()
            return

        follow_range = phase_indices.get('follow', (strike_end, strike_end))

        if strike_speed is not None and strike_speed > 0.01:
            print(f"    [Strike] Impact at {strike_speed:.3f} m/s → swing-through")

            # Follow-through 끝 지점 (공 너머 5cm) = 스윙의 최종 목표
            T_follow = trajectory[-1]
            q_follow = self.ik.solve_step(q_i, T_follow)

            # === 가속 시작: 공 6cm 앞에서 MoveRobot(q_follow)로 먼 목표 설정 ===
            # PD 제어기는 먼 목표를 향해 큰 토크를 발생시켜 강하게 가속
            # 동시에 qdot_des를 주입하여 Kd 브레이크(감쇠)를 부스터로 전환

            # 가속에 필요한 시간 추정
            T_impact = trajectory[strike_end - 1]
            p_ready = T_ready[0:3, 3]
            p_follow = T_follow[0:3, 3]
            full_dist = np.linalg.norm(p_follow - p_ready)
            # 총 스윙 거리에 대해 평균 속도로 가속 시간 계산
            swing_time = full_dist / (strike_speed * 0.7)
            swing_time = np.clip(swing_time, 0.05, 0.8)

            # 관절 공간에서의 목표 속도 계산
            avg_qdot = (q_follow - q_ready) / swing_time
            if hasattr(self.pb.my_robot, '_qdot_des'):
                self.pb.my_robot._qdot_des = avg_qdot

            # 풀스윙 목표 설정 → PD가 큰 위치 오차 + 양의 속도 목표로 강하게 가속
            self.pb.MoveRobot(q_follow, degree=False)

            # 스윙 시간 대기 (공을 관통하며 치는 시간)
            time.sleep(swing_time)

            # === 스윙 완료 → 후퇴 ===
            # 속도 목표 초기화 (제동 역할 복구)
            if hasattr(self.pb.my_robot, '_qdot_des'):
                self.pb.my_robot._qdot_des = np.zeros([self.pb.my_robot.numJoints, 1])

            # === 즉시 Home 복귀 (대기 없음) — 공이 로봇을 치기 전에 빠짐 ===
            # PD 컨트롤러가 백그라운드에서 이동, 공 물리 동시 진행
            self.pb.MoveRobot(list(HOME_Q_DEG), degree=True)
            time.sleep(0.05)  # 최소 시뮬 진행
            q_i = self.get_current_q()
        else:
            # fallback: 단순 임팩트 (strike_speed 미제공)
            print(f"    [Strike] Simple impact -> retract")
            T_impact = trajectory[strike_end - 1]
            q_impact = self.ik.solve_step(q_i, T_impact)
            self.pb.MoveRobot(q_impact, degree=False)
            time.sleep(0.1)
            # 즉시 Home (대기 없음)
            self.pb.MoveRobot(list(HOME_Q_DEG), degree=True)
            time.sleep(0.05)
            q_i = self.get_current_q()

        self._q_current = q_i.copy()

    def _execute_real(self, trajectory, dt, phase_indices=None, **kwargs):
        """Phase-aware 실제 로봇 실행

        기획서 3.3절: 초고속 시계열 궤적 스트리밍
        - Approach: movel (내장 감속 OK, 안전)
        - Strike+Follow: 1kHz teleop 스트리밍 (vel_ratio=1.0, 감속 무력화)
        """
        if phase_indices is None:
            # Phase 정보 없으면 전체를 균일 teleop 재생 (기존 호환)
            self._execute_real_uniform(trajectory, dt)
            return

        approach_end = phase_indices['approach'][1]

        # Phase 1: Approach — movel로 안전하게 접근
        T_ready = trajectory[approach_end - 1]
        p_ready = self._SE3_to_task_pose(T_ready)
        print(f"    [Real] Approach via movel...")
        self.indy.movel(p_ready)
        self._wait_indy()
        print(f"    [Real] Approach complete")

        # Phase 2: Strike + Follow — 1kHz teleop 스트리밍
        strike_traj = trajectory[approach_end:]
        print(f"    [Real] Strike streaming: {len(strike_traj)} pts @ 1kHz...")
        self.indy.start_teleop(0)
        time.sleep(0.3)

        idx = 0
        while idx < len(strike_traj):
            tic = time.time()
            p_des = self._SE3_to_task_pose(strike_traj[idx])
            self.indy.movetelel_abs(p_des, vel_ratio=1.0, acc_ratio=1.0)
            toc = time.time()
            idx += max(1, int((toc - tic) / dt))

        self._wait_indy()
        self.indy.stop_teleop()
        print(f"    [Real] Strike complete")

    def _execute_real_uniform(self, trajectory, dt):
        """기존 호환용: 전체 궤적 균일 teleop 재생"""
        self.indy.start_teleop(0)
        time.sleep(1)
        idx = 0
        while idx < len(trajectory):
            tic = time.time()
            T_des = trajectory[idx]
            p_des = self._SE3_to_task_pose(T_des)
            self.indy.movetelel_abs(p_des, vel_ratio=0.5, acc_ratio=1)
            self._sync_indy()
            toc = time.time()
            didx = int((toc - tic) // dt) + 1
            idx += didx
        self._wait_indy()
        self.indy.stop_teleop()

    def _SE3_to_task_pose(self, T):
        p_des = np.zeros(6)
        p_des[0:3] = 1000 * T[0:3, 3]
        p_des[3:6] = Rot2eul(T[0:3, 0:3], seq='XYZ', degree=True)
        return p_des.tolist()

    def _sync_indy(self):
        if self.indy is not None:
            q = self.indy.get_control_data()['q']
            self.pb.MoveRobot(q, degree=True)

    def _wait_indy(self):
        if self.indy is None:
            return
        while True:
            self._sync_indy()
            if not self.indy.get_motion_data()["is_in_motion"]:
                break
            time.sleep(0.01)
        print("[RobotController] Motion complete")

    def add_debug_frames(self, T_list):
        if self.pb is not None:
            self.pb.add_debug_frames(T_list)

    def destroy_debug_frames(self):
        if self.pb is not None:
            self.pb.destroy_debug_frames()
