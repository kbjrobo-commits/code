"""
SimMode / RealMode ?듯빀 濡쒕큸 而⑦듃濡ㅻ윭
=======================================
?숈씪???명꽣?섏씠?ㅻ줈 PyBullet ?쒕??덉씠?섍낵 ?ㅼ젣 Indy7 濡쒕큸???쒖뼱
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
    """?듯빀 濡쒕큸 而⑦듃濡ㅻ윭 ??Sim / Real 紐⑤뱶 ?먮룞 ?꾪솚"""

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
        self._env = None  # ?섍꼍 李몄“ ???꾪뙥?????꾧뎄 異⑸룎 鍮꾪솢?깊솕??

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
        # 愿???쒓퀎瑜??꾨젅?꾩썙?ъ뿉??媛?몄? IK???꾨떖
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
        # Indy7 URDF 愿???쒓퀎 (rad) 횞 SAFETY_FACTOR=0.95
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
        """??蹂듦? ??safe_retract=True硫?癒쇱? ?붿쓣 ?ㅼ뼱?щ┛ ??蹂듦?"""
        if safe_retract:
            # ?꾩옱 ?먯꽭?먯꽌 Joint 2,3留??ㅼ뼱?щ젮 ?뚯씠釉????곴났 ?뺣낫
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
        """PD ?쒖뼱 寃뚯씤 媛뺥솕 ???꾧뎄 ?μ갑 ???덉젙???μ긽

        PybulletRobot??_compute_torque_input 硫붿꽌?쒕? 吏곸젒 援먯껜?섏뿬
        ??媛뺥븳 Kp/Kd濡??꾩튂 異붿쟻 ?뺣????μ긽
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

    def set_environment(self, env):
        """?섍꼍 李몄“ ?ㅼ젙 ???꾪뙥?????꾧뎄-?먮낵 異⑸룎 鍮꾪솢?깊솕???꾩슂"""
        self._env = env

    def execute_trajectory(self, trajectory_SE3, dt=None, visualize=True,
                           phase_indices=None, strike_speed=None,
                           ball_velocity=None, q_trajectory=None):
        """SE3 沅ㅼ쟻 ?ㅽ뻾

        phase_indices媛 ?덉쑝硫?
        - Approach: ?뺣? ?묎렐 (留??ъ씤??time.sleep)
        - Strike: ball_velocity媛 ?덉쑝硫?怨듭뿉 吏곸젒 ?띾룄 遺??

        ball_velocity: [vx, vy] ??怨듭뿉 吏곸젒 遺?ы븷 ?섑룊 ?띾룄
        """
        if dt is None:
            dt = TRAJECTORY_DT

        if self.headless:
            self._execute_headless(trajectory_SE3, dt)
        elif self.mode == 'sim':
            self._execute_sim(trajectory_SE3, dt, visualize, phase_indices,
                              strike_speed=strike_speed,
                              ball_velocity=ball_velocity,
                              q_trajectory=q_trajectory)
        elif self.mode == 'real':
            self._execute_real(trajectory_SE3, dt, phase_indices=phase_indices,
                               strike_speed=strike_speed)

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
                      strike_speed=None, ball_velocity=None, q_trajectory=None):
        """?쒕??덉씠??紐⑤뱶 沅ㅼ쟻 ?ㅽ뻾 ???쒖닔 PD ?쒖뼱 (?ㅼ젣 濡쒕큸 ???

        Approach: PD ?쒖뼱 (Kp=800) ??Ready ?섎졃 ?湲?
        Strike: PD 怨좉쾶??(Kp=5000) 240Hz ?ㅽ듃由щ컢
        Post: 怨?援щ쫫 愿李?+ ?묒큺 異붿쟻
        """
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

        import pybullet as _p
        env = self._env
        client = self.pb.ClientId
        robot = self.pb.my_robot

        # === Phase 1: Approach (沅ㅼ쟻 ?쒖감 ?ㅽ듃由щ컢 ???뚯씠釉?愿??諛⑹?) ===
        # MoveRobot ?먰봽 諛⑹떇? PD 愿???쒖뼱 怨쇰룄?묐떟?쇰줈 EE媛 ?뚯씠釉??꾨옒濡?
        # 21cm ?댁긽 鍮좎???移섎챸??臾몄젣媛 ?덉쓬. trajectory_planner媛 ?앹꽦??
        # ?덉쟾??寃쎈줈(Home?묨bove?뭃eady)瑜?q_trajectory濡??쒖감 ?ㅽ듃由щ컢?섏뿬 ?닿껐.
        approach_range = phase_indices.get('approach', (0, 0))
        n_approach = approach_range[1] - approach_range[0]
        print(f"    [Approach] {n_approach} pts (streaming)")

        T_ready = trajectory[approach_range[1] - 1]
        ready_pos = T_ready[:3, 3]
        ready_z = T_ready[:3, 2]

        if q_trajectory is not None and len(q_trajectory) >= approach_range[1]:
            # IK 寃利앸맂 approach 沅ㅼ쟻???쒖감 ?ㅽ듃由щ컢
            # 留?50踰덉㎏ ?ъ씤?몃? MoveRobot?쇰줈 蹂대궡怨??섎졃 ?湲?
            step_size = 50
            for idx in range(approach_range[0], approach_range[1], step_size):
                q_target = q_trajectory[min(idx, approach_range[1] - 1)]
                self.pb.MoveRobot(q_target, degree=False)
                time.sleep(0.05)  # 50ms per step
            # 留덉?留??ъ씤??Ready) ?뺥솗???섎졃
            q_ready = q_trajectory[approach_range[1] - 1].copy()
            self.pb.MoveRobot(q_ready, degree=False)
            for _wait in range(500):  # 理쒕? 5珥?
                time.sleep(0.01)
                ee_T = robot._T_end
                pos_err = np.linalg.norm(ee_T[:3, 3] - ready_pos)
                z_dot = np.dot(ee_T[:3, 2], ready_z)
                if pos_err < 0.002 and z_dot > 0.99:
                    break
        else:
            # q_trajectory ?놁쑝硫?fallback: IK 怨꾩궛 ???쒖감 ?대룞
            q_prev = q_i.copy()
            for idx in range(approach_range[0], approach_range[1], 50):
                q_prev = self.ik.solve_step(q_prev, trajectory[idx])
                self.pb.MoveRobot(q_prev, degree=False)
                time.sleep(0.05)
            q_ready = q_prev.copy()
            for _ in range(50):
                q_ready = self.ik.solve_step(q_ready, T_ready)
        for _wait in range(1000):  # 理쒕? 10珥?
            time.sleep(0.01)
            ee_T = robot._T_end
            pos_err = np.linalg.norm(ee_T[:3, 3] - ready_pos)
            z_dot = np.dot(ee_T[:3, 2], ready_z)
            if pos_err < 0.002 and z_dot > 0.99:
                break
        q_i = q_ready
        ee_err = np.linalg.norm(robot._T_end[:3, 3] - ready_pos)
        z_align = np.dot(robot._T_end[:3, 2], ready_z)
        print(f"    [Ready] EE err={ee_err*1000:.1f}mm, z_align={z_align:.4f}")

        # Ready 媛?? ?ㅼ감 ?щ㈃ strike ?ш린
        if ee_err > 0.010:  # 10mm
            print(f"    [SKIP] Ready err too large ({ee_err*1000:.1f}mm > 10mm), aborting strike")
            return False

        # === Phase 2: Strike (PD 怨좉쾶??Kp=5000 ?ㅽ듃由щ컢) ===
        # ?묒큺 異붿쟻 由ъ뀑 (strike 吏곸쟾)
        if env is not None and hasattr(env, 'reset_contact_tracking'):
            env.reset_contact_tracking()
        strike_range = phase_indices.get('strike', (0, 0))
        follow_range = phase_indices.get('follow', (strike_range[1], strike_range[1]))
        T_follow_end = trajectory[min(follow_range[1] - 1, len(trajectory) - 1)]

        sim_dt = self.pb.dt
        actual_speed = strike_speed if strike_speed else 1.0

        from project.trajectory_planner import TrajectoryPlanner
        tp = TrajectoryPlanner()

        # IK + qdot ?ъ쟾 怨꾩궛
        # ?듭떖 ?섏젙: trajectory_planner??dt=0.002(500Hz)濡?沅ㅼ쟻??吏곗?留?
        # _thread_pre(臾쇰━ ?붿쭊)? 240Hz濡??뺣땲??
        # q_trajectory瑜?洹몃?濡??ㅽ듃由щ컢?섎㈃ 1.0m/s濡?爾먯빞??嫄?0.48m/s濡?移섍쾶 ?⑸땲??
        # ?곕씪??strike 援ш컙? 臾댁“嫄?sim_dt(240Hz) 湲곗??쇰줈 ?덈줈 戮묒븘???⑸땲??
        full_traj = tp.plan_constant_speed_linear(
            T_ready, T_follow_end, actual_speed, sim_dt)
        print(f"    [Strike] {len(full_traj)} pts @ {actual_speed:.2f} m/s (resampled to sim_dt)")
        q_prev = q_ready.copy()
        q_traj = []
        for T in full_traj:
            # IK 오차를 줄이기 위해 여러 번 반복하여 수렴시킴
            for _ in range(10):
                q_prev = self.ik.solve_step(q_prev, T)
            q_traj.append(q_prev.copy())

        qdot_traj = []
        for k in range(len(q_traj)):
            if k < len(q_traj) - 1:
                qdot_traj.append((q_traj[k + 1] - q_traj[k]) / sim_dt)
            else:
                qdot_traj.append(np.zeros_like(q_traj[0]))

        # PD 寃뚯씤 洹밸???(strike 以? ??_compute_torque_input??吏곸젒 援먯껜
        original_torque_fn = robot._compute_torque_input

        def _strike_torque_input():
            KP_STRIKE, KD_STRIKE = 5000, 200
            qddot = robot._qddot_des + \
                     KP_STRIKE * (robot._q_des - robot._q) + \
                     KD_STRIKE * (robot._qdot_des - robot._qdot)
            robot._tau = robot._M @ qddot + robot._c + robot._g

        robot._compute_torque_input = _strike_torque_input

        # 沅ㅼ쟻 踰꾪띁 + ?묒큺 ?곹깭
        robot._strike_buf = list(zip(q_traj, qdot_traj))
        robot._strike_idx = 0
        robot._contact_step = -1
        robot._collision_off = False
        robot._initial_ball_recorded = False
        if env is not None:
            for attr in ['_last_actual_ball_velocity', '_last_actual_ball_speed',
                         '_last_actual_ball_angle_deg', '_last_tool_contact_step']:
                if hasattr(env, attr):
                    delattr(env, attr)

        original_thread_pre = self.pb._thread_pre

        def _streaming_thread_pre():
            original_thread_pre()
            if not hasattr(robot, '_strike_buf'):
                return
            # 沅ㅼ쟻 ?ㅽ듃由щ컢
            if robot._strike_idx < len(robot._strike_buf):
                q_des, qdot_des = robot._strike_buf[robot._strike_idx]
                robot._q_des = q_des
                robot._qdot_des = qdot_des
                robot._strike_idx += 1
            # ?꾧뎄-?먮낵 ?묒큺 媛먯? + 1-step ??異⑸룎 ?댁젣 (headless planner? ?숆린??
            if env is not None and not robot._collision_off:
                if robot._contact_step < 0:
                    contacts = _p.getContactPoints(
                        bodyA=env.tool_id, bodyB=env.cue_ball_id,
                        physicsClientId=client)
                    if len(contacts) > 0:
                        robot._contact_step = robot._strike_idx
                elif robot._strike_idx - robot._contact_step >= 10:
                    _p.setCollisionFilterPair(
                        env.tool_id, env.cue_ball_id, -1, -1, 0,
                        physicsClientId=client)
                    robot._collision_off = True
                if (robot._contact_step >= 0 and
                        not robot._initial_ball_recorded and
                        robot._strike_idx - robot._contact_step >= 1):
                    cue_vel, _ = _p.getBaseVelocity(env.cue_ball_id,
                                                    physicsClientId=client)
                    cue_speed = np.linalg.norm(cue_vel[:2])
                    cue_angle = (np.degrees(np.arctan2(cue_vel[1], cue_vel[0]))
                                 if cue_speed > 1e-6 else None)
                    env._last_actual_ball_velocity = [cue_vel[0], cue_vel[1], cue_vel[2]]
                    env._last_actual_ball_speed = cue_speed
                    env._last_actual_ball_angle_deg = cue_angle
                    env._last_tool_contact_step = robot._contact_step
                    robot._initial_ball_recorded = True

            # ?먮낵-紐⑺몴怨?荑좎뀡 ?묒큺 異붿쟻 (?ㅼ쐷 沅ㅼ쟻 以?異⑸룎 媛먯? ?꾨씫 諛⑹?)
            if env is not None:
                if not hasattr(env, '_contact_events'):
                    env._contact_events = []
                    env._contact_cushion_set = set()
                    env._contact_cushion_count = 0
                ball_contacts = _p.getContactPoints(
                    bodyA=env.cue_ball_id, physicsClientId=client)
                cur_cushion = set()
                for bc in ball_contacts:
                    if bc[2] == env.target_ball_id and not getattr(env, '_contact_hit_t1', False):
                        env._contact_hit_t1 = True
                        env._contact_events.append('t1')
                    elif bc[2] == getattr(env, 'ball2_id', -1) and not getattr(env, '_contact_hit_t2', False):
                        env._contact_hit_t2 = True
                        env._contact_events.append('t2')
                    elif hasattr(env, 'cushion_ids') and bc[2] in env.cushion_ids:
                        cur_cushion.add(bc[2])
                new_cushions = cur_cushion - env._contact_cushion_set
                for _ in new_cushions:
                    env._contact_cushion_count += 1
                    env._contact_events.append('c')
                env._contact_cushion_set = cur_cushion

                # 포켓 범위 내 공 감지 및 제거 (실시간)
                if hasattr(env, 'check_and_pocket_balls'):
                    env.check_and_pocket_balls()

        self.pb._thread_pre = _streaming_thread_pre

        # 踰꾪띁 ?뚯쭊 ?湲?
        t0 = time.time()
        timeout = len(q_traj) * sim_dt * 10
        while robot._strike_idx < len(robot._strike_buf):
            if time.time() - t0 > timeout:
                break
            time.sleep(0.005)

        # 吏꾨떒 濡쒓렇
        if env is not None:
            if hasattr(env, '_last_actual_ball_velocity'):
                cue_vel = env._last_actual_ball_velocity
                cue_speed = getattr(env, '_last_actual_ball_speed',
                                    np.linalg.norm(cue_vel[:2]))
                cue_angle = getattr(env, '_last_actual_ball_angle_deg', None)
            else:
                cue_vel, _ = _p.getBaseVelocity(env.cue_ball_id,
                                                physicsClientId=client)
                cue_speed = np.linalg.norm(cue_vel[:2])
                cue_angle = (np.degrees(np.arctan2(cue_vel[1], cue_vel[0]))
                             if cue_speed > 1e-6 else None)
            contact_at = getattr(robot, '_contact_step', -1)
            env._last_actual_ball_velocity = [cue_vel[0], cue_vel[1], cue_vel[2]]
            env._last_actual_ball_speed = cue_speed
            env._last_actual_ball_angle_deg = cue_angle
            env._last_tool_contact_step = contact_at
            print(f"    [DIAG] Contact at step {contact_at}/{len(q_traj)}")
            if cue_angle is None:
                print(f"    [DIAG] Ball vel: [{cue_vel[0]:.3f}, {cue_vel[1]:.3f}] speed={cue_speed:.3f}")
            else:
                print(f"    [DIAG] Ball vel: [{cue_vel[0]:.3f}, {cue_vel[1]:.3f}] "
                      f"speed={cue_speed:.3f}, angle={cue_angle:.1f}deg")

        # strike 肄쒕갚 ?쒓굅 + PD 蹂듭썝
        robot._compute_torque_input = original_torque_fn
        for attr in ['_strike_buf', '_strike_idx',
                      '_contact_step', '_collision_off',
                      '_initial_ball_recorded']:
            if hasattr(robot, attr):
                delattr(robot, attr)
        robot._qdot_des = np.zeros([robot.numJoints, 1])

        # ?묒큺 異붿쟻 ?꾩슜 肄쒕갚 ?ㅼ튂 (240Hz濡?怨?援щ쫫 以??묒큺 媛먯?)
        # 荑좎뀡 + ?곴뎄 ?묒큺 ?쒖꽌 湲곕줉 (3荑좎뀡 洹쒖튃 寃利앹슜)
        if env is not None and not hasattr(env, '_contact_events'):
            env._contact_events = []
            env._contact_cushion_set = set()
            env._contact_cushion_count = 0

        def _contact_tracking_pre():
            original_thread_pre()
            if env is not None:
                try:
                    ball_contacts = _p.getContactPoints(
                        bodyA=env.cue_ball_id, physicsClientId=client)
                    cur_cushion = set()
                    for bc in ball_contacts:
                        if bc[2] == env.target_ball_id and not getattr(env, '_contact_hit_t1', False):
                            env._contact_hit_t1 = True
                            env._contact_events.append('t1')
                        elif bc[2] == getattr(env, 'ball2_id', -1) and not getattr(env, '_contact_hit_t2', False):
                            env._contact_hit_t2 = True
                            env._contact_events.append('t2')
                        elif hasattr(env, 'cushion_ids') and bc[2] in env.cushion_ids:
                            cur_cushion.add(bc[2])
                    # ?덈줈??荑좎뀡 ?묒큺留?湲곕줉
                    new_cushions = cur_cushion - env._contact_cushion_set
                    for _ in new_cushions:
                        env._contact_cushion_count += 1
                        env._contact_events.append('c')
                    env._contact_cushion_set = cur_cushion
                except Exception:
                    pass  # disconnect ???몄텧 ??臾댁떆

        self.pb._thread_pre = _contact_tracking_pre

        print(f"    [DIAG] Hit t1={getattr(env, '_contact_hit_t1', False)}, "
              f"t2={getattr(env, '_contact_hit_t2', False)}")
        if env is not None:
            print(f"    [DIAG] Events so far={getattr(env, '_contact_events', [])}, "
                  f"cushions={getattr(env, '_contact_cushion_count', 0)}")

        # Task Space ?섏쭅 ?곸듅 ?꾪눜 (?뚯씠釉?怨?媛꾩꽠 ?꾨꼍 諛⑹?, 瑗쇱닔 ?쒓굅)
        print("    [Retract] Vertical lift to safe position...")
        T_curr = self.get_current_T()
        T_lift = T_curr.copy()
        T_lift[2, 3] += 0.15

        retract_traj = []
        for i in range(1, 11):
            T_step = T_curr.copy()
            T_step[2, 3] += 0.15 * (i / 10.0)
            retract_traj.append(T_step)

        q_retract = self.ik.solve_trajectory(self.get_current_q(), retract_traj)
        for q_des in q_retract:
            self.pb.MoveRobot(list(q_des.flatten()), degree=False)
            time.sleep(0.02)

        print("    [Retract] Moving home...")
        self.pb.MoveRobot(list(HOME_Q_DEG), degree=True)
        time.sleep(1.0)
        q_i = self.get_current_q()
        self._q_current = q_i.copy()

    def _execute_real(self, trajectory, dt, phase_indices=None, **kwargs):
        """Phase-aware ?ㅼ젣 濡쒕큸 ?ㅽ뻾 ??movel/movej ?⑥씪 ?쒖뼱

        teleop???꾨㈃ ?먭린?섍퀬 movel/movej留??ъ슜:
        - Approach: movel 2??(?곴났 寃쎌쑀 ??Ready ?꾩튂)
        - Strike:   ?⑥씪 movel ??ㅼ쐷 (vel_ratio=100%, 1.0 m/s)
        - Retract:  ?섏쭅 ?곸듅 movel ??Home movej
        """
        if phase_indices is None:
            self._execute_real_uniform(trajectory, dt)
            return

        approach_end = phase_indices['approach'][1]

        # Phase 1: Approach ??movel 2??(?곴났 寃쎌쑀 + Ready)
        print(f"    [Real] Movel Approach...")

        # 1-a. ?곴났 ?덉쟾 寃쎌쑀??(Ready ??20cm)
        idx_above = int(approach_end * 0.6)
        T_above = trajectory[min(idx_above, len(trajectory) - 1)].copy()
        T_above[2, 3] += 0.20
        p_above = self._SE3_to_task_pose(T_above)
        print(f"    [Real] Moving to above waypoint (vel=30%)...")
        self.indy.movel(p_above, vel_ratio=30, acc_ratio=50)
        self._wait_indy()

        # 1-b. Ready ?꾩튂 ?뺣? 吏꾩엯
        T_ready = trajectory[approach_end - 1]
        p_ready = self._SE3_to_task_pose(T_ready)
        print(f"    [Real] Moving to ready position (vel=20%)...")
        self.indy.movel(p_ready, vel_ratio=20, acc_ratio=30)
        self._wait_indy()
        print(f"    [Real] Approach complete")

        # Phase 2: Strike ???⑥씪 movel ??ㅼ쐷 (1.0 m/s)
        follow_end = phase_indices.get('follow', (0, 0))[1]
        if follow_end <= 0:
            follow_end = len(trajectory)
        T_follow_end = trajectory[min(follow_end - 1, len(trajectory) - 1)]
        p_target = self._SE3_to_task_pose(T_follow_end)

        print(f"    [Real] MoveL Strike! vel=100%, acc=100%")
        print(f"    Target: [{p_target[0]:.1f}, {p_target[1]:.1f}, {p_target[2]:.1f}] mm")
        self.indy.movel(p_target, vel_ratio=100, acc_ratio=100)
        self._wait_indy()
        print(f"    [Real] Strike complete")

        # Phase 3: ?섏쭅 ?곸듅 ?꾪눜
        T_lift = T_follow_end.copy()
        T_lift[2, 3] += 0.15
        p_lift = self._SE3_to_task_pose(T_lift)
        print(f"    [Real] Vertical lift (vel=30%)...")
        self.indy.movel(p_lift, vel_ratio=30, acc_ratio=100)
        self._wait_indy()
        print(f"    [Real] Retract complete")

    def _execute_real_uniform(self, trajectory, dt):
        """Phase ?뺣낫 ?녿뒗 沅ㅼ쟻???⑥닚 ?쒖감 movel ?ъ깮"""
        step = max(len(trajectory) // 20, 1)
        for idx in range(0, len(trajectory), step):
            T_des = trajectory[idx]
            p_des = self._SE3_to_task_pose(T_des)
            self.indy.movel(p_des, vel_ratio=30, acc_ratio=50)
            self._wait_indy()
        # 留덉?留????뺥솗???꾨떖
        T_last = trajectory[-1]
        p_last = self._SE3_to_task_pose(T_last)
        self.indy.movel(p_last, vel_ratio=30, acc_ratio=50)
        self._wait_indy()

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

    def _disable_tool_cue_collision(self):
        """?꾪뙥?????꾧뎄?뷀걧蹂?異⑸룎 鍮꾪솢?깊솕

        Headless planner?먯꽌??異⑸룎 10?ㅽ뀦 ???꾧뎄瑜?[0,0,-10]?쇰줈 ?쒓컙?대룞?쒗궡.
        GUI?먯꽌??濡쒕큸???꾪뙥??吏?먯뿉???곸듅?섎뒗 ?숈븞 ?꾧뎄媛 怨듭쓣 諛?대깂.
        ??異⑸룎??鍮꾪솢?깊솕?섏뿬 ?숈씪???④낵 ?ъ꽦.
        """
        if self._env is None or self.pb is None:
            return
        env = self._env
        import pybullet as _p
        client = self.pb.ClientId

        tool_id = getattr(env, 'tool_id', None)
        cue_id = getattr(env, 'cue_ball_id', None)
        if tool_id is not None and cue_id is not None:
            _p.setCollisionFilterPair(tool_id, cue_id, -1, -1,
                                      enableCollision=0, physicsClientId=client)

    def _reenable_tool_cue_collision(self):
        """?ㅼ쓬 ?寃⑹쓣 ?꾪빐 ?꾧뎄?뷀걧蹂?異⑸룎 ?ы솢?깊솕"""
        if self._env is None or self.pb is None:
            return
        env = self._env
        import pybullet as _p
        client = self.pb.ClientId

        tool_id = getattr(env, 'tool_id', None)
        cue_id = getattr(env, 'cue_ball_id', None)
        if tool_id is not None and cue_id is not None:
            _p.setCollisionFilterPair(tool_id, cue_id, -1, -1,
                                      enableCollision=1, physicsClientId=client)
