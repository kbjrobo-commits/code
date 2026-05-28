"""
誘몃줈 ?섍꼍 (PyBullet)
=====================
?먯꽍 洹몃━??+ 臾댁옉???먭린???μ븷臾?+ 荑좎뀡 4硫??밴뎄?
湲고쉷??2.1?? ?댁궛???먯꽍 媛뺤껜 洹몃━??
"""
import numpy as np
import pybullet as p
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project.config import *
from project.physics.cushion_rules import CushionContactTracker


class MazeEnvironment:
    """?먯꽍 洹몃━??+ 臾댁옉???μ븷臾?誘몃줈 ?섍꼍"""

    def __init__(self, client_id):
        self.client = client_id
        self.table_id = None
        self.cushion_ids = []
        self.cue_ball_id = None
        self.target_ball_id = None
        self.obstacle_ids = []
        self.obstacle_positions = []  # [(x, y, radius), ...]
        self.tool_id = None

    def setup(self, cue_pos=None, target_pos=None, ball2_pos=None,
              num_obstacles=5, seed=None, obstacle_positions=None):
        """?섍꼍 珥덇린??

        Args:
            cue_pos: ?먮낵 ?꾩튂 [x, y, z]
            target_pos: 紐⑺몴怨??꾩튂 [x, y, z]
            num_obstacles: 臾댁옉???μ븷臾?媛쒖닔
            seed: ?쒕뜡 ?쒕뱶
            obstacle_positions: ?섎룞 ?μ븷臾?醫뚰몴 [(x,y), ...] ??鍮꾩쟾 ?ㅼ틪 寃곌낵 ?낅젰??
        """
        L = MAZE_TABLE_LENGTH
        W = MAZE_TABLE_WIDTH
        H = MAZE_TABLE_SURFACE_HEIGHT
        CX = MAZE_TABLE_CENTER_X
        CY = MAZE_TABLE_CENTER_Y
        TH = MAZE_TABLE_HEIGHT
        ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

        self.table_center = np.array([CX, CY, H])
        self.table_bounds = {
            'x_min': CX - L / 2, 'x_max': CX + L / 2,
            'y_min': CY - W / 2, 'y_max': CY + W / 2
        }
        self._surface_z = H + TH / 2  # ?뚯씠釉??쒕㈃ z 醫뚰몴

        if cue_pos is None:
            cue_pos = [CX, CY - W / 4, ball_h]
        if target_pos is None:
            target_pos = [CX, CY + W / 8, ball_h]
        # 3踰덉㎏ 怨?(?곕━荑좎뀡: 諛? ?? ??
        if ball2_pos is None:
            ball2_pos = [CX + L / 6, CY, ball_h]

        self.cue_start_pos = np.array(cue_pos)
        self.target_start_pos = np.array(target_pos)
        self.ball2_start_pos = np.array(ball2_pos)

        self._create_table()
        self._create_cushions()
        self._create_cue_ball(cue_pos)
        self._create_target_ball(target_pos)
        self._create_ball2(ball2_pos)

        if obstacle_positions is not None:
            self._place_obstacles_manual(obstacle_positions)
        else:
            self._place_obstacles_random(num_obstacles, seed)

        print(f"[Maze] Environment setup complete (2-cushion)")
        print(f"  Table: {L}m x {W}m, center Y={CY}")
        print(f"  Cue (white): {cue_pos}")
        print(f"  Target1 (yellow): {target_pos}")
        print(f"  Target2 (red): {ball2_pos}")
        print(f"  Obstacles: {len(self.obstacle_positions)}")

    # ??? ?뚯씠釉?& 荑좎뀡 ?????????????????????????????????

    def _create_table(self):
        L, W, TH = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH, MAZE_TABLE_HEIGHT
        center = self.table_center
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2],
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2],
                                  rgbaColor=COLOR_FELT_GREEN, physicsClientId=self.client)
        self.table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis,
                                          basePosition=[center[0], center[1], center[2]],
                                          physicsClientId=self.client)
        p.changeDynamics(self.table_id, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=0.5, physicsClientId=self.client)

    def _create_cushions(self):
        L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
        CH = MAZE_CUSHION_HEIGHT
        TH = MAZE_TABLE_HEIGHT
        center = self.table_center
        top_z = center[2] + TH / 2 + CH / 2
        thickness = 0.03

        configs = [
            ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
            ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ]
        for pos, half_ext in configs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                         physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                                      rgbaColor=COLOR_BROWN, physicsClientId=self.client)
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis, basePosition=pos,
                                    physicsClientId=self.client)
            p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION,
                             physicsClientId=self.client)
            self.cushion_ids.append(cid)

    # ??? 怨???????????????????????????????????????????

    def _create_ball(self, position, color, mass=MAZE_BALL_MASS):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS,
                                  rgbaColor=color, physicsClientId=self.client)
        bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis, basePosition=position,
                                physicsClientId=self.client)
        p.changeDynamics(bid, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=MAZE_BALL_RESTITUTION,
                         rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                         spinningFriction=0.02,
                         ccdSweptSphereRadius=MAZE_BALL_RADIUS * 0.5,
                         contactProcessingThreshold=0,
                         physicsClientId=self.client)
        return bid

    def _create_cue_ball(self, position):
        self.cue_ball_id = self._create_ball(position, COLOR_WHITE)

    def _create_target_ball(self, position):
        self.target_ball_id = self._create_ball(position, COLOR_YELLOW)

    def _create_ball2(self, position):
        self.ball2_id = self._create_ball(position, COLOR_RED)

    # ??? ?μ븷臾?諛곗튂 ?????????????????????????????????

    def _place_obstacles_random(self, n, seed=None):
        """5cm 洹몃━?쒖뿉 ?ㅻ깄?섏뿬 臾댁옉???μ븷臾?諛곗튂"""
        if seed is not None:
            np.random.seed(seed)

        b = self.table_bounds
        spacing = MAZE_GRID_SPACING

        # 洹몃━???ъ씤???앹꽦
        xs = np.arange(b['x_min'] + spacing, b['x_max'], spacing)
        ys = np.arange(b['y_min'] + spacing, b['y_max'], spacing)
        grid_points = [(x, y) for x in xs for y in ys]

        # 怨?洹쇱쿂(諛섍꼍 8cm) ?쒖쇅
        cue_2d = self.cue_start_pos[:2]
        tgt_2d = self.target_start_pos[:2]
        valid = []
        for gx, gy in grid_points:
            if np.linalg.norm([gx - cue_2d[0], gy - cue_2d[1]]) < 0.08:
                continue
            if np.linalg.norm([gx - tgt_2d[0], gy - tgt_2d[1]]) < 0.08:
                continue
            valid.append((gx, gy))

        # ?쒕뜡 ?좏깮
        n = min(n, len(valid))
        chosen = [valid[i] for i in np.random.choice(len(valid), n, replace=False)]
        self._place_obstacles_at(chosen)

    def _place_obstacles_manual(self, positions):
        """?섎룞 醫뚰몴 湲곕컲 ?μ븷臾?諛곗튂 (鍮꾩쟾 ?ㅼ틪 寃곌낵 ?낅젰??"""
        self._place_obstacles_at(positions)

    def _place_obstacles_at(self, positions_2d):
        """二쇱뼱吏?2D 醫뚰몴???먭린???μ븷臾??앹꽦"""
        r = MAZE_OBSTACLE_RADIUS
        h = MAZE_OBSTACLE_HEIGHT
        TH = MAZE_TABLE_HEIGHT
        z = self.table_center[2] + TH / 2 + h / 2

        for (x, y) in positions_2d:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                      rgbaColor=COLOR_OBSTACLE, physicsClientId=self.client)
            oid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, z],
                                    physicsClientId=self.client)
            p.changeDynamics(oid, -1, restitution=0.5, lateralFriction=0.3,
                             physicsClientId=self.client)
            self.obstacle_ids.append(oid)
            self.obstacle_positions.append((x, y, r))

    # ??? ?쇱꽌 ?명꽣?섏씠??(Perception?? ?????????????

    def get_cue_ball_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.cue_ball_id,
                                                  physicsClientId=self.client)
        return np.array(pos)

    def get_target_ball_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.target_ball_id,
                                                  physicsClientId=self.client)
        return np.array(pos)

    def get_ball2_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball2_id,
                                                  physicsClientId=self.client)
        return np.array(pos)

    def get_ball_velocity(self, ball_id):
        vel, _ = p.getBaseVelocity(ball_id, physicsClientId=self.client)
        return np.array(vel)

    def get_obstacle_positions(self):
        """Return obstacle positions for the planner."""
        return list(self.obstacle_positions)

    def are_balls_stopped(self, threshold=0.005):
        v1 = np.linalg.norm(self.get_ball_velocity(self.cue_ball_id))
        v2 = np.linalg.norm(self.get_ball_velocity(self.target_ball_id))
        v3 = np.linalg.norm(self.get_ball_velocity(self.ball2_id))
        return v1 < threshold and v2 < threshold and v3 < threshold

    def is_ball_out_of_table(self, ball_id):
        """怨듭씠 ?뚯씠釉?踰붿쐞瑜?踰쀬뼱?щ뒗吏 ?뺤씤"""
        pos, _ = p.getBasePositionAndOrientation(ball_id,
                                                  physicsClientId=self.client)
        b = self.table_bounds
        margin = 0.05  # ?쎄컙???ъ쑀
        if pos[0] < b['x_min'] - margin or pos[0] > b['x_max'] + margin:
            return True
        if pos[1] < b['y_min'] - margin or pos[1] > b['y_max'] + margin:
            return True
        if pos[2] < self._surface_z - 0.05:  # ?뚯씠釉??꾨옒濡??⑥뼱吏?
            return True
        return False

    def is_target_hit(self, threshold=0.01):
        """Return whether the cue ball contacted both target balls."""
        return getattr(self, '_contact_hit_t1', False) and \
               getattr(self, '_contact_hit_t2', False)

    def wait_balls_stop(self, timeout=10.0, check_interval=0.1):
        """Wait until balls stop while tracking contacts."""
        import time
        tracker = getattr(self, '_contact_tracker', None)
        if tracker is None:
            tracker = CushionContactTracker(
                self.target_ball_id,
                getattr(self, 'ball2_id', None),
                self.cushion_ids,
            )
            tracker.hit_t1 = getattr(self, '_contact_hit_t1', False)
            tracker.hit_t2 = getattr(self, '_contact_hit_t2', False)
            tracker.events = list(getattr(self, '_contact_events', []))
            tracker.cushion_count = int(getattr(self, '_contact_cushion_count', 0))
            tracker._prev_cushions = set(getattr(self, '_contact_cushion_set', set()))
            self._contact_tracker = tracker
        legacy_events = list(getattr(self, '_contact_events', []))
        if len(legacy_events) >= len(tracker.events):
            tracker.events = legacy_events
        tracker.hit_t1 = tracker.hit_t1 or getattr(self, '_contact_hit_t1', False)
        tracker.hit_t2 = tracker.hit_t2 or getattr(self, '_contact_hit_t2', False)
        tracker.cushion_count = max(
            tracker.cushion_count,
            int(getattr(self, '_contact_cushion_count', 0)),
        )
        tracker._prev_cushions = set(getattr(self, '_contact_cushion_set', tracker._prev_cushions))
        start = time.time()
        while time.time() - start < timeout:
            # 접촉 추적 (240Hz _contact_tracking_pre의 보조 — 놓친 것만 보충)
            contacts = p.getContactPoints(bodyA=self.cue_ball_id,
                                          physicsClientId=self.client)
            tracker.update_from_contacts(contacts)
            # 240Hz pre-hook이 이미 env에 쓴 값과 tracker 값을 병합
            # (둘 중 더 많은 이벤트를 유지)
            if len(tracker.events) > len(getattr(self, '_contact_events', [])):
                self._contact_events = list(tracker.events)
                self._contact_hit_t1 = tracker.hit_t1
                self._contact_hit_t2 = tracker.hit_t2
                self._contact_cushion_count = tracker.cushion_count
                self._contact_cushion_set = set(tracker._prev_cushions)
                self._cushion_contacts = tracker.cushion_count
            else:
                # 240Hz hook이 더 많이 잡았으면 tracker에 반영
                tracker.events = list(getattr(self, '_contact_events', []))
                tracker.hit_t1 = getattr(self, '_contact_hit_t1', False)
                tracker.hit_t2 = getattr(self, '_contact_hit_t2', False)
                tracker.cushion_count = max(tracker.cushion_count,
                                            int(getattr(self, '_contact_cushion_count', 0)))
            if self.are_balls_stopped():
                return True
            time.sleep(check_interval)
        return False

    def _sync_contact_tracker_state(self):
        tracker = getattr(self, '_contact_tracker', None)
        if tracker is None:
            return
        self._contact_hit_t1 = tracker.hit_t1
        self._contact_hit_t2 = tracker.hit_t2
        self._contact_events = list(tracker.events)
        self._contact_cushion_count = tracker.cushion_count
        self._contact_cushion_set = set(tracker._prev_cushions)
        self._cushion_contacts = tracker.cushion_count

    def reset_contact_tracking(self):
        """???쇱슫???쒖옉 ???묒큺 異붿쟻 由ъ뀑"""
        self._contact_hit_t1 = False
        self._contact_hit_t2 = False
        self._cushion_contacts = 0
        self._contact_events = []
        self._contact_cushion_set = set()
        self._contact_cushion_count = 0
        self._contact_tracker = CushionContactTracker(
            self.target_ball_id,
            getattr(self, 'ball2_id', None),
            self.cushion_ids,
        )

    def reset_balls(self, cue_pos=None, target_pos=None, ball2_pos=None):
        """怨??꾩튂 由ъ뀑 ??None?대㈃ ?대떦 怨듭? 嫄대뱶由ъ? ?딆쓬"""
        if cue_pos is not None:
            p.resetBasePositionAndOrientation(self.cue_ball_id, list(cue_pos), [0,0,0,1],
                                              physicsClientId=self.client)
            p.resetBaseVelocity(self.cue_ball_id, [0,0,0], [0,0,0],
                                physicsClientId=self.client)
        if target_pos is not None:
            p.resetBasePositionAndOrientation(self.target_ball_id, list(target_pos), [0,0,0,1],
                                              physicsClientId=self.client)
            p.resetBaseVelocity(self.target_ball_id, [0,0,0], [0,0,0],
                                physicsClientId=self.client)
        if ball2_pos is not None and hasattr(self, 'ball2_id'):
            p.resetBasePositionAndOrientation(self.ball2_id, list(ball2_pos), [0,0,0,1],
                                              physicsClientId=self.client)
            p.resetBaseVelocity(self.ball2_id, [0,0,0], [0,0,0],
                                physicsClientId=self.client)

    # ??? ?꾧뎄 & 異⑸룎 愿由??????????????????????????????

    def attach_compact_tool(self, robot_id, ee_link_index,
                            head_length=None, head_radius=None,
                            head_mass=None, head_restitution=None):
        """ㄴ자 큐팁 도구를 EE에 부착.

        도구 형상:
          EE
           |  (TOOL_VERTICAL_DROP = 60mm 수직 하강)
           |
           └──● (TOOL_HORIZONTAL_EXT = 30mm 수평 연장, 끝에 큐팁)

        PyBullet에서는 큐팁(작은 실린더)만 충돌체로 모델링하고,
        parentFramePosition으로 EE 대비 오프셋을 설정합니다.
        큐팁 실린더의 축은 EE의 z축(수직)과 직교하는 수평 방향입니다.
        """
        tip_radius = TOOL_TIP_RADIUS
        tip_length = TOOL_TIP_LENGTH
        if head_mass is None: head_mass = TOOL_HEAD_MASS
        if head_restitution is None: head_restitution = TOOL_HEAD_RESTITUTION

        # 큐팁 충돌체 (작은 수평 실린더)
        head_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=tip_radius,
                                          height=tip_length, physicsClientId=self.client)
        head_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=tip_radius,
                                       length=tip_length, rgbaColor=COLOR_STEEL,
                                       physicsClientId=self.client)
        head_id = p.createMultiBody(baseMass=head_mass, baseCollisionShapeIndex=head_col,
                                    baseVisualShapeIndex=head_vis,
                                    basePosition=[0, 0, 0], physicsClientId=self.client)
        p.changeDynamics(head_id, -1, restitution=head_restitution,
                         lateralFriction=0.3, physicsClientId=self.client)

        # ㄴ자 오프셋: EE 로컬 프레임 (Z=아래, X=타격방향)
        # 수직 부분 = EE +Z (아래로 60mm), 수평 부분 = EE +X (앞으로 30mm)
        # TOOL_YAW_OFFSET: 실제 도구 장착 z축 회전 오프셋 → 위치만 반영
        # 큐팁 자세는 항상 EE x축(타격방향) 수직 → 공을 정면으로 타격
        yaw = TOOL_YAW_OFFSET
        tool_x = TOOL_HORIZONTAL_EXT * np.cos(yaw)
        tool_y = TOOL_HORIZONTAL_EXT * np.sin(yaw)
        tip_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])  # 팁 자세: 항상 strike_dir 수직
        cid = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_link_index,
                                 childBodyUniqueId=head_id, childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                 parentFramePosition=[tool_x, tool_y, TOOL_VERTICAL_DROP],
                                 parentFrameOrientation=tip_orn,
                                 childFramePosition=[0, 0, 0],
                                 physicsClientId=self.client)
        p.changeConstraint(cid, maxForce=TOOL_CONSTRAINT_FORCE, physicsClientId=self.client)
        self.tool_id = head_id
        self._tool_cid = cid
        return head_id

    def disable_robot_env_collision(self, robot_id):
        """濡쒕큸 留곹겕? ?뚯씠釉?荑좎뀡/?μ븷臾?怨?媛?異⑸룎 鍮꾪솢?깊솕

        濡쒕큸 紐몄껜媛 ?묎렐 ??怨듭씠???μ븷臾쇱쓣 諛?대쾭由щ뒗 寃껋쓣 諛⑹?.
        ?꾧뎄-?먮낵 異⑸룎留?蹂꾨룄濡??좎???
        """
        num_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
        env_bodies = ([self.table_id] + self.cushion_ids + self.obstacle_ids
                      + [self.cue_ball_id, self.target_ball_id, self.ball2_id])
        for env_body in env_bodies:
            if env_body is None:
                continue
            for link_idx in range(-1, num_joints):
                p.setCollisionFilterPair(robot_id, env_body, link_idx, -1,
                                         enableCollision=0, physicsClientId=self.client)

    def disable_tool_env_collision(self):
        """?꾧뎄 異⑸룎 ?ㅼ젙: ?먮낵留?異⑸룎 ?좎?, ?섎㉧吏 ?꾨? 鍮꾪솢?깊솕"""
        if self.tool_id is None:
            return
        # ?뚯씠釉?荑좎뀡/?μ븷臾?紐⑺몴怨듦낵 異⑸룎 鍮꾪솢?깊솕
        no_collide = ([self.table_id] + self.cushion_ids + self.obstacle_ids
                      + [self.target_ball_id, self.ball2_id])
        for env_body in no_collide:
            if env_body is None:
                continue
            p.setCollisionFilterPair(self.tool_id, env_body, -1, -1,
                                     enableCollision=0, physicsClientId=self.client)
        # ?꾧뎄-?먮낵 異⑸룎? 紐낆떆?곸쑝濡??쒖꽦??(?寃⑹슜)
        if self.cue_ball_id is not None:
            p.setCollisionFilterPair(self.tool_id, self.cue_ball_id, -1, -1,
                                     enableCollision=1, physicsClientId=self.client)
