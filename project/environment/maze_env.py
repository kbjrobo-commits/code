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
        self.obstacle_positions = []
        self.tool_id = None

    def setup(self, cue_pos=None, target_pos=None, ball2_pos=None,
              ball3_pos=None, num_obstacles=5, seed=None,
              obstacle_positions=None, skip_balls=False,
              setup_pockets=False, position_offset=None):
        """환경 초기화

        Args:
            cue_pos: 큐볼 위치 [x, y, z]
            target_pos: 목표구 위치 [x, y, z]
            ball3_pos: 검정 공 위치 [x, y, z] (포켓 데모용)
            num_obstacles: 장애물 개수
            seed: 랜덤 시드
            obstacle_positions: 수동 장애물 좌표 [(x,y), ...]
            skip_balls: True면 공 생성 생략 (캘리브레이션용)
            setup_pockets: True면 6개 포켓(코너4+사이드2) 생성
            position_offset: {'x': float, 'y': float} 캘리브레이션 위치 오프셋
        """
        L = MAZE_TABLE_LENGTH
        W = MAZE_TABLE_WIDTH
        H = MAZE_TABLE_SURFACE_HEIGHT
        CX = MAZE_TABLE_CENTER_X
        CY = MAZE_TABLE_CENTER_Y

        # 캘리브레이션 위치 오프셋 적용 (테이블/쿠션/포켓 전체 이동)
        if position_offset is not None:
            CX += position_offset.get('x', 0.0)
            CY += position_offset.get('y', 0.0)
        TH = MAZE_TABLE_HEIGHT
        ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

        self.table_center = np.array([CX, CY, H])
        self.table_bounds = {
            'x_min': CX - L / 2, 'x_max': CX + L / 2,
            'y_min': CY - W / 2, 'y_max': CY + W / 2
        }
        self._surface_z = H + TH / 2
        self._setup_pockets = setup_pockets

        if cue_pos is None:
            cue_pos = [CX, CY - W / 4, ball_h]
        if target_pos is None:
            target_pos = [CX, CY + W / 8, ball_h]
        if ball2_pos is None:
            ball2_pos = [CX + L / 6, CY, ball_h]

        self.cue_start_pos = np.array(cue_pos)
        self.target_start_pos = np.array(target_pos)
        self.ball2_start_pos = np.array(ball2_pos)

        self._create_table()
        if setup_pockets:
            self._create_cushions_with_pockets()
            self._compute_pocket_positions()
        else:
            self._create_cushions()
            self.pocket_positions = []

        if not skip_balls:
            self._create_cue_ball(cue_pos)
            self._create_target_ball(target_pos)
            self._create_ball2(ball2_pos)
            if ball3_pos is not None:
                self._create_ball3(ball3_pos)

        if obstacle_positions is not None:
            self._place_obstacles_manual(obstacle_positions)
        elif not skip_balls:
            self._place_obstacles_random(num_obstacles, seed)

        mode = "pocket-demo" if setup_pockets else ("table-only" if skip_balls else "2-cushion")
        print(f"[Maze] Environment setup complete ({mode})")
        print(f"  Table: {L}m x {W}m, center Y={CY}")
        if setup_pockets:
            print(f"  Pockets: {len(self.pocket_positions)}")
        if not skip_balls:
            print(f"  Cue (white): {cue_pos}")
            print(f"  Target1 (yellow): {target_pos}")
            print(f"  Target2 (red): {ball2_pos}")
            if ball3_pos is not None:
                print(f"  Target3 (black): {ball3_pos}")
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

    def _create_cushions_with_pockets(self):
        """6개 포켓(코너4+사이드2) 갭이 있는 분할 쿠션 생성.

        각 변의 쿠션을 포켓 위치에서 잘라 2~3개 세그먼트로 분할.
        포켓 갭 크기 = POCKET_RADIUS * 2
        """
        L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
        CH = MAZE_CUSHION_HEIGHT
        TH = MAZE_TABLE_HEIGHT
        center = self.table_center
        top_z = center[2] + TH / 2 + CH / 2
        thickness = 0.03
        gap = POCKET_RADIUS * 2  # 포켓 갭 크기

        CX, CY = center[0], center[1]
        x_min, x_max = CX - L / 2, CX + L / 2
        y_min, y_max = CY - W / 2, CY + W / 2

        def _add_cushion(pos, half_ext):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext,
                                         physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                                      rgbaColor=COLOR_BROWN, physicsClientId=self.client)
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis, basePosition=pos,
                                    physicsClientId=self.client)
            p.changeDynamics(cid, -1, restitution=POCKET_DEMO_CUSHION_RESTITUTION,
                             physicsClientId=self.client)
            self.cushion_ids.append(cid)

        # 상변 (y_max): 코너2곳 갭 → 2세그먼트
        seg_len = (L - 2 * gap) / 2
        y_pos = y_max + thickness / 2
        _add_cushion([x_min + gap + seg_len / 2, y_pos, top_z],
                     [seg_len / 2, thickness / 2, CH / 2])
        _add_cushion([x_max - gap - seg_len / 2, y_pos, top_z],
                     [seg_len / 2, thickness / 2, CH / 2])

        # 하변 (y_min): 코너2곳 갭 → 2세그먼트
        y_pos = y_min - thickness / 2
        _add_cushion([x_min + gap + seg_len / 2, y_pos, top_z],
                     [seg_len / 2, thickness / 2, CH / 2])
        _add_cushion([x_max - gap - seg_len / 2, y_pos, top_z],
                     [seg_len / 2, thickness / 2, CH / 2])

        # 좌변 (x_min): 코너2곳 갭 + 사이드 갭 → 2세그먼트
        seg_len_side = (W - 2 * gap - gap) / 2  # 3갭: 코너2 + 사이드1
        x_pos = x_min - thickness / 2
        _add_cushion([x_pos, y_min + gap + seg_len_side / 2, top_z],
                     [thickness / 2, seg_len_side / 2, CH / 2])
        _add_cushion([x_pos, y_max - gap - seg_len_side / 2, top_z],
                     [thickness / 2, seg_len_side / 2, CH / 2])

        # 우변 (x_max): 코너2곳 갭 + 사이드 갭 → 2세그먼트
        x_pos = x_max + thickness / 2
        _add_cushion([x_pos, y_min + gap + seg_len_side / 2, top_z],
                     [thickness / 2, seg_len_side / 2, CH / 2])
        _add_cushion([x_pos, y_max - gap - seg_len_side / 2, top_z],
                     [thickness / 2, seg_len_side / 2, CH / 2])

    def _compute_pocket_positions(self):
        """6개 포켓 위치 계산 (코너 4 + 사이드 2)."""
        b = self.table_bounds
        sz = self._surface_z
        self.pocket_positions = [
            np.array([b['x_min'], b['y_min'], sz]),  # 좌하 코너
            np.array([b['x_max'], b['y_min'], sz]),  # 우하 코너
            np.array([b['x_min'], b['y_max'], sz]),  # 좌상 코너
            np.array([b['x_max'], b['y_max'], sz]),  # 우상 코너
            np.array([b['x_min'], (b['y_min'] + b['y_max']) / 2, sz]),  # 좌 사이드
            np.array([b['x_max'], (b['y_min'] + b['y_max']) / 2, sz]),  # 우 사이드
        ]

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

    def _create_ball3(self, position):
        self.ball3_id = self._create_ball(position, COLOR_BLACK_BALL)
        self.ball3_start_pos = np.array(position)

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

    def get_ball3_position(self):
        if not hasattr(self, 'ball3_id'):
            return None
        pos, _ = p.getBasePositionAndOrientation(self.ball3_id,
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
        if hasattr(self, 'ball3_id'):
            v4 = np.linalg.norm(self.get_ball_velocity(self.ball3_id))
            return v1 < threshold and v2 < threshold and v3 < threshold and v4 < threshold
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
            # 포켓 범위 내 공 감지 및 제거
            self.check_and_pocket_balls()
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

    def reset_balls(self, cue_pos=None, target_pos=None, ball2_pos=None, ball3_pos=None):
        """공 위치 리셋 — None이면 해당 공을 건드리지 않음"""
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
        if ball3_pos is not None and hasattr(self, 'ball3_id'):
            p.resetBasePositionAndOrientation(self.ball3_id, list(ball3_pos), [0,0,0,1],
                                              physicsClientId=self.client)
            p.resetBaseVelocity(self.ball3_id, [0,0,0], [0,0,0],
                                physicsClientId=self.client)

    def check_and_pocket_balls(self):
        """포켓 범위 내 공을 감지하여 테이블 아래로 제거.

        매 시뮬 스텝 또는 wait_balls_stop에서 호출.
        포켓 반경 내 공을 테이블 아래(-1m)로 이동시켜 사라지게 함.
        """
        if not hasattr(self, 'pocket_positions') or not self.pocket_positions:
            return
        if not hasattr(self, '_pocketed_balls'):
            self._pocketed_balls = set()

        all_balls = self.get_all_ball_ids()
        for bid in all_balls:
            if bid in self._pocketed_balls:
                continue
            pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=self.client)
            for pp in self.pocket_positions:
                dist = np.linalg.norm(np.array(pos[:2]) - pp[:2])
                if dist < POCKET_RADIUS:
                    # 포켓에 들어감 → 테이블 아래로 이동
                    p.resetBasePositionAndOrientation(
                        bid, [pp[0], pp[1], self._surface_z - 1.0],
                        [0, 0, 0, 1], physicsClientId=self.client)
                    p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0],
                                        physicsClientId=self.client)
                    self._pocketed_balls.add(bid)
                    print(f"    [POCKET] Ball {bid} pocketed at ({pp[0]:.3f}, {pp[1]:.3f})")
                    break

    def is_ball_pocketed(self, ball_id):
        """공이 포켓에 들어갔는지 (pocketed set 또는 z 위치 확인)."""
        if hasattr(self, '_pocketed_balls') and ball_id in self._pocketed_balls:
            return True
        pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=self.client)
        return pos[2] < self._surface_z - 0.02

    def which_pocket(self, ball_id):
        """공이 어떤 포켓에 들어갔는지 반환 (-1 = 없음)."""
        if not self.pocket_positions:
            return -1
        pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=self.client)
        for i, pp in enumerate(self.pocket_positions):
            if np.linalg.norm(np.array(pos[:2]) - pp[:2]) < POCKET_RADIUS * 2:
                if pos[2] < self._surface_z - 0.01:
                    return i
        return -1

    def get_all_ball_ids(self):
        """현재 환경의 모든 공 ID 반환 (존재하는 것만)."""
        ids = [self.cue_ball_id, self.target_ball_id]
        if hasattr(self, 'ball2_id'):
            ids.append(self.ball2_id)
        if hasattr(self, 'ball3_id'):
            ids.append(self.ball3_id)
        if hasattr(self, 'ball4_id'):
            ids.append(self.ball4_id)
        if hasattr(self, 'ball5_id'):
            ids.append(self.ball5_id)
        return [bid for bid in ids if bid is not None]

    def setup_lineup(self):
        """Phase 2 초기 배치: 4공을 y축 중심선 위에 일렬 배치.

        배치 순서 (y 증가): 큐볼 → 노랑 → 빨강 → 검정
        """
        CX = MAZE_TABLE_CENTER_X
        CY = MAZE_TABLE_CENTER_Y
        ball_h = self._surface_z + MAZE_BALL_RADIUS + 0.001

        cue_y = CY - LINEUP_CUE_OFFSET - LINEUP_SPACING
        b1_y = CY - LINEUP_SPACING
        b2_y = CY
        b3_y = CY + LINEUP_SPACING

        self.reset_balls(
            cue_pos=[CX, cue_y, ball_h],
            target_pos=[CX, b1_y, ball_h],
            ball2_pos=[CX, b2_y, ball_h],
            ball3_pos=[CX, b3_y, ball_h],
        )
        self.cue_start_pos = np.array([CX, cue_y, ball_h])
        self.target_start_pos = np.array([CX, b1_y, ball_h])
        self.ball2_start_pos = np.array([CX, b2_y, ball_h])
        self.ball3_start_pos = np.array([CX, b3_y, ball_h])
        print(f"[Maze] Lineup setup: cue_y={cue_y:.3f}, b1_y={b1_y:.3f}, b2_y={b2_y:.3f}, b3_y={b3_y:.3f}")

    def setup_postech_o(self):
        """Phase 2 POSTECH 트릭샷 배치.

        "PC'S.TECH" → 트릭샷 → "POSTECH"
        Y+ 방향으로 P→H 순서.

        배치:
        - C-shape (4공): 타원의 왼쪽 절반 (top, left-top, left-bottom, bottom)
        - Trick balls (2공): 아포스트로피 위치 (O와 S 사이)
        - Cue ball: "." 위치 (S와 T 사이)
        - 목표: 큐볼이 trick balls를 동시 타격 → O 우측 완성

        Returns:
            dict: 목표 위치 등 메타데이터
        """
        CX = MAZE_TABLE_CENTER_X   # 0.485
        CY = MAZE_TABLE_CENTER_Y   # 0.1615
        ball_h = self._surface_z + MAZE_BALL_RADIUS + 0.001
        R = MAZE_BALL_RADIUS       # 0.012

        # --- O 타원 파라미터 ---
        # O의 중심 (POSTECH에서 O 위치, Y 방향 P쪽)
        o_cx = CX                  # 테이블 중앙 X
        o_cy = CY - 0.12           # P쪽으로 (Y-)
        # 글자 방향: Y+=읽기 방향, X=글자 높이
        rx = 0.035                 # 타원 X 반경 (글자 높이 방향) — 키움
        ry = 0.045                 # 타원 Y 반경 (읽기 방향) — 키움

        # 6공의 타원 위치
        # cos(a) → X (글자 상하), sin(a) → Y (읽기 방향 좌우)
        # C형 = top(X+), bot(X-), top-left(X+Y-), bot-left(X-Y-) → Y- 쪽이 닫힘
        # O 갭을 X- 쪽(아포스트로피 방향)으로 배치
        # → 큐볼(X+)이 trick balls를 치면 자연스럽게 X- 갭으로 날아감
        c_angles_deg = [0, 60, 300, 270]         # X+/Y+ 쪽 닫힘
        target_angles_deg = [150, 210]            # X- 쪽 열림 (gap)

        c_positions = []
        for a_deg in c_angles_deg:
            a = np.radians(a_deg)
            ox = o_cx + rx * np.cos(a)
            oy = o_cy + ry * np.sin(a)
            c_positions.append([ox, oy, ball_h])

        target_positions = []
        for a_deg in target_angles_deg:
            a = np.radians(a_deg)
            ox = o_cx + rx * np.cos(a)
            oy = o_cy + ry * np.sin(a)
            target_positions.append([ox, oy, ball_h])

        # C형 4공 위치
        c_top, c_rtop, c_rbot, c_bot = c_positions
        # 목표 위치 — trick balls가 도달해야 할 곳 (X- 쪽 갭)
        target_ltop = np.array(target_positions[0])  # 150° → X-, Y+
        target_lbot = np.array(target_positions[1])  # 210° → X-, Y-

        # --- Trick balls (아포스트로피 ' 위치) ---
        # X- 방향 오프셋, Y 방향으로 세로 배치 (유저 승인 위치)
        apos_y = o_cy + ry + 0.04    # O 상단보다 4cm 위 (S쪽으로)
        apos_x = CX - 0.04           # X- 방향으로 4cm 이동
        apos_gap = R * 2.5           # 두 공 사이 간격 (Y방향)
        trick1_pos = [apos_x, apos_y + apos_gap / 2, ball_h]   # 위쪽 공
        trick2_pos = [apos_x, apos_y - apos_gap / 2, ball_h]   # 아래쪽 공

        # --- Cue ball ("." 위치) — X+ 방향으로 오프셋 ---
        cue_y = apos_y + 0.06  # 아포스트로피보다 6cm 위 (T쪽으로)
        cue_pos = [CX + 0.04, cue_y, ball_h]  # X+ 방향으로 4cm

        # --- 공 배치 ---
        # C형 4공: target_ball(C1=top), ball2(C2=l-top), ball3(C3=l-bot), ball4(C4=bot)
        # Trick 2공: ball5(T1=trick-right), cue가 칠 trick-left는... 

        # 기존 공 위치 리셋 또는 새로 생성
        # target_ball_id = C1 (top, yellow)
        # ball2_id = C2 (left-top, red)
        # ball3_id = C3 (left-bottom, black)
        # ball4_id = C4 (bottom, new - orange)
        # ball5_id = T1 (trick ball 1, new - blue)
        # cue_ball_id = cue (white)
        # 기존 target_ball = T2 (trick ball 2) — 여기선 역할 재배치

        # 실제 배치: 공 역할 재정의
        # cue = 큐볼
        # target_ball (yellow) = trick ball 1 (움직일 공)
        # ball2 (red) = trick ball 2 (움직일 공)
        # ball3 (black) = C-top
        # ball4 (orange, 신규) = C-left-top
        # ball5 (blue, 신규) = C-left-bottom
        # ball6 (purple, 신규) = C-bottom

        COLOR_ORANGE = [1.0, 0.5, 0.0, 1.0]
        COLOR_BLUE = [0.0, 0.4, 1.0, 1.0]
        COLOR_PURPLE = [0.6, 0.2, 0.8, 1.0]

        # 큐볼 리셋
        p.resetBasePositionAndOrientation(
            self.cue_ball_id, cue_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.cue_ball_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # Trick ball 1 (yellow = target_ball) → 아포스트로피 우측
        p.resetBasePositionAndOrientation(
            self.target_ball_id, trick1_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.target_ball_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # Trick ball 2 (red = ball2) → 아포스트로피 좌측
        p.resetBasePositionAndOrientation(
            self.ball2_id, trick2_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.ball2_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # C-top (black = ball3)
        p.resetBasePositionAndOrientation(
            self.ball3_id, c_top, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.ball3_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # C-right-top (신규 ball4 = orange)
        if not hasattr(self, 'ball4_id') or self.ball4_id is None:
            self.ball4_id = self._create_ball(c_rtop, COLOR_ORANGE)
        else:
            p.resetBasePositionAndOrientation(
                self.ball4_id, c_rtop, [0, 0, 0, 1], physicsClientId=self.client)
            p.resetBaseVelocity(self.ball4_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # C-right-bottom (신규 ball5 = blue)
        if not hasattr(self, 'ball5_id') or self.ball5_id is None:
            self.ball5_id = self._create_ball(c_rbot, COLOR_BLUE)
        else:
            p.resetBasePositionAndOrientation(
                self.ball5_id, c_rbot, [0, 0, 0, 1], physicsClientId=self.client)
            p.resetBaseVelocity(self.ball5_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # C-bottom (신규 ball6 = purple)
        if not hasattr(self, 'ball6_id') or self.ball6_id is None:
            self.ball6_id = self._create_ball(c_bot, COLOR_PURPLE)
        else:
            p.resetBasePositionAndOrientation(
                self.ball6_id, c_bot, [0, 0, 0, 1], physicsClientId=self.client)
            p.resetBaseVelocity(self.ball6_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        # 시작 위치 저장
        self.cue_start_pos = np.array(cue_pos)
        self.target_start_pos = np.array(trick1_pos)
        self.ball2_start_pos = np.array(trick2_pos)

        # 메타데이터 저장
        self.trick_target1_goal = target_ltop  # trick ball 1 목표 (O 좌상 150°)
        self.trick_target2_goal = target_lbot  # trick ball 2 목표 (O 좌하 210°)
        self.c_ball_ids = [self.ball3_id, self.ball4_id, self.ball5_id, self.ball6_id]
        self.trick_ball_ids = [self.target_ball_id, self.ball2_id]

        # 시각적 마커: 목표 위치에 작은 점 표시
        for goal_pos, color in [(target_ltop, [0, 1, 0, 0.5]), (target_lbot, [0, 1, 0, 0.5])]:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.005,
                                      rgbaColor=color, physicsClientId=self.client)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                              basePosition=list(goal_pos), physicsClientId=self.client)

        print(f"[Maze] POSTECH O setup:")
        print(f"  O center: ({o_cx:.3f}, {o_cy:.3f}), rx={rx}, ry={ry}")
        print(f"  C-shape: top/r-top/r-bot/bot (gap=X- side)")
        print(f"  Trick balls (apostrophe): ({trick1_pos[0]:.3f},{trick1_pos[1]:.3f}), "
              f"({trick2_pos[0]:.3f},{trick2_pos[1]:.3f})")
        print(f"  Cue ball (dot): ({cue_pos[0]:.3f}, {cue_pos[1]:.3f})")
        print(f"  Target goals: L-top ({target_ltop[0]:.3f},{target_ltop[1]:.3f}), "
              f"L-bot ({target_lbot[0]:.3f},{target_lbot[1]:.3f})")

        return {
            'cue_pos': cue_pos,
            'trick1_pos': trick1_pos,
            'trick2_pos': trick2_pos,
            'target1_goal': target_ltop,
            'target2_goal': target_lbot,
            'c_ball_ids': self.c_ball_ids,
            'trick_ball_ids': self.trick_ball_ids,
            'o_center': [o_cx, o_cy],
            'o_radii': [rx, ry],
        }

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
                      + [self.cue_ball_id, self.target_ball_id,
                         getattr(self, 'ball2_id', None),
                         getattr(self, 'ball3_id', None)])
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
                      + [self.target_ball_id, getattr(self, 'ball2_id', None),
                         getattr(self, 'ball3_id', None)])
        for env_body in no_collide:
            if env_body is None:
                continue
            p.setCollisionFilterPair(self.tool_id, env_body, -1, -1,
                                     enableCollision=0, physicsClientId=self.client)
        # ?꾧뎄-?먮낵 異⑸룎? 紐낆떆?곸쑝濡??쒖꽦??(?寃⑹슜)
        if self.cue_ball_id is not None:
            p.setCollisionFilterPair(self.tool_id, self.cue_ball_id, -1, -1,
                                     enableCollision=1, physicsClientId=self.client)
