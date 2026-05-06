"""
미로 환경 (PyBullet)
=====================
자석 그리드 + 무작위 원기둥 장애물 + 쿠션 4면 당구대
기획서 2.1절: 이산화 자석 강체 그리드
"""
import numpy as np
import pybullet as p
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project.config import *


class MazeEnvironment:
    """자석 그리드 + 무작위 장애물 미로 환경"""

    def __init__(self, client_id):
        self.client = client_id
        self.table_id = None
        self.cushion_ids = []
        self.cue_ball_id = None
        self.target_ball_id = None
        self.obstacle_ids = []
        self.obstacle_positions = []  # [(x, y, radius), ...]
        self.tool_id = None

    def setup(self, cue_pos=None, target_pos=None,
              num_obstacles=5, seed=None, obstacle_positions=None):
        """환경 초기화

        Args:
            cue_pos: 큐볼 위치 [x, y, z]
            target_pos: 목표공 위치 [x, y, z]
            num_obstacles: 무작위 장애물 개수
            seed: 랜덤 시드
            obstacle_positions: 수동 장애물 좌표 [(x,y), ...] — 비전 스캔 결과 입력용
        """
        L = MAZE_TABLE_LENGTH
        W = MAZE_TABLE_WIDTH
        H = MAZE_TABLE_SURFACE_HEIGHT
        CY = MAZE_TABLE_CENTER_Y
        TH = MAZE_TABLE_HEIGHT
        ball_h = H + TH / 2 + MAZE_BALL_RADIUS + 0.001

        self.table_center = np.array([0.5, CY, H])
        self.table_bounds = {
            'x_min': 0.5 - L / 2, 'x_max': 0.5 + L / 2,
            'y_min': CY - W / 2, 'y_max': CY + W / 2
        }

        if cue_pos is None:
            cue_pos = [0.5, CY - W / 4, ball_h]
        if target_pos is None:
            target_pos = [0.5, CY + W / 8, ball_h]

        self.cue_start_pos = np.array(cue_pos)
        self.target_start_pos = np.array(target_pos)

        self._create_table()
        self._create_cushions()
        self._create_cue_ball(cue_pos)
        self._create_target_ball(target_pos)

        if obstacle_positions is not None:
            self._place_obstacles_manual(obstacle_positions)
        else:
            self._place_obstacles_random(num_obstacles, seed)

        print(f"[Maze] Environment setup complete")
        print(f"  Table: {L}m × {W}m, center Y={CY}")
        print(f"  Cue ball: {cue_pos}")
        print(f"  Target: {target_pos}")
        print(f"  Obstacles: {len(self.obstacle_positions)}")

    # ─── 테이블 & 쿠션 ─────────────────────────────────

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
        thickness = 0.04

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

    # ─── 공 ──────────────────────────────────────────

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
        self.target_ball_id = self._create_ball(position, COLOR_RED)

    # ─── 장애물 배치 ─────────────────────────────────

    def _place_obstacles_random(self, n, seed=None):
        """5cm 그리드에 스냅하여 무작위 장애물 배치"""
        if seed is not None:
            np.random.seed(seed)

        b = self.table_bounds
        spacing = MAZE_GRID_SPACING

        # 그리드 포인트 생성
        xs = np.arange(b['x_min'] + spacing, b['x_max'], spacing)
        ys = np.arange(b['y_min'] + spacing, b['y_max'], spacing)
        grid_points = [(x, y) for x in xs for y in ys]

        # 공 근처(반경 8cm) 제외
        cue_2d = self.cue_start_pos[:2]
        tgt_2d = self.target_start_pos[:2]
        valid = []
        for gx, gy in grid_points:
            if np.linalg.norm([gx - cue_2d[0], gy - cue_2d[1]]) < 0.08:
                continue
            if np.linalg.norm([gx - tgt_2d[0], gy - tgt_2d[1]]) < 0.08:
                continue
            valid.append((gx, gy))

        # 랜덤 선택
        n = min(n, len(valid))
        chosen = [valid[i] for i in np.random.choice(len(valid), n, replace=False)]
        self._place_obstacles_at(chosen)

    def _place_obstacles_manual(self, positions):
        """수동 좌표 기반 장애물 배치 (비전 스캔 결과 입력용)"""
        self._place_obstacles_at(positions)

    def _place_obstacles_at(self, positions_2d):
        """주어진 2D 좌표에 원기둥 장애물 생성"""
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

    # ─── 센서 인터페이스 (Perception용) ─────────────

    def get_cue_ball_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.cue_ball_id,
                                                  physicsClientId=self.client)
        return np.array(pos)

    def get_target_ball_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.target_ball_id,
                                                  physicsClientId=self.client)
        return np.array(pos)

    def get_ball_velocity(self, ball_id):
        vel, _ = p.getBaseVelocity(ball_id, physicsClientId=self.client)
        return np.array(vel)

    def get_obstacle_positions(self):
        """장애물 좌표 리스트 반환 — 탐색기에 전달용"""
        return list(self.obstacle_positions)

    def are_balls_stopped(self, threshold=0.005):
        v1 = np.linalg.norm(self.get_ball_velocity(self.cue_ball_id))
        v2 = np.linalg.norm(self.get_ball_velocity(self.target_ball_id))
        return v1 < threshold and v2 < threshold

    def is_target_hit(self, threshold=0.01):
        """큐볼이 목표공에 충돌했는지 (근접 판정)"""
        cue = self.get_cue_ball_position()
        tgt = self.get_target_ball_position()
        dist = np.linalg.norm(cue[:2] - tgt[:2])
        return dist < MAZE_BALL_RADIUS * 2 + threshold

    def wait_balls_stop(self, timeout=10.0, check_interval=0.1):
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self.are_balls_stopped():
                return True
            time.sleep(check_interval)
        return False

    def reset_balls(self, cue_pos=None, target_pos=None):
        if cue_pos is None:
            cue_pos = self.cue_start_pos
        if target_pos is None:
            target_pos = self.target_start_pos
        p.resetBasePositionAndOrientation(self.cue_ball_id, list(cue_pos), [0,0,0,1],
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.cue_ball_id, [0,0,0], [0,0,0],
                            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.target_ball_id, list(target_pos), [0,0,0,1],
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.target_ball_id, [0,0,0], [0,0,0],
                            physicsClientId=self.client)

    # ─── 도구 & 충돌 관리 ─────────────────────────────

    def attach_compact_tool(self, robot_id, ee_link_index,
                            head_length=None, head_radius=None,
                            head_mass=None, head_restitution=None):
        """EE 끝단에 컴팩트 헤드 부착"""
        if head_length is None: head_length = TOOL_HEAD_LENGTH
        if head_radius is None: head_radius = TOOL_HEAD_RADIUS
        if head_mass is None: head_mass = TOOL_HEAD_MASS
        if head_restitution is None: head_restitution = TOOL_HEAD_RESTITUTION

        head_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=head_radius,
                                          height=head_length, physicsClientId=self.client)
        head_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=head_radius,
                                       length=head_length, rgbaColor=COLOR_STEEL,
                                       physicsClientId=self.client)
        head_id = p.createMultiBody(baseMass=head_mass, baseCollisionShapeIndex=head_col,
                                    baseVisualShapeIndex=head_vis,
                                    basePosition=[0, 0, 0], physicsClientId=self.client)
        p.changeDynamics(head_id, -1, restitution=head_restitution,
                         lateralFriction=0.3, physicsClientId=self.client)
        cid = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_link_index,
                                 childBodyUniqueId=head_id, childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                 parentFramePosition=[0, 0, head_length / 2],
                                 childFramePosition=[0, 0, 0],
                                 physicsClientId=self.client)
        p.changeConstraint(cid, maxForce=TOOL_CONSTRAINT_FORCE, physicsClientId=self.client)
        self.tool_id = head_id
        self._tool_cid = cid
        return head_id

    def disable_robot_env_collision(self, robot_id):
        """로봇 링크와 테이블/쿠션/장애물/공 간 충돌 비활성화

        로봇 몸체가 접근 시 공이나 장애물을 밀어버리는 것을 방지.
        도구-큐볼 충돌만 별도로 유지됨.
        """
        num_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
        env_bodies = ([self.table_id] + self.cushion_ids + self.obstacle_ids
                      + [self.cue_ball_id, self.target_ball_id])
        for env_body in env_bodies:
            if env_body is None:
                continue
            for link_idx in range(-1, num_joints):
                p.setCollisionFilterPair(robot_id, env_body, link_idx, -1,
                                         enableCollision=0, physicsClientId=self.client)

    def disable_tool_env_collision(self):
        """도구 충돌 설정: 큐볼만 충돌 유지, 나머지 전부 비활성화"""
        if self.tool_id is None:
            return
        # 테이블/쿠션/장애물/목표공과 충돌 비활성화
        no_collide = ([self.table_id] + self.cushion_ids + self.obstacle_ids
                      + [self.target_ball_id])
        for env_body in no_collide:
            if env_body is None:
                continue
            p.setCollisionFilterPair(self.tool_id, env_body, -1, -1,
                                     enableCollision=0, physicsClientId=self.client)
        # 도구-큐볼 충돌은 명시적으로 활성화 (타격용)
        if self.cue_ball_id is not None:
            p.setCollisionFilterPair(self.tool_id, self.cue_ball_id, -1, -1,
                                     enableCollision=1, physicsClientId=self.client)
