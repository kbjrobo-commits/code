"""
포켓볼(당구) PyBullet 환경
============================
당구대 + 흰 공 + 목표 공 + 6개 포켓 + 쿠션
로봇은 테이블 바깥(Y−쪽)에 위치하여 실제 사람처럼 타격
"""
import numpy as np
import pybullet as p
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project.config import *


class BilliardsEnvironment:
    """PyBullet 포켓볼 환경"""

    def __init__(self, client_id):
        self.client = client_id
        self.table_id = None
        self.cushion_ids = []
        self.cue_ball_id = None
        self.target_ball_id = None
        self.pocket_markers = []
        self.pocket_positions = []
        self.tool_id = None

    def setup(self, cue_pos=None, target_pos=None):
        """환경 초기화"""
        L = BILLIARD_TABLE_LENGTH
        W = BILLIARD_TABLE_WIDTH
        H = BILLIARD_TABLE_SURFACE_HEIGHT
        CY = BILLIARD_TABLE_CENTER_Y
        ball_h = H + BILLIARD_TABLE_HEIGHT / 2 + BILLIARD_BALL_RADIUS

        self.table_center = np.array([0.5, CY, H])

        if cue_pos is None:
            cue_pos = [0.5, CY - W / 4, ball_h]
        if target_pos is None:
            target_pos = [0.5, CY + W / 8, ball_h]

        self.cue_start_pos = np.array(cue_pos)
        self.target_start_pos = np.array(target_pos)

        self._create_table()
        self._create_cushions()
        self._create_pockets()
        self._create_cue_ball(cue_pos)
        self._create_target_ball(target_pos)

        print(f"[Billiards] Environment setup complete")
        print(f"  Table center: {self.table_center}")
        print(f"  Table Y range: [{CY - W/2:.2f}, {CY + W/2:.2f}]")
        print(f"  Cue ball: {cue_pos}")
        print(f"  Target ball: {target_pos}")

    def _create_table(self):
        L = BILLIARD_TABLE_LENGTH
        W = BILLIARD_TABLE_WIDTH
        H = BILLIARD_TABLE_HEIGHT
        center = self.table_center

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[L/2, W/2, H/2],
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[L/2, W/2, H/2],
            rgbaColor=COLOR_FELT_GREEN,
            physicsClientId=self.client
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[center[0], center[1], center[2]],
            physicsClientId=self.client
        )
        p.changeDynamics(
            self.table_id, -1,
            lateralFriction=BILLIARD_TABLE_FRICTION,
            restitution=0.5,
            physicsClientId=self.client
        )

    def _create_cushions(self):
        L = BILLIARD_TABLE_LENGTH
        W = BILLIARD_TABLE_WIDTH
        CH = BILLIARD_CUSHION_HEIGHT
        TH = BILLIARD_TABLE_HEIGHT
        center = self.table_center
        top_z = center[2] + TH/2 + CH/2
        thickness = 0.04  # 두꺼운 쿠션 — 공이 관통하지 못하게

        cushion_configs = [
            ([center[0], center[1] + W/2 + thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0], center[1] - W/2 - thickness/2, top_z], [L/2, thickness/2, CH/2]),
            ([center[0] - L/2 - thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
            ([center[0] + L/2 + thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ]

        for pos, half_ext in cushion_configs:
            col = p.createCollisionShape(
                shapeType=p.GEOM_BOX, halfExtents=half_ext,
                physicsClientId=self.client
            )
            vis = p.createVisualShape(
                shapeType=p.GEOM_BOX, halfExtents=half_ext,
                rgbaColor=COLOR_BROWN,
                physicsClientId=self.client
            )
            cid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self.client
            )
            p.changeDynamics(cid, -1, restitution=0.8, physicsClientId=self.client)
            self.cushion_ids.append(cid)

    def _create_pockets(self):
        L = BILLIARD_TABLE_LENGTH
        W = BILLIARD_TABLE_WIDTH
        center = self.table_center
        TH = BILLIARD_TABLE_HEIGHT
        z = center[2] + TH/2

        pockets = [
            [center[0] - L/2, center[1] - W/2, z],
            [center[0] - L/2, center[1] + W/2, z],
            [center[0] + L/2, center[1] - W/2, z],
            [center[0] + L/2, center[1] + W/2, z],
            [center[0], center[1] - W/2, z],
            [center[0], center[1] + W/2, z],
        ]

        for pos in pockets:
            self.pocket_positions.append(np.array(pos))
            vis = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=BILLIARD_POCKET_RADIUS,
                length=0.005,
                rgbaColor=COLOR_HOLE_BLACK,
                physicsClientId=self.client
            )
            mid = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self.client
            )
            self.pocket_markers.append(mid)

    def _create_ball(self, position, color, mass=None):
        if mass is None:
            mass = BILLIARD_BALL_MASS
        col = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=BILLIARD_BALL_RADIUS,
            physicsClientId=self.client
        )
        vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=BILLIARD_BALL_RADIUS,
            rgbaColor=color,
            physicsClientId=self.client
        )
        bid = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            physicsClientId=self.client
        )
        p.changeDynamics(
            bid, -1,
            lateralFriction=BILLIARD_BALL_FRICTION,
            restitution=BILLIARD_BALL_RESTITUTION,
            rollingFriction=BILLIARD_BALL_ROLLING_FRICTION,
            spinningFriction=BILLIARD_BALL_SPINNING_FRICTION,
            # CCD: 고속 충돌 시 관통 방지
            ccdSweptSphereRadius=BILLIARD_BALL_RADIUS * 0.5,
            physicsClientId=self.client
        )
        # 터널링 방지를 위한 연속 충돌 감지 활성화
        p.changeDynamics(
            bid, -1,
            contactProcessingThreshold=0,
            physicsClientId=self.client
        )
        return bid

    def _create_cue_ball(self, position):
        self.cue_ball_id = self._create_ball(position, COLOR_WHITE)

    def _create_target_ball(self, position):
        self.target_ball_id = self._create_ball(position, COLOR_RED)

    def get_cue_ball_position(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.cue_ball_id, physicsClientId=self.client
        )
        return np.array(pos)

    def get_target_ball_position(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.target_ball_id, physicsClientId=self.client
        )
        return np.array(pos)

    def get_ball_velocity(self, ball_id):
        vel, _ = p.getBaseVelocity(ball_id, physicsClientId=self.client)
        return np.array(vel)

    def are_balls_stopped(self, threshold=0.005):
        v1 = np.linalg.norm(self.get_ball_velocity(self.cue_ball_id))
        v2 = np.linalg.norm(self.get_ball_velocity(self.target_ball_id))
        return v1 < threshold and v2 < threshold

    def is_pocketed(self, ball_id=None, threshold=None):
        if ball_id is None:
            ball_id = self.target_ball_id
        if threshold is None:
            threshold = BILLIARD_POCKET_RADIUS
        pos, _ = p.getBasePositionAndOrientation(
            ball_id, physicsClientId=self.client
        )
        ball_pos = np.array(pos)
        for pocket_pos in self.pocket_positions:
            dist = np.linalg.norm(ball_pos[:2] - pocket_pos[:2])
            if dist < threshold:
                return True
        return False

    def get_nearest_pocket(self, target_pos=None):
        if target_pos is None:
            target_pos = self.get_target_ball_position()
        min_dist = float('inf')
        nearest = None
        for pp in self.pocket_positions:
            dist = np.linalg.norm(target_pos[:2] - pp[:2])
            if dist < min_dist:
                min_dist = dist
                nearest = pp
        return nearest, min_dist

    def reset_balls(self, cue_pos=None, target_pos=None):
        if cue_pos is None:
            cue_pos = self.cue_start_pos
        if target_pos is None:
            target_pos = self.target_start_pos
        p.resetBasePositionAndOrientation(
            self.cue_ball_id, list(cue_pos), [0,0,0,1],
            physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.cue_ball_id, [0,0,0], [0,0,0],
            physicsClientId=self.client
        )
        p.resetBasePositionAndOrientation(
            self.target_ball_id, list(target_pos), [0,0,0,1],
            physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.target_ball_id, [0,0,0], [0,0,0],
            physicsClientId=self.client
        )

    def wait_balls_stop(self, timeout=10.0, check_interval=0.1):
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self.are_balls_stopped():
                return True
            time.sleep(check_interval)
        return False

    def is_ball_out_of_table(self, ball_id=None):
        """공이 테이블 밖으로 이탈했는지 확인"""
        if ball_id is None:
            ball_id = self.cue_ball_id
        pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=self.client)
        pos = np.array(pos)
        L = BILLIARD_TABLE_LENGTH
        W = BILLIARD_TABLE_WIDTH
        center = self.table_center
        # 테이블 표면 높이보다 아래로 떨어졌거나, 경계 밖이면 이탈
        surface_z = center[2] + BILLIARD_TABLE_HEIGHT / 2
        margin = 0.05
        if pos[2] < surface_z - margin:
            return True
        if abs(pos[0] - center[0]) > L / 2 + margin:
            return True
        if abs(pos[1] - center[1]) > W / 2 + margin:
            return True
        return False

    def attach_compact_tool(self, robot_id, ee_link_index,
                            head_length=None, head_radius=None,
                            head_mass=None, head_restitution=None):
        """EE 끝단에 컴팩트 헤드만 직결 부착 (자루 없음)

        짧고 무거운 헤드 → 안정적이고 임팩트 효과적
        """
        if head_length is None:
            head_length = TOOL_HEAD_LENGTH
        if head_radius is None:
            head_radius = TOOL_HEAD_RADIUS
        if head_mass is None:
            head_mass = TOOL_HEAD_MASS
        if head_restitution is None:
            head_restitution = TOOL_HEAD_RESTITUTION

        head_col = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=head_radius,
            height=head_length,
            physicsClientId=self.client
        )
        head_vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=head_radius,
            length=head_length,
            rgbaColor=COLOR_STEEL,
            physicsClientId=self.client
        )
        head_id = p.createMultiBody(
            baseMass=head_mass,
            baseCollisionShapeIndex=head_col,
            baseVisualShapeIndex=head_vis,
            basePosition=[0, 0, 0],
            physicsClientId=self.client
        )
        p.changeDynamics(
            head_id, -1,
            restitution=head_restitution,
            lateralFriction=0.3,
            physicsClientId=self.client
        )

        # EE z축 방향으로 직결 (헤드 중심이 EE에서 head_length/2만큼 앞)
        cid = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=ee_link_index,
            childBodyUniqueId=head_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, head_length / 2],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client
        )
        # 매우 강한 constraint → 흔들림 최소화
        p.changeConstraint(cid, maxForce=TOOL_CONSTRAINT_FORCE,
                           physicsClientId=self.client)

        self.tool_id = head_id
        self._tool_cid = cid
        return head_id

    def disable_robot_env_collision(self, robot_id):
        """로봇 링크와 당구대/쿠션 간 충돌 비활성화"""
        num_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
        env_bodies = [self.table_id] + self.cushion_ids
        for env_body in env_bodies:
            if env_body is None:
                continue
            for link_idx in range(-1, num_joints):
                p.setCollisionFilterPair(
                    robot_id, env_body, link_idx, -1,
                    enableCollision=0,
                    physicsClientId=self.client
                )

    def disable_tool_env_collision(self):
        """도구-테이블/쿠션 충돌 비활성화"""
        if self.tool_id is None:
            return
        env_bodies = [self.table_id] + self.cushion_ids
        for env_body in env_bodies:
            if env_body is None:
                continue
            p.setCollisionFilterPair(
                self.tool_id, env_body, -1, -1,
                enableCollision=0,
                physicsClientId=self.client
            )
