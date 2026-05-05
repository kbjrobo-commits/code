"""
미니골프 PyBullet 환경
========================
3D 굴곡 지형 메쉬 + 골프공 + 홀 컵
"""
import numpy as np
import pybullet as p
import pybullet_data
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project.config import *


def _generate_terrain_heightfield(size_x, size_y, resolution, seed=42, hole_pos=None, terrain_offset=None):
    """굴곡 지형 높이 맵 생성 (부드러운 언덕/계곡)

    Returns:
        heights: (resolution, resolution) ndarray
    """
    np.random.seed(seed)
    rows = resolution
    cols = resolution

    # 여러 주파수의 사인파 조합으로 자연스러운 굴곡 생성
    x = np.linspace(0, size_x, cols)
    y = np.linspace(0, size_y, rows)
    X, Y = np.meshgrid(x, y)

    heights = np.zeros_like(X)

    # 큰 언덕
    heights += 0.015 * np.sin(2 * np.pi * X / size_x * 1.5) * np.cos(2 * np.pi * Y / size_y * 1.0)
    # 중간 굴곡
    heights += 0.008 * np.sin(2 * np.pi * X / size_x * 3.0 + 0.5) * np.sin(2 * np.pi * Y / size_y * 2.0)
    # 작은 굴곡
    heights += 0.004 * np.cos(2 * np.pi * X / size_x * 5.0) * np.cos(2 * np.pi * Y / size_y * 4.0 + 1.0)

    # 가장자리를 약간 높게 (경계 역할)
    edge_mask_x = np.minimum(X / (size_x * 0.1), (size_x - X) / (size_x * 0.1))
    edge_mask_y = np.minimum(Y / (size_y * 0.1), (size_y - Y) / (size_y * 0.1))
    edge_mask = np.clip(np.minimum(edge_mask_x, edge_mask_y), 0, 1)
    heights += 0.02 * (1 - edge_mask)

    if hole_pos is not None and terrain_offset is not None:
        # 월드 좌표의 hole_pos를 메쉬 로컬 좌표계로 변환
        # World X = X_mesh_local + terrain_offset[0]
        # X_mesh_local = X - (size_x / 2) -> World X = X - (size_x / 2) + terrain_offset[0]
        hx = hole_pos[0] + (size_x / 2) - terrain_offset[0]
        hy = hole_pos[1] + (size_y / 2) - terrain_offset[1]

        # 홀 중심으로부터의 거리 계산
        dist = np.sqrt((X - hx)**2 + (Y - hy)**2)

        # 구멍 반경 내에 있는 정점들의 Z값을 지하(-0.05m)로 파냄
        hole_mask = dist < MINIGOLF_HOLE_RADIUS
        heights[hole_mask] = -0.05

    return heights


def _create_terrain_mesh_obj(heights, size_x, size_y, filename):
    """높이 맵을 OBJ 메쉬 파일로 변환"""
    rows, cols = heights.shape

    vertices = []
    faces = []

    dx = size_x / (cols - 1)
    dy = size_y / (rows - 1)

    # 오프셋: 메쉬 중심이 원점이 되도록
    offset_x = size_x / 2
    offset_y = size_y / 2

    for r in range(rows):
        for c in range(cols):
            x = c * dx - offset_x
            y = r * dy - offset_y
            z = heights[r, c]
            vertices.append((x, y, z))

    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c + 1
            v10 = (r + 1) * cols + c + 1
            v01 = r * cols + (c + 1) + 1
            v11 = (r + 1) * cols + (c + 1) + 1
            # Winding order: 반시계→시계로 변경하여 노멀이 위(+Z)를 향하도록
            faces.append((v00, v01, v10))
            faces.append((v10, v01, v11))

    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    return filename


class MiniGolfEnvironment:
    """PyBullet 미니골프 환경

    - 3D 굴곡 지형 (메쉬)
    - 골프공
    - 홀 컵 (시각적 마커)
    """

    def __init__(self, client_id):
        self.client = client_id
        self.terrain_id = None
        self.ball_id = None
        self.hole_marker_id = None
        self.hole_pos = None
        self.ball_start_pos = None
        self._temp_files = []

    def setup(self, ball_pos=None, hole_pos=None, terrain_seed=42):
        """환경 초기화

        Args:
            ball_pos: 공 초기 위치 [x, y, z] (None이면 기본값)
            hole_pos: 홀 위치 [x, y, z] (None이면 기본값)
            terrain_seed: 지형 생성 시드
        """
        size_x, size_y = MINIGOLF_TERRAIN_SIZE

        # 기본 위치 설정 (로봇이 원점에 있으므로 로봇 전방에 배치)
        if ball_pos is None:
            ball_pos = [0.45, -0.15, MINIGOLF_BALL_RADIUS + 0.01]
        if hole_pos is None:
            hole_pos = [0.55, 0.15, 0.0]

        self.ball_start_pos = np.array(ball_pos)
        self.hole_pos = np.array(hole_pos)

        # 1. 지형 생성
        self._create_terrain(size_x, size_y, terrain_seed)

        # 2. 골프공 생성
        self._create_ball(ball_pos)

        # 3. 홀 마커 생성
        self._create_hole_marker(hole_pos)

        # 4. 퍼터 도구 (EE에 부착할 physical body)
        self.tool_id = None

        print(f"[MiniGolf] Environment setup complete")
        print(f"  Ball: {ball_pos}")
        print(f"  Hole: {hole_pos}")

    def _create_terrain(self, size_x, size_y, seed):
        """3D 굴곡 지형 생성"""
        resolution = MINIGOLF_TERRAIN_RESOLUTION
        
        # PyBullet에 메쉬 로드 시 적용할 오프셋 (로봇 전방 배치)
        terrain_offset = [0.5, 0, 0]
        self.terrain_offset = terrain_offset

        # 높이 맵 생성 시 홀 위치와 오프셋을 전달하여 구멍(Hole) 뚫기
        heights = _generate_terrain_heightfield(
            size_x, size_y, resolution, seed,
            hole_pos=self.hole_pos, terrain_offset=terrain_offset
        )

        # OBJ 파일로 저장
        obj_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'project',
            'terrain_mesh.obj'
        )
        obj_path = os.path.abspath(obj_path)
        _create_terrain_mesh_obj(heights, size_x, size_y, obj_path)
        self._temp_files.append(obj_path)
        self.terrain_obj_path = obj_path

        # PyBullet에 메쉬 로드 (오프셋은 위에서 정의됨)

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,  # 오목 메쉬 강제 — convex hull 방지
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[1, 1, 1],
            rgbaColor=COLOR_GOLF_GREEN,
            physicsClientId=self.client
        )

        self.terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=terrain_offset,
            physicsClientId=self.client
        )

        # 마찰 설정 + 접촉 정밀도 향상
        p.changeDynamics(
            self.terrain_id, -1,
            lateralFriction=MINIGOLF_GROUND_FRICTION,
            restitution=MINIGOLF_GROUND_RESTITUTION,
            contactProcessingThreshold=0.001,  # 접촉 감지 정밀도
            physicsClientId=self.client
        )

    def _create_ball(self, position):
        """골프공 생성"""
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=MINIGOLF_BALL_RADIUS,
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=MINIGOLF_BALL_RADIUS,
            rgbaColor=COLOR_WHITE,
            physicsClientId=self.client
        )

        self.ball_id = p.createMultiBody(
            baseMass=MINIGOLF_BALL_MASS,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )

        p.changeDynamics(
            self.ball_id, -1,
            lateralFriction=MINIGOLF_BALL_FRICTION,
            restitution=MINIGOLF_BALL_RESTITUTION,
            rollingFriction=0.005,
            spinningFriction=0.005,
            physicsClientId=self.client
        )

    def _create_hole_marker(self, position):
        """홀 컵 시각적 마커 (원통형)"""
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=MINIGOLF_HOLE_RADIUS,
            length=0.005,
            rgbaColor=COLOR_HOLE_BLACK,
            physicsClientId=self.client
        )
        self.hole_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], position[2] - 0.002],
            physicsClientId=self.client
        )

    def get_ball_position(self):
        """공 현재 위치 반환"""
        pos, _ = p.getBasePositionAndOrientation(
            self.ball_id, physicsClientId=self.client
        )
        return np.array(pos)

    def get_ball_velocity(self):
        """공 현재 속도 반환"""
        vel, _ = p.getBaseVelocity(
            self.ball_id, physicsClientId=self.client
        )
        return np.array(vel)

    def is_ball_stopped(self, threshold=0.005):
        """공이 멈췄는지 확인"""
        vel = self.get_ball_velocity()
        return np.linalg.norm(vel) < threshold

    def is_hole_in(self, threshold=None):
        """홀인원 확인"""
        if threshold is None:
            threshold = MINIGOLF_HOLE_RADIUS
        ball_pos = self.get_ball_position()
        dist = np.linalg.norm(ball_pos[:2] - self.hole_pos[:2])
        return dist < threshold

    def get_distance_to_hole(self):
        """공과 홀 사이의 거리"""
        ball_pos = self.get_ball_position()
        return np.linalg.norm(ball_pos[:2] - self.hole_pos[:2])

    def reset_ball(self, position=None):
        """공 위치 리셋"""
        if position is None:
            position = self.ball_start_pos
        p.resetBasePositionAndOrientation(
            self.ball_id, list(position), [0, 0, 0, 1],
            physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.ball_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client
        )

    def wait_ball_stop(self, timeout=10.0, check_interval=0.1):
        """공이 멈출 때까지 대기"""
        import time
        start = time.time()
        while time.time() - start < timeout:
            # 공이 홀(지하)에 빠지면 즉시 속도를 0으로 트랩(Trap)
            # Z < -0.01: 확실히 지형 아래(구멍 안)로 떨어진 경우만
            pos = self.get_ball_position()
            if pos[2] < -0.01:
                p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
                return True
                
            if self.is_ball_stopped():
                return True
            time.sleep(check_interval)
        return False

    def attach_compact_tool(self, robot_id, ee_link_index,
                            head_length=None, head_radius=None,
                            head_mass=None, head_restitution=None):
        """EE 끝단에 컴팩트 헤드만 직결 부착 (자루 없음)"""
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
        p.changeConstraint(cid, maxForce=TOOL_CONSTRAINT_FORCE,
                           physicsClientId=self.client)

        self.tool_id = head_id
        self._tool_cid = cid
        return head_id

    def disable_robot_env_collision(self, robot_id):
        """로봇 링크와 환경(지형) 간 충돌 비활성화"""
        num_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
        env_bodies = [self.terrain_id]
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
        """도구-지형 충돌 비활성화"""
        if self.tool_id is None:
            return
        env_bodies = [self.terrain_id]
        for env_body in env_bodies:
            if env_body is None:
                continue
            p.setCollisionFilterPair(
                self.tool_id, env_body, -1, -1,
                enableCollision=0,
                physicsClientId=self.client
            )

    def cleanup(self):
        """임시 파일 정리"""
        for f in self._temp_files:
            if os.path.exists(f):
                os.remove(f)

