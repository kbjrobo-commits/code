"""
최적 타격 벡터 계산
=====================
미니골프: 방향/속도 Grid Search + 비용 함수 평가
포켓볼: 기하학적 공-공-포켓 정렬 계산
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project.config import *


def ball_speed_to_ee_speed(v_ball, m_tool=TOOL_HEAD_MASS,
                           m_ball=MINIGOLF_BALL_MASS,
                           e_tool=TOOL_HEAD_RESTITUTION,
                           e_ball=MINIGOLF_BALL_RESTITUTION):
    """운동량 보존 + 반발계수로 공 속도에서 필요한 EE 속도 역산

    1차원 정면충돌 모델:
      v_ball = (1 + e) * m_tool / (m_tool + m_ball) * v_EE

    여기서 e = sqrt(e_tool * e_ball) (합성 반발계수)

    Args:
        v_ball: Grid Search에서 구한 공의 목표 초기 속도 (m/s)
        m_tool: 도구 헤드 질량 (kg)
        m_ball: 공 질량 (kg)
        e_tool: 도구 반발계수
        e_ball: 공 반발계수

    Returns:
        v_ee: 로봇 EE가 달성해야 할 임팩트 속도 (m/s)
    """
    e = np.sqrt(e_tool * e_ball)  # 합성 반발계수
    transfer_ratio = (1 + e) * m_tool / (m_tool + m_ball)
    if transfer_ratio < 1e-6:
        return v_ball
    v_ee = v_ball / transfer_ratio
    return v_ee


class MinigolfShotPlanner:
    """미니골프 타격 계획 — 공→홀 방향 기반"""

    def plan_shot(self, ball_pos, hole_pos):
        """최적 타격 방향 및 속도 계산 (단순 직선 버전)

        Returns:
            strike_dir: 타격 방향 벡터 (정규화됨)
            strike_speed: EE 임팩트 속도 (m/s)
        """
        ball_pos = np.array(ball_pos).flatten()
        hole_pos = np.array(hole_pos).flatten()

        direction = hole_pos[:2] - ball_pos[:2]
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return np.array([1, 0, 0]), 0.3

        strike_dir_2d = direction / distance
        strike_dir = np.array([strike_dir_2d[0], strike_dir_2d[1], 0.0])

        # 공 속도 → EE 속도 역산
        v_ball = np.clip(distance * 2.0, 0.2, MINIGOLF_STRIKE_SPEED)
        v_ee = ball_speed_to_ee_speed(v_ball)

        return strike_dir, v_ee

    def plan_shot_physics_search(self, ball_pos, hole_pos, terrain_obj_path,
                                 terrain_offset=None):
        """PyBullet 물리 시뮬레이션 기반 Grid Search

        별도 headless 시뮬레이션에서 실제 지형 위에 공을 굴려보고
        홀에 가장 가까이 가는 각도/속도 조합을 탐색

        Args:
            ball_pos: 공 위치 [x, y, z]
            hole_pos: 홀 위치 [x, y, z]
            terrain_obj_path: 지형 OBJ 메쉬 파일 경로
            terrain_offset: 지형 오프셋 (기본 [0.5, 0, 0])
        """
        import pybullet as p
        import pybullet_data

        if terrain_offset is None:
            terrain_offset = [0.5, 0, 0]

        ball_pos = np.array(ball_pos).flatten()
        hole_pos = np.array(hole_pos).flatten()

        # === 별도 headless 시뮬레이션 ===
        sim = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=sim)
        p.setTimeStep(1./240, physicsClientId=sim)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=sim)

        # 지형 로드
        terrain_col = p.createCollisionShape(
            p.GEOM_MESH, fileName=terrain_obj_path,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=sim
        )
        terrain_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=terrain_col,
            basePosition=terrain_offset, physicsClientId=sim
        )
        p.changeDynamics(
            terrain_id, -1,
            lateralFriction=MINIGOLF_GROUND_FRICTION,
            restitution=MINIGOLF_GROUND_RESTITUTION,
            contactProcessingThreshold=0.001,
            physicsClientId=sim
        )

        # 공 생성
        ball_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=MINIGOLF_BALL_RADIUS, physicsClientId=sim
        )
        ball_id = p.createMultiBody(
            baseMass=MINIGOLF_BALL_MASS,
            baseCollisionShapeIndex=ball_col,
            basePosition=list(ball_pos), physicsClientId=sim
        )
        p.changeDynamics(
            ball_id, -1,
            lateralFriction=MINIGOLF_BALL_FRICTION,
            restitution=MINIGOLF_BALL_RESTITUTION,
            rollingFriction=0.005, spinningFriction=0.005,
            physicsClientId=sim
        )

        # === Grid Search ===
        base_dir = hole_pos[:2] - ball_pos[:2]
        base_angle = np.arctan2(base_dir[1], base_dir[0])

        best_dist = float('inf')
        best_dir = None
        best_speed = None

        angles = np.arange(
            GRID_ANGLE_RANGE[0],
            GRID_ANGLE_RANGE[1] + GRID_ANGLE_STEP,
            GRID_ANGLE_STEP
        )
        speeds = np.arange(
            GRID_SPEED_RANGE[0],
            GRID_SPEED_RANGE[1] + GRID_SPEED_STEP,
            GRID_SPEED_STEP
        )
        total = len(angles) * len(speeds)

        for angle_offset in angles:
            angle = base_angle + np.radians(angle_offset)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])

            for speed in speeds:
                # Reset ball
                p.resetBasePositionAndOrientation(
                    ball_id, list(ball_pos), [0, 0, 0, 1],
                    physicsClientId=sim
                )
                p.resetBaseVelocity(
                    ball_id,
                    linearVelocity=(direction * speed).tolist(),
                    angularVelocity=[0, 0, 0],
                    physicsClientId=sim
                )

                # Step simulation
                for _ in range(GRID_SIM_STEPS):
                    p.stepSimulation(physicsClientId=sim)
                    # 공이 홀(지하)에 빠지면 즉시 속도를 0으로 트랩(Trap)
                    # Z < -0.01: 확실히 지형 아래(구멍 안)로 떨어진 경우만
                    pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=sim)
                    if pos[2] < -0.01:
                        p.resetBaseVelocity(ball_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=sim)
                        break

                # Check result
                final_pos, _ = p.getBasePositionAndOrientation(
                    ball_id, physicsClientId=sim
                )
                dist_to_hole = np.linalg.norm(
                    np.array(final_pos[:2]) - hole_pos[:2]
                )

                if dist_to_hole < best_dist:
                    best_dist = dist_to_hole
                    best_dir = direction.copy()
                    best_speed = speed

        p.disconnect(sim)

        # v_ball → v_EE 역산 (운동량 보존)
        v_ee = ball_speed_to_ee_speed(best_speed)

        print(f"  Grid Search: {total} evals, best dist={best_dist:.4f}m")
        print(f"    v_ball={best_speed:.3f}m/s -> v_EE={v_ee:.3f}m/s (momentum transfer)")

        return best_dir, v_ee


class BilliardsShotPlanner:
    """포켓볼 타격 계획 — 기하학적 공-공-포켓 정렬"""

    def plan_shot(self, cue_pos, target_pos, pocket_pos):
        """최적 타격 방향 계산

        목표공이 포켓 방향으로 이동하도록 흰 공의 타격 방향 결정

        Args:
            cue_pos: 흰 공 위치
            target_pos: 목표 공 위치
            pocket_pos: 포켓 위치

        Returns:
            strike_dir: 타격 방향 (정규화됨)
            strike_speed: 타격 속도
            contact_point: 흰 공이 맞춰야 할 지점
        """
        cue_pos = np.array(cue_pos).flatten()
        target_pos = np.array(target_pos).flatten()
        pocket_pos = np.array(pocket_pos).flatten()

        # 목표공 → 포켓 방향
        target_to_pocket = pocket_pos[:2] - target_pos[:2]
        dist_tp = np.linalg.norm(target_to_pocket)
        if dist_tp < 1e-6:
            target_to_pocket_dir = np.array([1, 0])
        else:
            target_to_pocket_dir = target_to_pocket / dist_tp

        # 접촉점: 목표공 중심에서 타격 반대 방향으로 2*공반지름
        contact_offset = 2 * BILLIARD_BALL_RADIUS
        contact_point_2d = target_pos[:2] - target_to_pocket_dir * contact_offset
        contact_point = np.array([contact_point_2d[0], contact_point_2d[1], cue_pos[2]])

        # 타격 방향: 흰 공 → 접촉점
        strike_dir_2d = contact_point_2d - cue_pos[:2]
        dist_ct = np.linalg.norm(strike_dir_2d)
        if dist_ct < 1e-6:
            strike_dir_2d = target_to_pocket_dir
        else:
            strike_dir_2d = strike_dir_2d / dist_ct

        strike_dir = np.array([strike_dir_2d[0], strike_dir_2d[1], 0.0])

        # 속도: 목표공~포켓 거리에 비례 (최소 0.5 m/s — 가속 임팩트에 충분한 속도)
        strike_speed = np.clip(dist_tp * 2.5, 0.5, BILLIARD_STRIKE_SPEED)

        return strike_dir, strike_speed, contact_point

    def find_best_pocket_shot(self, cue_pos, target_pos, pocket_positions):
        """가장 유리한 포켓을 선택하고 타격 계획"""
        best_score = float('inf')
        best_result = None

        for pocket_pos in pocket_positions:
            strike_dir, speed, contact = self.plan_shot(cue_pos, target_pos, pocket_pos)

            # 점수: 타격 각도가 작을수록(직선에 가까울수록) 좋음
            cue_to_contact = contact[:2] - np.array(cue_pos[:2])
            contact_to_target = np.array(target_pos[:2]) - contact[:2]

            if np.linalg.norm(cue_to_contact) > 1e-6 and np.linalg.norm(contact_to_target) > 1e-6:
                cos_angle = np.dot(cue_to_contact, contact_to_target) / (
                    np.linalg.norm(cue_to_contact) * np.linalg.norm(contact_to_target)
                )
                angle = np.arccos(np.clip(cos_angle, -1, 1))
            else:
                angle = np.pi

            # 거리 점수
            dist = np.linalg.norm(np.array(target_pos[:2]) - np.array(pocket_pos[:2]))
            score = angle + dist * 0.5  # 각도 + 거리 가중치

            if score < best_score:
                best_score = score
                best_result = {
                    'pocket': pocket_pos,
                    'strike_dir': strike_dir,
                    'strike_speed': speed,
                    'contact_point': contact,
                    'score': score
                }

        return best_result
