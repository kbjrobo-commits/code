"""
공유 접촉 물리 모델 (Goal 2)
============================
GUI(maze_env)와 Headless(pocket_planner)가 동일한 dynamics 파라미터를 쓰도록
changeDynamics 적용을 한 곳에서 수행한다. 값은 project.config 의 단일 소스를 참조.
"""
import pybullet as p
from project import config as cfg


def apply_ball_dynamics(body_id, client_id):
    p.changeDynamics(
        body_id, -1,
        lateralFriction=cfg.LATERAL_FRICTION,
        rollingFriction=cfg.ROLLING_FRICTION,
        spinningFriction=cfg.SPINNING_FRICTION,
        restitution=cfg.BALL_RESTITUTION,
        ccdSweptSphereRadius=cfg.MAZE_BALL_RADIUS * 0.5,
        contactProcessingThreshold=0,
        physicsClientId=client_id,
    )


def apply_table_dynamics(body_id, client_id):
    p.changeDynamics(
        body_id, -1,
        lateralFriction=cfg.LATERAL_FRICTION,
        rollingFriction=cfg.ROLLING_FRICTION,
        restitution=cfg.TABLE_RESTITUTION,
        physicsClientId=client_id,
    )


def apply_cushion_dynamics(body_id, client_id):
    p.changeDynamics(
        body_id, -1,
        restitution=cfg.CUSHION_RESTITUTION,
        physicsClientId=client_id,
    )
