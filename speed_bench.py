"""
속도 벤치마크: resetBaseVelocity vs Tool-Push
"""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from project.config import *
import pybullet as p
import pybullet_data


def setup_env(sim):
    p.setGravity(0, 0, -9.81, physicsClientId=sim)
    p.setTimeStep(1./240, physicsClientId=sim)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=sim)

    L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
    TH, H = MAZE_TABLE_HEIGHT, MAZE_TABLE_SURFACE_HEIGHT
    CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y
    center = [CX, CY, H]

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], physicsClientId=sim)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=center, physicsClientId=sim)
    p.changeDynamics(1, -1, lateralFriction=MAZE_BALL_FRICTION, restitution=0.5, physicsClientId=sim)

    CH = MAZE_CUSHION_HEIGHT
    top_z = center[2] + TH/2 + CH/2
    thickness = 0.03
    configs = [
        ([center[0], center[1]+W/2+thickness/2, top_z], [L/2, thickness/2, CH/2]),
        ([center[0], center[1]-W/2-thickness/2, top_z], [L/2, thickness/2, CH/2]),
        ([center[0]-L/2-thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
        ([center[0]+L/2+thickness/2, center[1], top_z], [thickness/2, W/2, CH/2]),
    ]
    cushion_ids = []
    for pos2, half_ext in configs:
        c2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=sim)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c2, basePosition=pos2, physicsClientId=sim)
        p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION, physicsClientId=sim)
        cushion_ids.append(cid)

    return center, cushion_ids


def make_ball(sim, pos):
    c2 = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, physicsClientId=sim)
    bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=c2,
                            basePosition=list(pos), physicsClientId=sim)
    p.changeDynamics(bid, -1, lateralFriction=MAZE_BALL_FRICTION,
                     restitution=MAZE_BALL_RESTITUTION,
                     rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                     spinningFriction=0.02,
                     ccdSweptSphereRadius=MAZE_BALL_RADIUS*0.5,
                     contactProcessingThreshold=0, physicsClientId=sim)
    return bid


def make_striker(sim, pos, orn_quat):
    """도구 팁과 동일한 작은 실린더 striker"""
    tip_r = TOOL_TIP_RADIUS
    tip_l = TOOL_TIP_LENGTH
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=tip_r, height=tip_l, physicsClientId=sim)
    sid = p.createMultiBody(baseMass=TOOL_HEAD_MASS, baseCollisionShapeIndex=col,
                            basePosition=list(pos),
                            baseOrientation=orn_quat,
                            physicsClientId=sim)
    p.changeDynamics(sid, -1, restitution=TOOL_HEAD_RESTITUTION,
                     lateralFriction=0.3, physicsClientId=sim)
    return sid


def bench_reset_velocity(n_tests=2520):
    """기존 방식: resetBaseVelocity"""
    sim = p.connect(p.DIRECT)
    center, cushion_ids = setup_env(sim)

    cue_pos = [0.45, 0.1, 0.071]
    tgt1_pos = [0.5, 0.3, 0.071]
    tgt2_pos = [0.55, 0.0, 0.071]

    cue_id = make_ball(sim, cue_pos)
    tgt1_id = make_ball(sim, tgt1_pos)
    tgt2_id = make_ball(sim, tgt2_pos)

    for _ in range(50):
        p.stepSimulation(physicsClientId=sim)

    angles = np.linspace(0, 2*np.pi, n_tests, endpoint=False)
    speed = 1.87

    t0 = time.perf_counter()
    for i in range(n_tests):
        # Reset
        p.resetBasePositionAndOrientation(cue_id, cue_pos, [0,0,0,1], physicsClientId=sim)
        p.resetBaseVelocity(cue_id, [0,0,0], [0,0,0], physicsClientId=sim)
        p.resetBasePositionAndOrientation(tgt1_id, tgt1_pos, [0,0,0,1], physicsClientId=sim)
        p.resetBaseVelocity(tgt1_id, [0,0,0], [0,0,0], physicsClientId=sim)
        p.resetBasePositionAndOrientation(tgt2_id, tgt2_pos, [0,0,0,1], physicsClientId=sim)
        p.resetBaseVelocity(tgt2_id, [0,0,0], [0,0,0], physicsClientId=sim)

        # Apply velocity
        a = angles[i]
        vx = speed * np.cos(a)
        vy = speed * np.sin(a)
        p.resetBaseVelocity(cue_id, [vx, vy, 0], [0, 0, 0], physicsClientId=sim)

        # Simulate
        for step in range(2000):
            p.stepSimulation(physicsClientId=sim)
            if step > 200 and step % 50 == 0:
                spds = [np.linalg.norm(p.getBaseVelocity(b, physicsClientId=sim)[0][:2])
                        for b in [cue_id, tgt1_id, tgt2_id]]
                if all(s < 0.005 for s in spds):
                    break

    elapsed = time.perf_counter() - t0
    p.disconnect(sim)
    return elapsed


def bench_tool_push(n_tests=2520):
    """새 방식: tool-push (striker로 공을 물리적으로 타격)"""
    sim = p.connect(p.DIRECT)
    center, cushion_ids = setup_env(sim)

    cue_pos = [0.45, 0.1, 0.071]
    tgt1_pos = [0.5, 0.3, 0.071]
    tgt2_pos = [0.55, 0.0, 0.071]

    cue_id = make_ball(sim, cue_pos)
    tgt1_id = make_ball(sim, tgt1_pos)
    tgt2_id = make_ball(sim, tgt2_pos)

    for _ in range(50):
        p.stepSimulation(physicsClientId=sim)

    angles = np.linspace(0, 2*np.pi, n_tests, endpoint=False)
    tool_speed = MAX_TOOL_SPEED  # 1.0 m/s
    approach_dist = 0.03  # striker 시작 위치: 공 뒤 3cm

    # Striker 미리 생성 (매번 생성/삭제보다 리셋이 빠름)
    striker_pos = [0, 0, -1]  # 초기: 테이블 밖
    tip_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    striker_id = make_striker(sim, striker_pos, tip_orn)

    t0 = time.perf_counter()
    for i in range(n_tests):
        a = angles[i]
        dx = np.cos(a)
        dy = np.sin(a)

        # Reset balls
        p.resetBasePositionAndOrientation(cue_id, cue_pos, [0,0,0,1], physicsClientId=sim)
        p.resetBaseVelocity(cue_id, [0,0,0], [0,0,0], physicsClientId=sim)
        p.resetBasePositionAndOrientation(tgt1_id, tgt1_pos, [0,0,0,1], physicsClientId=sim)
        p.resetBaseVelocity(tgt1_id, [0,0,0], [0,0,0], physicsClientId=sim)
        p.resetBasePositionAndOrientation(tgt2_id, tgt2_pos, [0,0,0,1], physicsClientId=sim)
        p.resetBaseVelocity(tgt2_id, [0,0,0], [0,0,0], physicsClientId=sim)

        # Position striker behind ball
        sx = cue_pos[0] - dx * approach_dist
        sy = cue_pos[1] - dy * approach_dist
        sz = cue_pos[2]

        # Striker orientation: cylinder axis perpendicular to strike direction
        strike_yaw = a
        striker_orn = p.getQuaternionFromEuler([0, np.pi/2, strike_yaw])
        p.resetBasePositionAndOrientation(striker_id, [sx, sy, sz], striker_orn,
                                          physicsClientId=sim)
        p.resetBaseVelocity(striker_id, [dx*tool_speed, dy*tool_speed, 0], [0,0,0],
                            physicsClientId=sim)

        # Simulate collision phase (striker hits ball, ~30 steps)
        for _ in range(30):
            p.stepSimulation(physicsClientId=sim)

        # Remove striker from play
        p.resetBasePositionAndOrientation(striker_id, [0, 0, -1], [0,0,0,1],
                                          physicsClientId=sim)
        p.resetBaseVelocity(striker_id, [0,0,0], [0,0,0], physicsClientId=sim)

        # Continue ball simulation
        for step in range(2000):
            p.stepSimulation(physicsClientId=sim)
            if step > 200 and step % 50 == 0:
                spds = [np.linalg.norm(p.getBaseVelocity(b, physicsClientId=sim)[0][:2])
                        for b in [cue_id, tgt1_id, tgt2_id]]
                if all(s < 0.005 for s in spds):
                    break

    elapsed = time.perf_counter() - t0
    p.disconnect(sim)
    return elapsed


if __name__ == '__main__':
    print("=" * 50)
    print("  SPEED BENCHMARK")
    print("=" * 50)

    n = 500  # quick test
    print(f"\n  Testing {n} simulations...")

    t1 = bench_reset_velocity(n)
    print(f"  resetBaseVelocity: {t1:.2f}s ({t1/n*1000:.1f}ms/test)")

    t2 = bench_tool_push(n)
    print(f"  tool-push:         {t2:.2f}s ({t2/n*1000:.1f}ms/test)")

    print(f"\n  Slowdown: {t2/t1:.2f}x")
    print(f"  For 2520 tests: resetVel={t1/n*2520:.1f}s, tool-push={t2/n*2520:.1f}s")
