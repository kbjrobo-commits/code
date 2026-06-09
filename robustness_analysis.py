"""
강건성 기준 분석: 성공 vs 실패 케이스의 speed sweep 비교
=========================================================
각 테스트 케이스에서 +-5% 속도 범위 21개 중 몇 개가 headless에서 성공하는지
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from project.config import *
import pybullet as p


ALL_CASES = [
    # (test#, success, cue, tgt1, tgt2, actual_angle, actual_speed)
    (1, True, [0.384,0.240,0.071], [0.470,-0.080,0.071], [0.484,-0.130,0.071], 60.8, 1.811),
    (2, False, [0.587,-0.002,0.071], [0.526,0.028,0.071], [0.490,0.162,0.071], 17.8, 1.859),
    (3, False, [0.406,0.403,0.071], [0.554,0.386,0.071], [0.584,0.191,0.071], 42.3, 1.816),
    (4, False, [0.590,-0.100,0.071], [0.409,-0.124,0.071], [0.441,0.072,0.071], 351.3, 1.834),
    (5, False, [0.428,0.322,0.071], [0.449,0.010,0.071], [0.496,-0.070,0.071], 55.3, 1.837),
    (6, False, [0.561,-0.108,0.071], [0.607,0.290,0.071], [0.410,-0.147,0.071], 349.7, 1.841),
    (7, False, [0.389,0.342,0.071], [0.516,0.039,0.071], [0.376,0.027,0.071], 74.7, 1.836),
    (8, True, [0.441,0.266,0.071], [0.519,0.356,0.071], [0.478,-0.082,0.071], 74.5, 1.836),
    (9, False, [0.467,-0.136,0.071], [0.387,-0.132,0.071], [0.519,0.029,0.071], 6.5, 1.956),
    (10, False, [0.487,0.367,0.071], [0.422,0.084,0.071], [0.549,-0.020,0.071], 80.0, 1.837),
    (11, True, [0.379,0.015,0.071], [0.400,0.380,0.071], [0.562,0.211,0.071], 271.9, 1.846),
    (12, False, [0.578,0.308,0.071], [0.407,0.359,0.071], [0.495,0.310,0.071], 17.5, 1.814),
    (13, False, [0.584,0.031,0.071], [0.388,-0.020,0.071], [0.467,0.316,0.071], 310.3, 1.862),
    (14, False, [0.575,-0.146,0.071], [0.488,0.088,0.071], [0.416,-0.082,0.071], 19.4, 1.889),
    (15, True, [0.444,0.387,0.071], [0.441,0.146,0.071], [0.536,0.057,0.071], 82.5, 1.837),
    (16, True, [0.603,0.399,0.071], [0.423,0.133,0.071], [0.435,0.012,0.071], 57.4, 1.882),
    (17, True, [0.369,0.197,0.071], [0.486,-0.121,0.071], [0.430,0.368,0.071], 81.0, 1.835),
    (18, True, [0.420,-0.067,0.071], [0.482,0.412,0.071], [0.421,0.233,0.071], 263.1, 1.858),
    (19, False, [0.550,-0.015,0.071], [0.542,0.060,0.071], [0.518,0.211,0.071], 27.4, 1.836),
    (20, False, [0.494,-0.099,0.071], [0.569,0.033,0.071], [0.407,-0.127,0.071], 36.4, 1.837),
]


def run_sim(cue, tgt1, tgt2, angle_deg, speed):
    sim = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=sim)
    p.setTimeStep(1./240, physicsClientId=sim)

    L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
    TH, H = MAZE_TABLE_HEIGHT, MAZE_TABLE_SURFACE_HEIGHT
    CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y
    center = [CX, CY, H]

    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=sim)

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], physicsClientId=sim)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=center, physicsClientId=sim)
    p.changeDynamics(0, -1, lateralFriction=MAZE_BALL_FRICTION, restitution=0.5, physicsClientId=sim)

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

    def ball(pos2):
        c2 = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, physicsClientId=sim)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=c2,
                                basePosition=list(pos2), physicsClientId=sim)
        p.changeDynamics(bid, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=MAZE_BALL_RESTITUTION,
                         rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                         spinningFriction=0.02,
                         ccdSweptSphereRadius=MAZE_BALL_RADIUS*0.5,
                         contactProcessingThreshold=0, physicsClientId=sim)
        return bid

    cue_id = ball(cue)
    tgt1_id = ball(tgt1)
    tgt2_id = ball(tgt2)
    for _ in range(50): p.stepSimulation(physicsClientId=sim)

    a = np.radians(angle_deg)
    p.resetBaseVelocity(cue_id, [speed*np.cos(a), speed*np.sin(a), 0], [0,0,0], physicsClientId=sim)

    ht1, ht2, cc = False, False, 0
    evts = []
    prev = set()
    for step in range(2000):
        p.stepSimulation(physicsClientId=sim)
        contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim)
        cur = set()
        for c in contacts:
            if c[2] == tgt1_id and not ht1: ht1 = True; evts.append('t1')
            elif c[2] == tgt2_id and not ht2: ht2 = True; evts.append('t2')
            elif c[2] in cushion_ids: cur.add(c[2])
        for _ in cur - prev: cc += 1; evts.append('c')
        prev = cur
        if step > 200 and step % 50 == 0:
            spds = [np.linalg.norm(p.getBaseVelocity(b, physicsClientId=sim)[0][:2])
                    for b in [cue_id, tgt1_id, tgt2_id]]
            if all(s < 0.005 for s in spds): break

    p.disconnect(sim)

    valid = False
    if ht1 and ht2:
        ft = None; cb = 0; cbt = 0
        for e in evts:
            if e in ('t1','t2') and ft is None: ft = e
            elif e == 'c':
                if ft is None: cb += 1
                else: cbt += 1
        if cb >= 2 or (cb >= 1 and cbt >= 1): valid = True
    return valid


def main():
    print("=" * 70)
    print("  ROBUSTNESS ANALYSIS")
    print("  For each case: sweep angle +-1 deg (0.1 step) x 5 speeds")
    print("=" * 70)

    speeds = [1.777, 1.81, 1.837, 1.87, 1.96]

    for case in ALL_CASES:
        test_num, gui_success, cue, tgt1, tgt2, actual_angle, actual_speed = case
        # Sweep angles and speeds
        angle_offsets = np.arange(-1.0, 1.05, 0.1)
        total = 0
        success_count = 0
        for da in angle_offsets:
            for spd in speeds:
                total += 1
                if run_sim(cue, tgt1, tgt2, actual_angle + da, spd):
                    success_count += 1

        robustness = success_count / total * 100
        marker = "OK " if gui_success else "MISS"
        bar = "#" * int(robustness / 5) + "." * (20 - int(robustness / 5))
        print(f"  Test {test_num:2d} [{marker}]: {robustness:5.1f}% [{bar}] "
              f"({success_count}/{total})")

    print()


if __name__ == '__main__':
    main()
