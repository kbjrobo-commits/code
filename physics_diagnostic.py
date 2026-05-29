"""
물리 일치 진단 스크립트
========================
실패한 13개 케이스에 대해:
1) GUI에서 측정된 실제 속도/각도를 headless에 넣으면 성공하는가?
2) 물리 불일치가 속도 차이인지, 쿠션 역학 차이인지 판별
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from project.config import *
from project.physics.cushion_planner import CushionShotPlanner
import pybullet as p

# stress_test seed=42 실패 케이스 (20개 중 13개 실패)
FAILURE_CASES = [
    # (test#, cue, tgt1, tgt2, planned_angle_deg, actual_angle_deg, actual_speed, gui_events, gui_cushions)
    (2, [0.587,-0.002,0.071], [0.526,0.028,0.071], [0.490,0.162,0.071],
     17.9, 17.8, 1.859, ['c','c','c','c','c'], 5),
    (3, [0.406,0.403,0.071], [0.554,0.386,0.071], [0.584,0.191,0.071],
     43.1, 42.3, 1.816, ['c','t1','c','c','c'], 4),
    (4, [0.590,-0.100,0.071], [0.409,-0.124,0.071], [0.441,0.072,0.071],
     351.6, 351.3, 1.834, ['c','c','c','t1','c','c','c','c','c'], 8),
    (5, [0.428,0.322,0.071], [0.449,0.010,0.071], [0.496,-0.070,0.071],
     55.4, 55.3, 1.837, ['c','c','t1'], 2),
    (6, [0.561,-0.108,0.071], [0.607,0.290,0.071], [0.410,-0.147,0.071],
     350.0, 349.7, 1.841, ['c','t2','c'], 2),
    (7, [0.389,0.342,0.071], [0.516,0.039,0.071], [0.376,0.027,0.071],
     74.7, 74.7, 1.836, ['c','t1','c','c'], 3),
    (9, [0.467,-0.136,0.071], [0.387,-0.132,0.071], [0.519,0.029,0.071],
     6.9, 6.5, 1.956, ['c','t1','c','c','c','c'], 5),
    (10, [0.487,0.367,0.071], [0.422,0.084,0.071], [0.549,-0.020,0.071],
     79.9, 80.0, 1.837, ['c','t2','c','c','c'], 4),
    (12, [0.578,0.308,0.071], [0.407,0.359,0.071], [0.495,0.310,0.071],
     16.8, 17.5, 1.814, ['c','c','c','t1','c'], 4),
    (13, [0.584,0.031,0.071], [0.388,-0.020,0.071], [0.467,0.316,0.071],
     310.8, 310.3, 1.862, ['c','c','c','t2'], 3),
    (14, [0.575,-0.146,0.071], [0.488,0.088,0.071], [0.416,-0.082,0.071],
     19.5, 19.4, 1.889, ['c','t2','c','c','c','c'], 5),
    (19, [0.550,-0.015,0.071], [0.542,0.060,0.071], [0.518,0.211,0.071],
     27.4, 27.4, 1.836, ['c','t1','t2'], 1),
    (20, [0.494,-0.099,0.071], [0.569,0.033,0.071], [0.407,-0.127,0.071],
     36.3, 36.4, 1.837, ['c','t1','c'], 2),
]

# 성공 케이스 (7개)
SUCCESS_CASES = [
    (1, [0.384,0.240,0.071], [0.470,-0.080,0.071], [0.484,-0.130,0.071],
     61.0, 60.8, 1.811),
    (8, [0.441,0.266,0.071], [0.519,0.356,0.071], [0.478,-0.082,0.071],
     74.4, 74.5, 1.836),
    (11, [0.379,0.015,0.071], [0.400,0.380,0.071], [0.562,0.211,0.071],
     272.0, 271.9, 1.846),
    (15, [0.444,0.387,0.071], [0.441,0.146,0.071], [0.536,0.057,0.071],
     82.4, 82.5, 1.837),
    (16, [0.603,0.399,0.071], [0.423,0.133,0.071], [0.435,0.012,0.071],
     57.4, 57.4, 1.882),
    (17, [0.369,0.197,0.071], [0.486,-0.121,0.071], [0.430,0.368,0.071],
     80.9, 81.0, 1.835),
    (18, [0.420,-0.067,0.071], [0.482,0.412,0.071], [0.421,0.233,0.071],
     263.2, 263.1, 1.858),
]


def run_headless_sim(cue_pos, tgt1_pos, tgt2_pos, angle_deg, speed):
    """headless에서 주어진 속도/각도로 시뮬 실행, 결과 반환"""
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

    # Table
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], physicsClientId=sim)
    table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 basePosition=center, physicsClientId=sim)
    p.changeDynamics(table_id, -1, lateralFriction=MAZE_BALL_FRICTION,
                     restitution=0.5, physicsClientId=sim)

    # Cushions
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
    for pos, half_ext in configs:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=sim)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                basePosition=pos, physicsClientId=sim)
        p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION, physicsClientId=sim)
        cushion_ids.append(cid)

    # Balls
    def make_ball(pos):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, physicsClientId=sim)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=col,
                                basePosition=list(pos), physicsClientId=sim)
        p.changeDynamics(bid, -1,
                         lateralFriction=MAZE_BALL_FRICTION,
                         restitution=MAZE_BALL_RESTITUTION,
                         rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                         spinningFriction=0.02,
                         ccdSweptSphereRadius=MAZE_BALL_RADIUS * 0.5,
                         contactProcessingThreshold=0,
                         physicsClientId=sim)
        return bid

    cue_id = make_ball(cue_pos)
    tgt1_id = make_ball(tgt1_pos)
    tgt2_id = make_ball(tgt2_pos)

    for _ in range(50):
        p.stepSimulation(physicsClientId=sim)

    # Apply velocity
    angle = np.radians(angle_deg)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    p.resetBaseVelocity(cue_id, [vx, vy, 0], [0, 0, 0], physicsClientId=sim)

    # Simulate and track contacts
    hit_t1, hit_t2, cushion_contacts = False, False, 0
    events = []
    prev_cushion = set()

    for step in range(2000):
        p.stepSimulation(physicsClientId=sim)
        contacts = p.getContactPoints(bodyA=cue_id, physicsClientId=sim)
        cur_cushion = set()
        for c in contacts:
            if c[2] == tgt1_id and not hit_t1:
                hit_t1 = True
                events.append('t1')
            elif c[2] == tgt2_id and not hit_t2:
                hit_t2 = True
                events.append('t2')
            elif c[2] in cushion_ids:
                cur_cushion.add(c[2])
        new_cushions = cur_cushion - prev_cushion
        for _ in new_cushions:
            cushion_contacts += 1
            events.append('c')
        prev_cushion = cur_cushion

        if step > 200 and step % 50 == 0:
            speeds = [np.linalg.norm(p.getBaseVelocity(bid, physicsClientId=sim)[0][:2])
                      for bid in [cue_id, tgt1_id, tgt2_id]]
            if all(s < 0.005 for s in speeds):
                break

    p.disconnect(sim)

    # Valid check
    valid = False
    if hit_t1 and hit_t2:
        first_t = None
        cushions_before = 0
        cushions_between = 0
        for e in events:
            if e in ('t1', 't2') and first_t is None:
                first_t = e
            elif e == 'c':
                if first_t is None:
                    cushions_before += 1
                else:
                    cushions_between += 1
        if cushions_before >= 2 or (cushions_before >= 1 and cushions_between >= 1):
            valid = True

    return {
        'hit_t1': hit_t1, 'hit_t2': hit_t2,
        'cushions': cushion_contacts, 'events': events,
        'valid': valid
    }


def main():
    print("=" * 70)
    print("  PHYSICS MATCH DIAGNOSTIC")
    print("  Testing: if headless uses ACTUAL GUI speed/angle, does it succeed?")
    print("=" * 70)

    # Test 1: failures with ACTUAL GUI speed
    print("\n--- FAILURES: headless with ACTUAL GUI speed/angle ---")
    match_count = 0
    for case in FAILURE_CASES:
        test_num, cue, tgt1, tgt2, planned_deg, actual_deg, actual_speed, gui_events, gui_cushions = case
        result = run_headless_sim(cue, tgt1, tgt2, actual_deg, actual_speed)
        match = result['valid']
        if match:
            match_count += 1
        status = "MATCH (would succeed)" if match else "STILL FAIL"
        print(f"  Test {test_num:2d}: {status} | "
              f"headless: t1={result['hit_t1']}, t2={result['hit_t2']}, "
              f"cushions={result['cushions']} | "
              f"GUI: events={gui_events[:5]}...")

    print(f"\n  Fixable by speed match: {match_count}/{len(FAILURE_CASES)}")

    # Test 2: failures with PLANNED speed (1.87) but ACTUAL angle
    print("\n--- FAILURES: headless with PLANNED speed (1.87) + ACTUAL angle ---")
    planned_match = 0
    for case in FAILURE_CASES:
        test_num, cue, tgt1, tgt2, planned_deg, actual_deg, actual_speed, gui_events, gui_cushions = case
        result = run_headless_sim(cue, tgt1, tgt2, actual_deg, 1.87)
        if result['valid']:
            planned_match += 1
        status = "WOULD SUCCEED" if result['valid'] else "STILL FAIL"
        print(f"  Test {test_num:2d}: {status} | "
              f"t1={result['hit_t1']}, t2={result['hit_t2']}, cushions={result['cushions']}")

    print(f"\n  Fixable by planned speed: {planned_match}/{len(FAILURE_CASES)}")

    # Test 3: test sensitivity - for each failure, sweep speed +-10% around actual
    print("\n--- SPEED SENSITIVITY: how many speeds succeed in headless? ---")
    for case in FAILURE_CASES[:5]:  # first 5 for brevity
        test_num, cue, tgt1, tgt2, planned_deg, actual_deg, actual_speed, gui_events, gui_cushions = case
        speeds = np.linspace(actual_speed * 0.95, actual_speed * 1.05, 21)
        successes = []
        for spd in speeds:
            result = run_headless_sim(cue, tgt1, tgt2, actual_deg, spd)
            if result['valid']:
                successes.append(spd)
        pct = len(successes) / len(speeds) * 100
        if successes:
            print(f"  Test {test_num:2d}: {len(successes)}/{len(speeds)} speeds succeed "
                  f"({pct:.0f}%) range=[{min(successes):.3f}, {max(successes):.3f}]")
        else:
            print(f"  Test {test_num:2d}: 0/{len(speeds)} speeds succeed (0%) "
                  f"- physics mismatch is NOT just speed!")

    # Test 4: verify successes still work with actual speed
    print("\n--- SUCCESSES: verification with actual GUI speed ---")
    for case in SUCCESS_CASES:
        test_num, cue, tgt1, tgt2, planned_deg, actual_deg, actual_speed = case
        result = run_headless_sim(cue, tgt1, tgt2, actual_deg, actual_speed)
        status = "CONFIRMED" if result['valid'] else "MISMATCH!"
        print(f"  Test {test_num:2d}: {status} | "
              f"t1={result['hit_t1']}, t2={result['hit_t2']}, cushions={result['cushions']}")

    print(f"\n{'='*70}")
    print(f"  CONCLUSION")
    print(f"  Speed-match fix potential: {match_count}/{len(FAILURE_CASES)} failures fixable")
    print(f"  Expected new rate: {(7+match_count)}/20 = {(7+match_count)/20*100:.0f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
