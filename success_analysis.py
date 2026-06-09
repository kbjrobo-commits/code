"""
성공 vs 실패 패턴 분석 (headless only, 빠른 실행)
==============================================
1) 성공 케이스의 공통점은?
2) GUI 검증 시간 측정
3) saveState/restoreState 기반 GUI-내 공 시뮬 속도 측정
"""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from project.config import *
import pybullet as p
import pybullet_data

# Original stress test (seed=42) with robustness bonus: 7/20
# 실제 GUI 결과 데이터
CASES = [
    # (test, OK?, cue, tgt1, tgt2, planned_angle, actual_angle, actual_speed, gui_cushions, gui_events)
    (1, True, [0.384,0.240], [0.470,-0.080], [0.484,-0.130], 61.0, 60.8, 1.811, 4, ['c','c','t1','t2','c','c']),
    (2, False, [0.587,-0.002], [0.526,0.028], [0.490,0.162], 17.9, 17.8, 1.859, 5, ['c','c','c','c','c']),
    (3, False, [0.406,0.403], [0.554,0.386], [0.584,0.191], 43.1, 42.3, 1.816, 4, ['c','t1','c','c','c']),
    (4, False, [0.590,-0.100], [0.409,-0.124], [0.441,0.072], 351.6, 351.3, 1.834, 8, ['c','c','c','t1','c','c','c','c','c']),
    (5, False, [0.428,0.322], [0.449,0.010], [0.496,-0.070], 55.4, 55.3, 1.837, 2, ['c','c','t1']),
    (6, False, [0.561,-0.108], [0.607,0.290], [0.410,-0.147], 350.0, 349.7, 1.841, 2, ['c','t2','c']),
    (7, False, [0.389,0.342], [0.516,0.039], [0.376,0.027], 74.7, 74.7, 1.836, 3, ['c','t1','c','c']),
    (8, True, [0.441,0.266], [0.519,0.356], [0.478,-0.082], 74.4, 74.5, 1.836, 2, ['c','t1','c','t2']),
    (9, False, [0.467,-0.136], [0.387,-0.132], [0.519,0.029], 6.9, 6.5, 1.956, 5, ['c','t1','c','c','c','c']),
    (10, False, [0.487,0.367], [0.422,0.084], [0.549,-0.020], 79.9, 80.0, 1.837, 4, ['c','t2','c','c','c']),
    (11, True, [0.379,0.015], [0.400,0.380], [0.562,0.211], 272.0, 271.9, 1.846, 3, ['c','t1','c','c','t2']),
    (12, False, [0.578,0.308], [0.407,0.359], [0.495,0.310], 16.8, 17.5, 1.814, 4, ['c','c','c','t1','c']),
    (13, False, [0.584,0.031], [0.388,-0.020], [0.467,0.316], 310.8, 310.3, 1.862, 3, ['c','c','c','t2']),
    (14, False, [0.575,-0.146], [0.488,0.088], [0.416,-0.082], 19.5, 19.4, 1.889, 5, ['c','t2','c','c','c','c']),
    (15, True, [0.444,0.387], [0.441,0.146], [0.536,0.057], 82.4, 82.5, 1.837, 6, ['c','c','t1','c','c','c','c','t2']),
    (16, True, [0.603,0.399], [0.423,0.133], [0.435,0.012], 57.4, 57.4, 1.882, 9, ['c','c','c','c','c','c','t1','t2','c','c','c']),
    (17, True, [0.369,0.197], [0.486,-0.121], [0.430,0.368], 80.9, 81.0, 1.835, 3, ['c','t2','c','t1','c']),
    (18, True, [0.420,-0.067], [0.482,0.412], [0.421,0.233], 263.2, 263.1, 1.858, 4, ['c','t2','c','c','t1','c']),
    (19, False, [0.550,-0.015], [0.542,0.060], [0.518,0.211], 27.4, 27.4, 1.836, 1, ['c','t1','t2']),
    (20, False, [0.494,-0.099], [0.569,0.033], [0.407,-0.127], 36.3, 36.4, 1.837, 2, ['c','t1','c']),
]


def analyze_geometry():
    """성공/실패 케이스의 기하학적 특성 분석"""
    print("=" * 70)
    print("  SUCCESS vs FAILURE PATTERN ANALYSIS")
    print("=" * 70)

    for label, group in [("SUCCESSES", [c for c in CASES if c[1]]),
                         ("FAILURES", [c for c in CASES if not c[1]])]:
        print(f"\n  --- {label} ({len(group)} cases) ---")
        angles = []
        speeds = []
        cushions = []
        cue_tgt1_dists = []
        cue_tgt2_dists = []
        tgt1_tgt2_dists = []
        angle_diffs = []
        # 공 방향 vs 타겟 방향 관계
        strike_to_tgt1_angles = []
        
        for c in group:
            test, ok, cue, tgt1, tgt2, planned, actual, speed, cush, events = c
            angles.append(planned)
            speeds.append(speed)
            cushions.append(cush)
            angle_diffs.append(abs(planned - actual))
            
            cue, tgt1, tgt2 = np.array(cue), np.array(tgt1), np.array(tgt2)
            cue_tgt1_dists.append(np.linalg.norm(cue - tgt1))
            cue_tgt2_dists.append(np.linalg.norm(cue - tgt2))
            tgt1_tgt2_dists.append(np.linalg.norm(tgt1 - tgt2))
            
            # 타격 방향과 tgt1 방향 사이의 각도
            strike_dir = np.array([np.cos(np.radians(planned)), np.sin(np.radians(planned))])
            to_tgt1 = (tgt1 - cue)
            to_tgt1_norm = to_tgt1 / (np.linalg.norm(to_tgt1) + 1e-8)
            dot = np.clip(np.dot(strike_dir, to_tgt1_norm), -1, 1)
            strike_to_tgt1_angles.append(np.degrees(np.arccos(dot)))
            
            # hit 순서 (GUI)
            first_hit = None
            for e in events:
                if e in ('t1', 't2'):
                    first_hit = e
                    break
            cushions_before_first = 0
            for e in events:
                if e == 'c':
                    cushions_before_first += 1
                elif e in ('t1', 't2'):
                    break
            
            print(f"    Test {test:2d}: angle={planned:6.1f}, speed={speed:.3f}, "
                  f"d(cue-t1)={cue_tgt1_dists[-1]:.3f}, d(cue-t2)={cue_tgt2_dists[-1]:.3f}, "
                  f"d(t1-t2)={tgt1_tgt2_dists[-1]:.3f}, "
                  f"strike-to-t1={strike_to_tgt1_angles[-1]:.0f}deg, "
                  f"cushBefore1st={cushions_before_first}, "
                  f"firstHit={first_hit}")
        
        print(f"\n    Averages:")
        print(f"      Angle range: {min(angles):.0f} - {max(angles):.0f}")
        print(f"      Speed: {np.mean(speeds):.3f} +/- {np.std(speeds):.3f}")
        print(f"      GUI cushions: {np.mean(cushions):.1f} +/- {np.std(cushions):.1f}")
        print(f"      d(cue-tgt1): {np.mean(cue_tgt1_dists):.3f} +/- {np.std(cue_tgt1_dists):.3f}")
        print(f"      d(cue-tgt2): {np.mean(cue_tgt2_dists):.3f} +/- {np.std(cue_tgt2_dists):.3f}")
        print(f"      d(tgt1-tgt2): {np.mean(tgt1_tgt2_dists):.3f} +/- {np.std(tgt1_tgt2_dists):.3f}")
        print(f"      Strike-to-tgt1 angle: {np.mean(strike_to_tgt1_angles):.0f} +/- {np.std(strike_to_tgt1_angles):.0f}")
        print(f"      Angle diff (planned vs actual): {np.mean(angle_diffs):.2f}")


def measure_gui_verification_speed():
    """GUI saveState/restoreState 기반 검증 속도 측정"""
    print(f"\n{'='*70}")
    print(f"  GUI VERIFICATION SPEED BENCHMARK")
    print(f"{'='*70}")
    
    # GUI 모드로 PyBullet 시작
    gui_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=gui_id)
    p.setTimeStep(1./240, physicsClientId=gui_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=gui_id)
    
    # 렌더링 끄기 (속도 측정)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=gui_id)
    
    L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
    TH, H = MAZE_TABLE_HEIGHT, MAZE_TABLE_SURFACE_HEIGHT
    CX, CY = MAZE_TABLE_CENTER_X, MAZE_TABLE_CENTER_Y
    center = [CX, CY, H]
    
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], physicsClientId=gui_id)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=center, physicsClientId=gui_id)
    p.changeDynamics(1, -1, lateralFriction=MAZE_BALL_FRICTION, restitution=0.5, physicsClientId=gui_id)
    
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
        c2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=gui_id)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c2, basePosition=pos2, physicsClientId=gui_id)
        p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION, physicsClientId=gui_id)
        cushion_ids.append(cid)
    
    def ball(pos2):
        c2 = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, physicsClientId=gui_id)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=c2,
                                basePosition=list(pos2) + [0.071], physicsClientId=gui_id)
        p.changeDynamics(bid, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=MAZE_BALL_RESTITUTION,
                         rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                         spinningFriction=0.02,
                         ccdSweptSphereRadius=MAZE_BALL_RADIUS*0.5,
                         contactProcessingThreshold=0, physicsClientId=gui_id)
        return bid
    
    cue_id = ball([0.45, 0.1])
    tgt1_id = ball([0.5, 0.3])
    tgt2_id = ball([0.55, 0.0])
    
    for _ in range(50): p.stepSimulation(physicsClientId=gui_id)
    
    # 속도 측정: saveState → apply vel → simulate → restore
    n_verify = 20  # 후보 수
    
    t0 = time.perf_counter()
    for i in range(n_verify):
        state_id = p.saveState(physicsClientId=gui_id)
        
        a = np.random.uniform(0, 2*np.pi)
        spd = 1.87
        p.resetBaseVelocity(cue_id, [spd*np.cos(a), spd*np.sin(a), 0], [0,0,0],
                            physicsClientId=gui_id)
        
        for step in range(2000):
            p.stepSimulation(physicsClientId=gui_id)
            if step > 200 and step % 50 == 0:
                speeds = [np.linalg.norm(p.getBaseVelocity(b, physicsClientId=gui_id)[0][:2])
                          for b in [cue_id, tgt1_id, tgt2_id]]
                if all(s < 0.005 for s in speeds):
                    break
        
        p.restoreState(stateId=state_id, physicsClientId=gui_id)
        p.removeState(state_id, physicsClientId=gui_id)
    
    elapsed = time.perf_counter() - t0
    print(f"\n  GUI saveState/restoreState: {n_verify} verifications in {elapsed:.2f}s")
    print(f"  Per candidate: {elapsed/n_verify*1000:.0f}ms")
    print(f"  For top 10 candidates: {elapsed/n_verify*10:.2f}s")
    
    # 비교: DIRECT 모드
    p.disconnect(gui_id)
    
    direct_id = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=direct_id)
    p.setTimeStep(1./240, physicsClientId=direct_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=direct_id)
    
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], physicsClientId=direct_id)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=center, physicsClientId=direct_id)
    
    for pos2, half_ext in configs:
        c2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext, physicsClientId=direct_id)
        cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c2, basePosition=pos2, physicsClientId=direct_id)
        p.changeDynamics(cid, -1, restitution=MAZE_CUSHION_RESTITUTION, physicsClientId=direct_id)
    
    def ball_d(pos2):
        c2 = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, physicsClientId=direct_id)
        bid = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=c2,
                                basePosition=list(pos2) + [0.071], physicsClientId=direct_id)
        p.changeDynamics(bid, -1, lateralFriction=MAZE_BALL_FRICTION,
                         restitution=MAZE_BALL_RESTITUTION,
                         rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                         spinningFriction=0.02,
                         ccdSweptSphereRadius=MAZE_BALL_RADIUS*0.5,
                         contactProcessingThreshold=0, physicsClientId=direct_id)
        return bid
    
    cue_d = ball_d([0.45, 0.1])
    tgt1_d = ball_d([0.5, 0.3])
    tgt2_d = ball_d([0.55, 0.0])
    
    for _ in range(50): p.stepSimulation(physicsClientId=direct_id)
    
    t0 = time.perf_counter()
    for i in range(n_verify):
        state_id = p.saveState(physicsClientId=direct_id)
        a = np.random.uniform(0, 2*np.pi)
        p.resetBaseVelocity(cue_d, [1.87*np.cos(a), 1.87*np.sin(a), 0], [0,0,0],
                            physicsClientId=direct_id)
        for step in range(2000):
            p.stepSimulation(physicsClientId=direct_id)
            if step > 200 and step % 50 == 0:
                speeds = [np.linalg.norm(p.getBaseVelocity(b, physicsClientId=direct_id)[0][:2])
                          for b in [cue_d, tgt1_d, tgt2_d]]
                if all(s < 0.005 for s in speeds): break
        p.restoreState(stateId=state_id, physicsClientId=direct_id)
        p.removeState(state_id, physicsClientId=direct_id)
    elapsed2 = time.perf_counter() - t0
    print(f"\n  DIRECT saveState/restoreState: {n_verify} verifications in {elapsed2:.2f}s")
    print(f"  Per candidate: {elapsed2/n_verify*1000:.0f}ms")
    
    p.disconnect(direct_id)


if __name__ == '__main__':
    analyze_geometry()
    measure_gui_verification_speed()
