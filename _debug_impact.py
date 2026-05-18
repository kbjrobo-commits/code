"""
디버그: headless 도구 타격 후 큐볼 속도 vs GUI 로봇 타격 후 큐볼 속도 비교
"""
import numpy as np
import pybullet as p
import sys, os
sys.path.append('.')
from project.config import *

# =============================================
# 1) Headless 도구 타격 (플래너 방식)
# =============================================
sim = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81, physicsClientId=sim)
p.setTimeStep(1./240, physicsClientId=sim)

L, W = MAZE_TABLE_LENGTH, MAZE_TABLE_WIDTH
TH = MAZE_TABLE_HEIGHT
H = MAZE_TABLE_SURFACE_HEIGHT
CY = MAZE_TABLE_CENTER_Y
center = [0.5, CY, H]

# 테이블
col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, TH/2], physicsClientId=sim)
table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                              basePosition=center, physicsClientId=sim)
p.changeDynamics(table_id, -1, lateralFriction=MAZE_BALL_FRICTION,
                 restitution=0.5, physicsClientId=sim)

# 큐볼
ball_z = center[2] + TH/2 + MAZE_BALL_RADIUS
cue_pos = [0.5, 0.25, ball_z]
col_b = p.createCollisionShape(p.GEOM_SPHERE, radius=MAZE_BALL_RADIUS, physicsClientId=sim)
cue_id = p.createMultiBody(baseMass=MAZE_BALL_MASS, baseCollisionShapeIndex=col_b,
                            basePosition=cue_pos, physicsClientId=sim)
p.changeDynamics(cue_id, -1, lateralFriction=MAZE_BALL_FRICTION,
                 restitution=MAZE_BALL_RESTITUTION,
                 rollingFriction=MAZE_BALL_ROLLING_FRICTION,
                 spinningFriction=0.02,
                 ccdSweptSphereRadius=MAZE_BALL_RADIUS*0.5,
                 contactProcessingThreshold=0, physicsClientId=sim)

# 도구 (50kg)
EFFECTIVE_MASS = 50.0
tool_col = p.createCollisionShape(p.GEOM_SPHERE, radius=TOOL_HEAD_RADIUS, physicsClientId=sim)
tool_id = p.createMultiBody(baseMass=EFFECTIVE_MASS, baseCollisionShapeIndex=tool_col,
                             basePosition=[0, 0, -1], physicsClientId=sim)
p.changeDynamics(tool_id, -1, restitution=TOOL_HEAD_RESTITUTION,
                 lateralFriction=0.3, physicsClientId=sim)
# 도구-테이블 충돌 비활성
p.setCollisionFilterPair(tool_id, table_id, -1, -1, 0, physicsClientId=sim)

# 타격: 각도 90° (위쪽, +y), ee_speed = 0.5 m/s
ee_speed = 0.5
angle = np.radians(90)
angle_rad = np.radians(MAZE_STRIKE_ANGLE_DEG)
dx = np.cos(angle) * np.cos(angle_rad)
dy = np.sin(angle) * np.cos(angle_rad)
dz = -np.sin(angle_rad)
strike_dir = np.array([dx, dy, dz])
strike_dir /= np.linalg.norm(strike_dir)

gap = MAZE_BALL_RADIUS + TOOL_HEAD_RADIUS + 0.005
tool_pos = np.array(cue_pos) - strike_dir * gap
p.resetBasePositionAndOrientation(tool_id, list(tool_pos), [0,0,0,1], physicsClientId=sim)
tool_vel = strike_dir * ee_speed
p.resetBaseVelocity(tool_id, list(tool_vel), [0,0,0], physicsClientId=sim)

# 시뮬 30스텝
for i in range(30):
    p.stepSimulation(physicsClientId=sim)

v_cue_headless, _ = p.getBaseVelocity(cue_id, physicsClientId=sim)
speed_headless = np.linalg.norm(v_cue_headless[:2])
pos_cue, _ = p.getBasePositionAndOrientation(cue_id, physicsClientId=sim)

print(f"=== Headless 도구 타격 (유효질량={EFFECTIVE_MASS}kg) ===")
print(f"  EE speed: {ee_speed} m/s")
print(f"  타격 후 큐볼 수평속도: {speed_headless:.4f} m/s")
print(f"  큐볼 속도벡터: [{v_cue_headless[0]:.4f}, {v_cue_headless[1]:.4f}, {v_cue_headless[2]:.4f}]")
print(f"  큐볼 위치: [{pos_cue[0]:.4f}, {pos_cue[1]:.4f}, {pos_cue[2]:.4f}]")
print(f"  이론값(무한질량): v_ball = (1+e)*v_tool*cos(15°) = {(1+TOOL_HEAD_RESTITUTION)*ee_speed*np.cos(angle_rad):.4f}")

p.disconnect(sim)

# =============================================
# 2) GUI 로봇 타격 결과 예상
# =============================================
print(f"\n=== 이론적 비교 ===")
print(f"  도구 반발계수: {TOOL_HEAD_RESTITUTION}")
print(f"  공 반발계수: {MAZE_BALL_RESTITUTION}")
print(f"  유효 반발계수 e = sqrt({TOOL_HEAD_RESTITUTION}*{MAZE_BALL_RESTITUTION}) = {np.sqrt(TOOL_HEAD_RESTITUTION*MAZE_BALL_RESTITUTION):.4f}")
e_eff = np.sqrt(TOOL_HEAD_RESTITUTION * MAZE_BALL_RESTITUTION)
m_tool_real = TOOL_HEAD_MASS
m_ball = MAZE_BALL_MASS
# 유한 질량 (0.5kg 도구)
ratio_finite = (1+e_eff) * m_tool_real / (m_tool_real + m_ball)
v_ball_finite = ratio_finite * ee_speed * np.cos(angle_rad)
# 무한 질량 (PD 로봇)
v_ball_infinite = (1+e_eff) * ee_speed * np.cos(angle_rad)
print(f"  0.5kg 도구: v_ball = {v_ball_finite:.4f} m/s")
print(f"  무한질량 도구: v_ball = {v_ball_infinite:.4f} m/s")
print(f"  비율(무한/유한): {v_ball_infinite/v_ball_finite:.2f}x")
