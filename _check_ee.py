import numpy as np
import sys, os
sys.path.append('.')
import pybullet as p

# headless direct connect
cid = p.connect(p.DIRECT)

from src.core.pybullet_core import PybulletCore
from project.config import *

pb = PybulletCore()
pb.connect(robot_name='indy7_v2', joint_limit=True, constraint_visualization=False)
q_home = np.array(HOME_Q_DEG) * np.pi / 180
T = pb.my_robot.pinModel.FK(q_home)

print("=== Pinocchio EE Frame at HOME ===")
print(f"Position: [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]")
print(f"z-axis:   [{T[0,2]:.4f}, {T[1,2]:.4f}, {T[2,2]:.4f}]")
print()
if T[2,2] < -0.5:
    print("z-axis points DOWN -> tool is on +z side (standard)")
elif T[2,2] > 0.5:
    print("z-axis points UP -> tool is on -z side (inverted!)")
else:
    print(f"z-axis is diagonal, z[2]={T[2,2]:.4f}")

# Also check what the EE link frame looks like in PyBullet
robot_id = pb.my_robot.robotId
ee_idx = pb.my_robot.RobotEEJointIdx[-1]
state = p.getLinkState(robot_id, ee_idx, physicsClientId=pb.ClientId)
pos = state[4]
orn = state[5]
mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
print(f"\n=== PyBullet EE Link Frame ===")
print(f"Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
print(f"z-axis:   [{mat[0,2]:.4f}, {mat[1,2]:.4f}, {mat[2,2]:.4f}]")

pb.disconnect()
