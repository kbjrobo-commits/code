
import os
import sys
import time
import json
import yaml
from math import *

import numpy as np
import pinocchio as pin

from .robotics_utils import *
from .rotation_utils import *

class PinocchioModel:
    def __init__(self, urdf_dir, T_W0=None):

        # Import robot
        if T_W0 is None:
            T_W0 = np.identity(4)

        self._robot_name = os.path.basename(urdf_dir)
        self._robot_type = os.path.basename(os.path.dirname(urdf_dir))

        # Open YAML file
        with open(urdf_dir + "/../robot_configs.yaml".format(self._robot_type)) as yaml_file:
            self._robot_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

        xyz = self._robot_configs[self._robot_name]["EndEffector"]["position"]
        rpy = self._robot_configs[self._robot_name]["EndEffector"]["orientation"]
        self._T_CoME = xyzeul2SE3(xyz, rpy, seq='XYZ', degree=True)

        self.RobotBaseJointIdx = self._robot_configs[self._robot_name]["JointInfo"]["RobotBaseJoint"]
        self.RobotMovableJointIdx = self._robot_configs[self._robot_name]["JointInfo"]["RobotMovableJoint"]
        self.RobotEEJointIdx = self._robot_configs[self._robot_name]["JointInfo"]["RobotEEJoint"]

        if len(self.RobotBaseJointIdx) == 0:
            self.RobotBaseJointIdx = [-1]
        if len(self.RobotEEJointIdx) == 0:
            self.RobotEEJointIdx = [self.RobotMovableJointIdx[-1]]

        # Robot's base coordinate in world coordinate
        self._T_W0 = T_W0
        self._Ad_W0 = Adjoint(self._T_W0)

        # Load pinocchio model
        self.pinModel = pin.buildModelFromUrdf(urdf_dir + "/../{}/model.urdf".format(self._robot_type))
        self.pinData = self.pinModel.createData()
        self.numJoints = self.pinModel.nq

        if self.numJoints != len(self.RobotMovableJointIdx):
            raise Exception("Wrong number of movable joints!")

        pin.forwardKinematics(self.pinModel, self.pinData, np.zeros([self.numJoints, 1]))
        pin.updateFramePlacements(self.pinModel, self.pinData)


    def FK(self, q):
        pin.forwardKinematics(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))
        pin.updateFramePlacements(self.pinModel, self.pinData)
        return self._T_W0 @ self.pinData.oMi[self.numJoints].np @ self._T_CoME
        # 2 [ground link + joint] + 2*(TCP index) [robot link + joint]
        # return self._T_W0 @ self.pinData.oMf[2+2*(self.RobotEEJointIdx[0]+1)].np @ self._T_CoME


    def Js(self, q):
        pin.forwardKinematics(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))
        J = pin.computeJointJacobians(self.pinModel, self.pinData)
        return self._Ad_W0 @ J[[3, 4, 5, 0, 1, 2], :] # [Jv; Jw] to [Jw; Jv]

    def Jb(self, q):
        return Adjoint(TransInv(self.FK(q))) @ self.Js(q)

    def Minv(self, q):
        return pin.computeMinverse(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))

    def M(self, q):
        return np.linalg.inv(self.Minv(q))

    def C(self, q, qdot):
        return pin.computeCoriolisMatrix(self.pinModel, self.pinData,
                                         np.asarray(q).reshape([-1, 1]), np.asarray(qdot).reshape([-1, 1]))

    def g(self, q):
        return pin.computeGeneralizedGravity(self.pinModel, self.pinData,
                                             np.asarray(q).reshape([-1, 1])).reshape([-1, 1])