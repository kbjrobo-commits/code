import numpy as np
import scipy as sp
import pinocchio as pin
import math

import scipy.spatial


def Rot2eul(R, seq='XYZ', degree=False):
    r = sp.spatial.transform.Rotation.from_matrix(R)
    return r.as_euler(seq, degrees=degree)

def Rot2quat(R):
    r = sp.spatial.transform.Rotation.from_matrix(R)
    return r.as_quat()

def quat2Rot(quat):
    r = sp.spatial.transform.Rotation.from_quat(quat)
    return r.as_matrix()

def quat2eul(quat, seq='XYZ', degree=False):
    r = sp.spatial.transform.Rotation.from_quat(quat)
    return r.as_euler(seq, degrees=degree)

def eul2Rot(eul, seq='XYZ', degree=False):
    r = sp.spatial.transform.Rotation.from_euler(seq, eul, degrees=degree)
    return r.as_matrix()

def eul2quat(eul, seq='XYZ', degree=False):
    r = sp.spatial.transform.Rotation.from_euler(seq, eul, degrees=degree)
    return r.as_quat()

def Rot2Vec(R):
    # so3ToVec(MatrixLog3(R))
    return pin.log3(R).reshape([-1, 1])

def Vec2Rot(w):
    # MatrixExp3(VecToso3(w))
    assert type(w) == np.ndarray, f'w={w} should be a numpy array'
    return pin.exp3(w)

def RotX(theta, degree=False):
     return sp.spatial.transform.Rotation.from_euler('XYZ', [theta, 0, 0], degrees=degree).as_matrix()

def RotY(theta, degree=False):
    return sp.spatial.transform.Rotation.from_euler('XYZ', [0, theta, 0], degrees=degree).as_matrix()

def RotZ(theta, degree=False):
    return sp.spatial.transform.Rotation.from_euler('XYZ', [0, 0, theta], degrees=degree).as_matrix()

def deg2radlist(deglist):
    """
    util function to convert degree list to radian list

    :param deglist: list of degrees
    :return: list of radians
    """
    radlist = [deg*math.pi/180 for deg in deglist]
    return radlist

def rad2deglist(radlist):
    """
    util function to convert radian list to degree list

    :param radlist: list of radians
    :return: list of degrees
    """
    deglist = [rad*180/math.pi for rad in radlist]
    return deglist

# SE3

def xyzquat2SE3(xyz, quat):

    T = np.identity(4)
    T[0:3, 0:3] = quat2Rot(quat)
    T[0:3, 3] = xyz

    return T

def xyzeul2SE3(xyz, eul, seq='XYZ', degree=False):

    SE3 = np.identity(4)
    SE3[0:3, 3] = xyz
    SE3[0:3, 0:3] = eul2Rot(eul, seq, degree=degree)

    return SE3

def Vec2SE3(xi):
    """
    util function to convert se3 vector to SE3 ndarray

    :param xi: se3 'vector' in list or ndarray, [w, v]
    :return: SE3 in ndarray
    """
    assert type(xi) == np.ndarray, f'xi={xi} should be a numpy array'
    # xi = np.asarray(xi).reshape(-1)
    return pin.exp6(xi[[3, 4, 5, 0, 1, 2]]).np

def SE32Vec(T):
    return pin.log6(T).np[[3, 4, 5, 0, 1, 2]].reshape(-1, 1)

def PoseVec2SE3(pose):
    """
    util function to convert PoseVec to SE3 ndarray

    :param pose: pose vector in list or ndarray, [xyz, w]
    :return: SE3 in ndarray
    """
    SE3 = np.identity(4)
    pose = np.asarray(pose).reshape(-1).copy()

    SE3[0:3, 3] = pose[0:3]
    SE3[0:3, 0:3] = Vec2Rot(pose[3:6])

    return SE3

def SE32PoseVec(SE3):
    pose = np.zeros([6, 1])
    pose[0:3, 0] = SE3[0:3, 3]
    pose[3:6, :] = Rot2Vec(SE3[0:3, 0:3])

    return pose