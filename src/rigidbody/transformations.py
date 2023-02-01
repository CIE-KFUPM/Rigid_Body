import numpy as np
from functools import partial


def rotation_matrix_x(theta: float) -> np.ndarray:
    """
    Rotation about x axis by angle theta.
    Parameters
    ----------
    theta: float

    Returns
    -------
    Rotation matrix: np.ndarray
    """
    R = np.array([
        [1., 0., 0.],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]])
    return R


def rotation_matrix_y(theta: float) -> np.ndarray:
    """
    Rotation about y axis by angle theta.
    Parameters
    ----------
    theta: float

    Returns
    -------
    Rotation matrix: np.ndarray
    """
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1., 0.],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return R


def rotation_matrix_z(theta: float) -> np.array:
    """
    Rotation about z axis by angle theta.
    Parameters
    ----------
    theta: float

    Returns
    -------
    Rotation matrix: np.ndarray
    """
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1.]])
    return R


def euler_angles_from_rot_matrix(r: np.ndarray, sign_of_sin_theta: float = 1) -> np.ndarray:
    """
    Compute euler angles [phi,theta, psi], from a rotation matrix r.
    Parameters
    ----------
    r   : np.array (3 , 3)
    sign_of_sin_theta : float

    Returns
    -------
    angles: np.array  [phi, theta, psi]
    """

    if sign_of_sin_theta == 1:
        theta = np.arctan2(np.sqrt(1 - r[2, 2] ** 2), r[2, 2])
        phi = np.arctan2(r[1, 2], r[0, 2])
        psi = np.arctan2(r[2, 1], -r[2, 0])
    elif sign_of_sin_theta == -1:
        theta = np.arctan2(-np.sqrt(1 - r[2, 2] ** 2), r[2, 2])
        phi = np.arctan2(-r[1, 2], -r[0, 2])
        psi = np.arctan2(-r[2, 1], r[2, 0])
    else:
        raise ValueError("sign_of_sin_theta needs to be either 1 or -1.")

    return np.array([phi, theta, psi])


def roll_pitch_yaw_angles_from_rot_matrix(r: np.ndarray, sign_of_cos_theta: float = 1) -> np.ndarray:
    """
    Compute roll-pitch-yaw angles [phi,theta, psi], from a rotation matrix r.
    Parameters
    ----------
    r   : np.array (3 , 3)
    sign_of_cos_theta : float

    Returns
    -------
    Angles: np.array [phi, theta, psi]
    """
    if sign_of_cos_theta == 1:
        theta = np.arctan2(-r[2, 0], np.sqrt(1 - r[2, 0] ** 2))
        phi = np.arctan2(r[1, 0], r[0, 0])
        psi = np.arctan2(r[2, 1], r[2, 2])
    elif sign_of_cos_theta == -1:
        theta = np.arctan2(-r[2, 0], -np.sqrt(1 - r[2, 0] ** 2))
        phi = np.arctan2(-r[1, 0], -r[0, 0])
        psi = np.arctan2(-r[2, 1], r[2, 2])
    else:
        raise ValueError("sign_of_cos_theta needs to be either 1 or -1.")

    return np.array([phi, theta, psi])


def rotation_matrix_from_roll_pitch_yaw(angles: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix for given roll-pitch-yaw angles

    Parameters
    ----------
    angles: np.ndarray (3,)

    Returns
    -------
    Rotation matrix: np.ndarray (3,3)
    """
    return rotation_matrix_z(angles[0]) @ rotation_matrix_y(angles[1]) @ rotation_matrix_x(angles[2])


def rotation_matrix_from_euler_angles(angles: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix for given Euler angles.

    Parameters
    ----------
    angles: np.ndarray (3,)

    Returns
    -------
    Rotation matrix: np.ndarray (3,3)
    """
    return rotation_matrix_z(angles[0]) @ rotation_matrix_y(angles[1]) @ rotation_matrix_z(angles[2])


def axis_angle_from_rot_matrix(r: np.ndarray) -> tuple:
    """
    Compute axis/angle representation from a rotation matrix r.

    Parameters
    ----------
    r : np.ndarray (3 , 3)

    Returns
    -------
    tuple: k (np.array 3x1) and theta (float)
    """

    theta = np.arccos((np.trace(r) - 1) / 2)
    k = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(theta))
    return k, theta


@partial(np.vectorize, signature='(3)->(4)')
def homogeneous_representation(vector: np.ndarray) -> np.ndarray:
    """
    Homogeneous representation of a vector.

    Parameters
    ----------
    vector : np.ndarray (3,)

    Returns
    -------
    homogeneous representation of vector:  np.array (4,)
    """

    return np.concatenate((vector, np.array([1.])))


def homogeneous_transformation(rot: np.ndarray = np.eye(3), d: np.ndarray = np.zeros((3,))) -> np.ndarray:
    """
    Homogeneous operation constructed from a 3x3 rotation matrix  rot and a 3x1 vector d.

    Parameters
    ----------
    rot : np.ndarray (3,3)
    d   : np.ndarray (3,)

    Returns
    -------
    Homogeneous transformation : np.ndarray (4,4)
    """
    o_1 = np.array([[0, 0, 0, 1]])
    return np.block([[rot, d[:, np.newaxis]], [o_1]])


def trans_x(a: float):
    """
    Homogenenous transformation of a translation on x axis by a.

    Parameters
    ----------
    a : float

    Returns
    -------
    Homogeneous transformation : np.ndarray (4,4)
    """
    return homogeneous_transformation(d=np.array([a, 0., 0.]))


def trans_y(a: float):
    """
    Homogenenous transformation of a translation on y axis by a.

    Parameters
    ----------
    a : float

    Returns
    -------
    Homogeneous transformation np.ndarray (4,4)
    """
    return homogeneous_transformation(d=np.array([0., a, 0.]))


def trans_z(a: float):
    """
    Homogenenous transformation of a translation on z axis by a.

    Parameters
    ----------
    a : float

    Returns
    -------
    Homogeneous transformation : np.ndarray (4,4)
    """
    return homogeneous_transformation(d=np.array([0., 0., a]))


def rot_x(theta: float):
    """
    Homogenenous transformation of a rotation about x axis by theta.

    Parameters
    ----------
    theta : float

    Returns
    -------
    Homogeneous transformation np.ndarray (4,4)
    """
    return homogeneous_transformation(rot=rotation_matrix_x(theta))


def rot_y(theta: float):
    """
    Homogenenous transformation of a rotation about y axis by theta.

    Parameters
    ----------
    theta : float

    Returns
    -------
    Homogeneous transformation : np.ndarray (4,4)
    """
    return homogeneous_transformation(rot=rotation_matrix_y(theta))


def rot_z(theta: float):
    """
    Homogenenous transformation of a rotation about z axis by theta.

    Parameters
    ----------
    theta : float

    Returns
    -------
    Homogeneous transformation : np.ndarray (4,4)
    """
    return homogeneous_transformation(rot=rotation_matrix_z(theta))


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Calculate a rotation matrix in two d case
    Parameters
    ----------
    theta: float

    Returns
    -------
    Rotation matrix: np.ndarray (2,2)
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


@partial(np.vectorize, signature='(2)->(3)')
def homogeneous_representation_2d(vector: np.ndarray) -> np.ndarray:
    """
    Homogeneous representation of a vector.

    Parameters
    ----------
    vector : np.ndarray (2,)

    Returns
    -------
    homogeneous representation of vector:  np.array (3,)
    """

    return np.concatenate((vector, np.array([1.])))


def homogeneous_transformation_2d(rot: np.ndarray = np.eye(2), d: np.ndarray = np.zeros((2,))) -> np.ndarray:
    """
    Homogeneous operation constructed from a 2x2 rotation matrix  rot and a 2x1 vector d.

    Parameters
    ----------
    rot : np.ndarray (2,2)
    d   : np.ndarray (2,)

    Returns
    -------
    Homogeneous transformation : np.ndarray (4,4)
    """
    o_1 = np.array([[0, 0, 1]])
    return np.block([[rot, d[:, np.newaxis]], [o_1]])
