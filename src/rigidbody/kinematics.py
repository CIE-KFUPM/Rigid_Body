import rigidbody.transformations as trans
import numpy as np
from functools import partial


def j_omega(angles: np.ndarray) -> np.ndarray:
    """
    Calculate rotation component of the analytical Jacobian of a rigid body where the angles
    are represented the roll,pitch, and yaw. The Jacobian relates between the
    angular velocities in the fixed frame to the angular
    velocities in the body frame.

    Parameters
    ----------
    angles: np.ndarray (3,)

    Returns
    -------
    Jacobian: np.ndarray (3,3)
    """

    # phi ->angles[0]
    # theta ->angles[1]
    # psi ->angles[2]
    return np.array([[1., 0., -np.sin(angles[1])],
                     [0, np.cos(angles[0]), np.sin(angles[0]) * np.cos(angles[1])],
                     [0, -np.sin(angles[0]), np.cos(angles[0]) * np.cos(angles[1])]])


def j_v(angles: np.ndarray) -> np.ndarray:
    """
    Calculate linear component of the analytical Jacobian of a rigid body where the angles
    are represented the roll,pitch, and yaw. The Jacobian relates between the
    linear velocities in the fixed frame to the linear
    velocities in the body frame.

    Parameters
    ----------
    angles: np.ndarray (3,)

    Returns
    -------
    Jacobian: np.ndarray (3,3)
    """
    return trans.rotation_matrix_from_roll_pitch_yaw(angles)


def analytical_jacobian_3d(angles: np.ndarray) -> np.ndarray:
    """
    Calculate the analytical Jacobian of a rigid body where the angles
    are represented the roll,pitch, and yaw. The Jacobian relates between the
    linear and angular velocities in the fixed frame to the linear and angular
    velocities in the body frame.

    Parameters
    ----------
    angles: np.ndarray (3,)

    Returns
    -------
    Jacobian: np.ndarray (6,6)
    """
    o_3_3 = np.zeros((3, 3))
    return np.block([[j_v(angles), o_3_3],
                     [o_3_3, j_omega(angles)]])


def inverse_analytical_jacobian_2d(psi: float) -> np.ndarray:
    """

    Calculate inverse of a two-dimensional Jacobian.

    Parameters
    ----------
    psi : float

    Returns
    -------
    Jacobian: np.ndarray (3,3)
    """
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1.]])
