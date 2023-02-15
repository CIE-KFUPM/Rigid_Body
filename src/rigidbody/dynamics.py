import rigidbody.kinematics as kine
import numpy as np
from dataclasses import dataclass


@dataclass
class RigidBody:
    mass: float
    inertia: np.ndarray
    center_mass_position: np.ndarray


def corriolis_matrix_two_d(body_velocity: np.ndarray, rb: RigidBody):
    """

    Parameters
    ----------
    body_velocity
    rb

    Returns
    -------

    """
    m = rb.mass
    vx = body_velocity[0]
    vy = body_velocity[1]
    psi_dot = body_velocity[2]
    rx = rb.center_mass_position[0]
    ry = rb.center_mass_position[1]

    return np.array([[0, 0, -m * (vy + rx * psi_dot)],
                     [0, 0, m * (vx - ry * psi_dot)],
                     [m * (vy + rx * psi_dot), -m * (vx - ry * psi_dot), 0]])


def mass_matrix_two_d(rb: RigidBody):
    """

    Parameters
    ----------
    rb

    Returns
    -------

    """
    m = rb.mass
    rx = rb.center_mass_position[0]
    ry = rb.center_mass_position[1]
    Iz = rb.inertia[0]
    return np.array([[m, 0, -m * ry],
                     [0, m, m * rx],
                     [m * ry, -m * rx, Iz]])


def rigid_body_dynamics_two_d(state: np.ndarray, force_torque: np.ndarray, rb: RigidBody):
    """

    Parameters
    ----------
    state
    force_torque
    rb

    Returns
    -------

    """
    position = state[:3]
    body_velocity = state[3:]
    mass_mat = mass_matrix_two_d(rb)
    corr_mat = corriolis_matrix_two_d(body_velocity, rb)
    position_dot = kine.inverse_analytical_jacobian_2d(position[-1]) @ body_velocity
    body_velocity_dot = np.linalg.solve(mass_mat, force_torque - corr_mat @ body_velocity)
    return np.concatenate((position_dot, body_velocity_dot))
