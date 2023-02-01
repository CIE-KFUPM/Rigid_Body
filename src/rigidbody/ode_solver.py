import numpy as np
from typing import Callable


def euler(fun: Callable[[np.ndarray, float], np.ndarray],
          tspan: np.ndarray, x0: np.ndarray, *args) -> np.ndarray:
    """
    Euler ODE Solver

    Parameters
    ----------
    fun
    tspan
    x0

    Returns
    -------

    """

    if isinstance(x0, float):
        x = np.zeros_like(tspan)
    else:
        x = np.zeros((tspan.shape[0], x0.shape[0]))

    x[0] = x0
    for i in range(1, tspan.shape[0]):
        x[i] = x[i - 1] + fun(tspan[i - 1], x[i - 1]) * (tspan[i] - tspan[i - 1])
    return x


def rk_4(fun: Callable[[np.ndarray, float], np.ndarray],
         tspan: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Runge-Kutta fourth order ODE solver.

    Parameters
    ----------
    fun
    tspan
    x0

    Returns
    -------

    """
    if isinstance(x0, float):
        x = np.zeros_like(tspan)
    else:
        x = np.zeros((tspan.shape[0], x0.shape[0]))
    x[0] = x0
    for i in range(1, tspan.shape[0]):
        h = (tspan[i] - tspan[i - 1])
        k1 = fun(tspan[i - 1], x[i - 1])
        k2 = fun(tspan[i - 1] + 0.5 * h, x[i - 1] + 0.5 * h * k1)
        k3 = fun(tspan[i - 1] + 0.5 * h, x[i - 1] + 0.5 * h * k2)
        k4 = fun(tspan[i - 1] + h, x[i - 1] + h * k3)
        x[i] = x[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x
