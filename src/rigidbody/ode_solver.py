import numpy as np
from typing import Callable


def euler_one_step(fun, x, dt, *args) -> np.ndarray:
    return x + dt*fun(x, *args)


def rk_one_step(fun, x, dt, *args) -> np.ndarray:
    k1 = fun(x, *args)
    k2 = fun(x + 0.5 * dt * k1, *args)
    k3 = fun(x + 0.5 * dt * k2, *args)
    k4 = fun(x + dt * k3, *args)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
