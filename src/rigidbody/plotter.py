import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import rigidbody.transformations as rb
from matplotlib.patches import Polygon


def set_context(a_context):
    """

    Parameters
    ----------
    a_context

    Returns
    -------

    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    sns.set_context(a_context)


def two_d_trajectory_attitude_plot(trajectory: np.ndarray, scale: float = 1., skip=1,
                                   color: str = 'black',
                                   initial_color: str = None) -> plt.Axes:
    """

    Parameters
    ----------
    initial_color
    color: string
    trajectory: np.ndarray (Nt, 3)
    scale : float (>0)
    skip : int (>1)

    Returns
    -------
    Axes : plt.Axes handle to the plot axis
    """

    # Default triangle
    triangle_points = scale * np.array([[0., -1],
                                        [0, 1],
                                        [4, 0]])
    middle = np.mean(triangle_points, axis=0)
    # shift the middle to the origin
    triangle_points[:, 0] -= middle[0]
    triangle_points[:, 1] -= middle[1]

    ax = plt.subplot()
    ax.plot(trajectory[:, 0], trajectory[:, 1])
    _P0 = rb.homogeneous_representation_2d(triangle_points)
    for i in range(trajectory.shape[0] // skip):
        x = trajectory[i * skip]
        _H = rb.homogeneous_transformation_2d(rb.rotation_matrix_2d(x[-1]), d=x[:2])
        _P = _P0 @ _H.T  # _H.T  # apply homogeneous transformation
        if i == 0:
            if initial_color:
                poly = Polygon(_P[:, :2], fill=True, color=initial_color, edgecolor=initial_color)
                ax.add_patch(poly)
                continue

        poly = Polygon(_P[:, :2], fill=True, color=color, edgecolor=color)
        ax.add_patch(poly)

    return ax
