# Rigid Body
A simple python package contains basic rigid body formalism, like rotations, homogeneous transformations, 
kinematics as well as some plotting functionalities.

Examples:
```python
import numpy as np
import rigidbody.transformations as trans
import rigidbody.plotter as plotter
R = trans.rotation_matrix_z(np.pi/6)@trans.rotation_matrix_y(np.pi/3)@trans.rotation_matrix_z(np.pi/6)
trans.euler_angles_from_rot_matrix(R)*180/np.pi
```

Check `notebooks` folder for more.
