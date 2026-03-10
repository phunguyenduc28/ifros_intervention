import numpy as np # Import Numpy

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    d_matrix = np.array([[1, 0, 0, 0], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, d], 
                        [0, 0, 0, 1]])
    theta_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    a_matrix = np.array([[1, 0, 0, a], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])
    alpha_matrix = np.array([[1, 0, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha), 0],
                            [0, np.sin(alpha), np.cos(alpha), 0],
                            [0, 0, 0, 1]])
    # 2. Multiply matrices in the correct order (result in T).
    T = d_matrix @ theta_matrix @ a_matrix @ alpha_matrix
    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    
    # For each set of DH parameters:
    for i in range(len(d)):
        d_i, theta_i, a_i, alpha_i = d[i], theta[i], a[i], alpha[i] # a set of DH parameter

        # 1. Compute the DH transformation matrix.
        T_local = DH(d_i, theta_i, a_i, alpha_i)

        # 2. Compute the resulting accumulated transformation from the base frame.
        T_i = T[-1] @ T_local

        # 3. Append the computed transformation to T.
        T = T + [T_i]
    
    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    J = []
    o_n = T[-1][:3, 3]
    # 2. For each joint of the robot
    for i in range(1, len(T)):
        # Check if joint is revolute or not
        rev_joint = revolute[i-1]
        # Extract individual transformation T_{i-1}
        T_i_1 = T[i-1] 
        # Extract z and o
        z_i_1 = T_i_1[:3, 2] 
        o_i_1 = T_i_1[:3, 3]

        # Construct a Jacobian column based on the joint type
        if rev_joint:
            # Revolute joint
            diff_o = o_n - o_i_1
            v_transpose = np.cross(z_i_1, diff_o)
            omega_transpose = z_i_1.T
            J_i = np.hstack((v_transpose, omega_transpose))
        else:
            # Prismatic joint
            v_transpose = z_i_1.T
            omega_transpose = np.zeros(3)
            J_i = np.hstack((v_transpose, omega_transpose))
        
        J = J + [J_i]

    # Full Jacobian matrix
    J = np.transpose(np.array(J))
    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    gram = A @ A.T
    inv = A.T @ np.linalg.inv((gram+ damping**2 * np.eye(gram.shape[0])))
    return inv

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P