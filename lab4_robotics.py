from lab2_robotics import * # Includes numpy import
from scipy.spatial.transform import Rotation as R 

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

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
    Jlink = J[:,:link]
    return Jlink


'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof
    
    '''Method that returns a link transformation'''
    # Updated in Part 2
    def getLinkTransform(self, link):
        return self.T[link] # slice the list of transformations to the transform to the link
    
    '''Method that returns a link Jacobian'''
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)

'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.feedforward_v = None
        self.K = None
        self.isActive = 0
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    def rotation_matrix(self, angles):
        def rot_x(angle):
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ], dtype=float)

        def rot_y(angle):
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=float)
        def rot_z(angle):
            return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=float)
        x, y, z = angles[0], angles[1], angles[2]
        return rot_x(x) @ rot_y(y) @ rot_z(z)
    
    def quaternion_from_euler_scipy(self, r):
        r = R.from_matrix(r)
        quat = r.as_quat(scalar_first=True)
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]
        epsilon = np.array([[x], [y], [z]])

        return w, epsilon
    
    def getFeedforwardVelocity(self):
        return self.feedforward_v
    def setFeedforwardVelocity(self, value):
        self.feedforward_v = value
    
    def getGainMatrix(self):
        return self.K
    def setGainMatrix(self, value):
        self.K = value
'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.setDesired(desired)
        self.err = np.zeros((2,1)) # Initialize with proper dimensions
        self.link = link
        self.feedforward_v = np.zeros((2,1))
        self.K = np.eye(2)
        self.isActive = 1
        
    def update(self, robot):
        self.J = robot.getLinkJacobian(self.link)[:2,:]   # Update task Jacobian (3DOFs - 2 positions)
        link_position = robot.getLinkTransform(self.link)[:2,3].reshape(2,1) 
        link_desired = self.getDesired()
        self.err = link_desired - link_position # Update task error
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.err = np.zeros((3,1))# Initialize with proper dimensions
        self.quat_w_err = 0 
        rotation_desired = self.rotation_matrix(np.vstack([np.zeros((2,1)), desired]).flatten().tolist()) # conver the desired angles to rotation matrix

        self.setDesired(rotation_desired)
        self.link = link

        self.feedforward_v = np.zeros((3,1))
        self.K = np.eye(3)

        self.isActive = 1
        
    def update(self, robot):
        self.J = robot.getLinkJacobian(self.link)[3:,:]   # Update task Jacobian (3DOFs - 3 angle)
        r_cur = robot.getLinkTransform(self.link)[:3,:3].reshape(3,3) # current ee orientation in rotation matrix
        w, e = self.quaternion_from_euler_scipy(r_cur) # quaternion reprenstation of the current rotaion matrix
        r_d = self.getDesired() # desired ee orientation in rotation matrix
        w_d, e_d = self.quaternion_from_euler_scipy(r_d) # quaternion reprenstation of the desired rotaion matrix
        self.quat_w_err = w*w_d + e.T@e_d
        quat_e_err = w*e_d - w_d*e - np.cross(e.flatten(), e_d.flatten()).reshape((3,1))
        self.err = quat_e_err
        
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.err = np.zeros((5,1)) # Initialize with proper dimensions
        self.quat_w_err = 0
        self.setDesired(desired)
        self.link = link

        self.feedforward_v = np.zeros((5,1))
        self.K = np.eye(5)

        self.isActive = 1

        
    def update(self, robot):
        # Position Jacobian
        Jpos = robot.getLinkJacobian(self.link)[:2,:] 
        # Position error
        link_position = robot.getLinkTransform(self.link)[:2,3].reshape(2,1) 
        link_desired = self.getDesired()[:2,].reshape(2,1)
        pos_err = (link_desired - link_position).reshape((2,1)) # Update task error

        # Orientation Jacobian
        Jorr = robot.getLinkJacobian(self.link)[3:,:] 
        # Orientation error
        r_cur = robot.getLinkTransform(self.link)[:3,:3].reshape(3,3) # current ee orientation in rotation matrix
        w, e = self.quaternion_from_euler_scipy(r_cur)
        angle_d = np.vstack([np.zeros((2,1)), self.getDesired()[2:,]])
        r_d = self.rotation_matrix(angle_d.flatten().tolist()) # desired ee orientation in rotation matrix
        w_d, e_d = self.quaternion_from_euler_scipy(r_d)
        self.quat_w_err = w*w_d + e.T@e_d
        quat_e_err = w*e_d - w_d*e - np.cross(e.flatten(), e_d.flatten()).reshape((3,1))

        self.J = np.vstack((Jpos, Jorr))
        self.err = np.vstack((pos_err, quat_e_err))
''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.err = np.zeros(1) # Initialize with proper dimensions
        self.setDesired(desired)
        self.link = link
        self.isActive = 1

        self.feedforward_v = np.zeros((1,1))
        self.K = np.eye(1)

    def update(self, robot):
        joint_ind = int(self.link - 1)
        joint_pos_cur = robot.getJointPos(joint_ind)

        joint_pos_d = self.getDesired()[0][0]

        J = np.zeros(robot.getDOF())
        J[joint_ind] = 1
        # self.J = J.reshape((1, robot.getDOF()))
        self.J = np.array([[1]])
        self.err = joint_pos_d - joint_pos_cur

class Obstacle2D(Task):
    def __init__(self, name, desired, obs_radius, link):
        super().__init__(name, desired)
        self.setDesired(desired)
        self.err = np.zeros((2,1)) # Initialize with proper dimensions
        self.link = link
        self.feedforward_v = np.zeros((2,1))
        self.K = np.eye(2)
        self.r_alpha = obs_radius[0]
        self.r_delta = obs_radius[1]
        self.isActive = 0

        self.dist_ee_obs = None
        
    def update(self, robot):
        self.J = robot.getLinkJacobian(self.link)[:2,:]   # Update task Jacobian (3DOFs - 2 positions)
        link_position = robot.getLinkTransform(self.link)[:2,3].reshape(2,1) 
        obs_position = self.getDesired()
        dist_ee_obs = link_position - obs_position 
        self.dist_ee_obs = dist_ee_obs
        norm_dist_ee_obs = np.linalg.norm(dist_ee_obs)

        # Update the task active flag based on the distance to the obstacle
        if self.isActive == 0 and norm_dist_ee_obs <= self.r_alpha:
            self.isActive = 1
        elif self.isActive == 1 and norm_dist_ee_obs >= self.r_delta:
            self.isActive = 0

        self.err = dist_ee_obs / norm_dist_ee_obs # Update task error

class JointLimit(Task):
    def __init__(self, name, desired, threshold, link):
        super().__init__(name, desired)
        self.err = np.array([[1]])
        self.setDesired(desired)
        self.threshold = threshold
        self.link = link

        self.isActive = 0
        self.q = None

        self.feedforward_v = np.zeros((1,1))
        self.K = np.eye(1)


    def update(self, robot):
        joint_ind = int(self.link - 1)
        q_cur = robot.getJointPos(joint_ind)
        self.q = q_cur[0]

        q_min = self.getDesired()[0][0]
        q_max = self.getDesired()[1][0]

        alpha = self.threshold[0][0] 
        delta = self.threshold[1][0]

        J = np.zeros(robot.getDOF())
        J[joint_ind] = 1
        self.J = np.array([[1]])

        if self.isActive == 0 and q_cur >= q_max - alpha:
            self.isActive = -1
        elif self.isActive == 0 and q_cur <= q_min + alpha:
            self.isActive = 1
        elif self.isActive == -1 and q_cur <= q_max - delta:
            self.isActive = 0
        elif self.isActive == 1 and q_cur >= q_min + delta:
            self.isActive = 0