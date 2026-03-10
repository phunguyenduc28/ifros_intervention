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
    # Code almost identical to the one from lab2_robotics...

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

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.setDesired(desired)
        self.err = np.zeros((2,1)) # Initialize with proper dimensions
        
    def update(self, robot):
        self.J = robot.getEEJacobian()[0:2,:]   # Update task Jacobian (3DOFs - 2 positions)
        ee_position = robot.getEETransform()[:2,3].reshape(2,1) 
        ee_desired = self.getDesired()
        self.err = ee_desired - ee_position # Update task error
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.err = np.zeros((3,1))# Initialize with proper dimensions
        rotation_desired = self.rotation_matrix(desired.flatten().tolist()) # conver the desired angles to rotation matrix
        self.setDesired(rotation_desired)

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
        
    def update(self, robot):
        self.J = robot.getEEJacobian()[3:,:]   # Update task Jacobian (3DOFs - 3 angle)
        r_cur = robot.getEETransform()[:3,:3].reshape(3,3)
        w, e = self.quaternion_from_euler_scipy(r_cur) # quaternion reprenstation of the current rotaion matrix
        r_d = self.getDesired()
        w_d, e_d = self.quaternion_from_euler_scipy(r_d) # quaternion reprenstation of the desired rotaion matrix
        quat_w_err = w*w_d + e.T@e_d
        quat_e_err = w*e_d - w_d*e - np.cross(e.flatten(), e_d.flatten()).reshape((3,1))
        self.err = quat_e_err
        
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.err = np.zeros((6,1)) # Initialize with proper dimensions
        self.setDesired(desired)

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
        
    def update(self, robot):
        # Position Jacobian
        Jpos = robot.getEEJacobian()[:2,:] 
        # Position error
        ee_position = robot.getEETransform()[:2,3].reshape(2,1) 
        ee_desired = self.getDesired()[:2,].reshape(2,1)
        pos_err = (ee_desired - ee_position).reshape((2,1)) # Update task error

        # Orientation Jacobian
        Jorr = robot.getEEJacobian()[3:,:] 
        # Orientation error
        r_cur = robot.getEETransform()[:3,:3].reshape(3,3)
        w, e = self.quaternion_from_euler_scipy(r_cur)
        angle_d = self.getDesired()[2:,].reshape(3,1)
        r_d = self.rotation_matrix(angle_d.flatten().tolist())
        w_d, e_d = self.quaternion_from_euler_scipy(r_d)
        quat_w_err = w*w_d + e.T@e_d
        quat_e_err = w*e_d - w_d*e - np.cross(e.flatten(), e_d.flatten()).reshape((3,1))

        self.J = np.vstack((Jpos, Jorr))
        self.err = np.vstack((pos_err, quat_e_err))
''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.err = np.zeros(1) # Initialize with proper dimensions
        self.setDesired(desired)

    def update(self, robot):
        joint_ind = int(self.getDesired()[0][0] - 1)
        joint_pos_cur = robot.getJointPos(joint_ind)

        joint_pos_d = self.getDesired()[1][0]

        J = np.zeros(robot.getDOF())
        J[joint_ind] = 1
        self.J = J.reshape((1, robot.getDOF()))
        self.err = joint_pos_d - joint_pos_cur