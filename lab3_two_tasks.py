# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)           # displacement along Z-axis
q = np.array([0.2, 0.5, 0.6])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.5]) # displacement along X-axis
alpha = np.zeros(3)       # rotation around X-axis
revolute = [True, True, True] # flags specifying the type of joints

# Desired values of task variables
sigma1_d = np.array([-0.8, 0.0]).reshape(2,1) # Position of the end-effector
sigma2_d = np.array([[0.0]]) # Position of joint 1

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Logging
err_log = [] 
time_log = []
current_time = 0.0

# Simulation initialization
def init():
    global sigma1_d
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    # Reset the desired position of the end effector every repeat
    new_x = np.random.uniform(-2.0, 2.0)
    new_y = np.random.uniform(-2.0, 2.0)
    sigma1_d = np.array([new_x, new_y]).reshape(2, 1)
    return line, path, point

# Simulation loop
def simulate_end_effector(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy, err_log, time_log, current_time
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    # Update control
    # TASK 1: End-Effector Position
    sigma1 = T[-1][:2,3].reshape(2,1)                 # Current position of the end-effector
    err1 = sigma1_d - sigma1                          # Error in Cartesian position
    J1 = J[:2,:3]                                     # Jacobian of the first task
    J1bar = J1.copy()
    
    J1bar_dls = DLS(J1bar, 0.5)
    dq1 = J1bar_dls @ err1                            # Velocity for the first task
    
    J1bar_pseudo_inv = np.linalg.pinv(J1bar)
    product_J_1 = J1bar_pseudo_inv @ J1bar            # Pseudo-invese is used to prevent the violation of task priorities
    P1 = np.eye(product_J_1.shape[0]) - product_J_1   # Null space projector
    
    # TASK 2: Joint 1 Position
    sigma2 = q[0]                              # Current position of joint 1
    err2 = sigma2_d - sigma2                   # Error in joint position
    J2 = np.array([1, 0, 0]).reshape(1,3)      # Jacobian of the second task
    J2bar = J2 @ P1                            # Augmented Jacobian
    
    J2bar_dls = DLS(J2bar, 0.5)
    dq12 = dq1 + J2bar_dls@(err2 - J2@dq1)     # Velocity for both tasks

    J2bar_pseudo_inv = np.linalg.pinv(J2bar)   # Pseudo-invese is used to prevent the violation of task priorities
    product_J_2 = J2bar_pseudo_inv@J2bar
    P2 = P1 - product_J_2
    
    # Combining tasks
    q = q + dq12.reshape(1,3)[0] * dt # Simulation update (dq12 has shape (3,1), must be reshape to match the shape of original q (,3))

    # Log the q and timestamp
    err_log.append([np.linalg.norm(err1.copy()), err2[0][0].copy()]) # log the norm of end-effector and desired position distance error, and joint 1 position error
    current_time = current_time + dt
    time_log.append(current_time)

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    return line, path, point

# Simulation loop
def simulate_first_joint(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy, err_log, time_log, current_time
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    # Update control
    
    # TASK 1: Joint 1 Position
    sigma1 = q[0]                                     # Current position of joint 1
    err1 = sigma2_d - sigma1                          # Error in joint position (using joint target)
    J1 = np.array([1, 0, 0]).reshape(1,3)             # Jacobian of the joint task
    J1bar = J1.copy()
    
    J1bar_dls = DLS(J1bar, 0.5)
    dq1 = J1bar_dls @ err1                            # Velocity for the first task
    
    J1bar_pseudo_inv = np.linalg.pinv(J1bar)
    product_J_1 = J1bar_pseudo_inv @ J1bar            # Pseudo-inverse for null space
    P1 = np.eye(product_J_1.shape[0]) - product_J_1   # Null space projector
    
    # TASK 2: End-Effector Position
    sigma2 = T[-1][:2,3].reshape(2,1)                 # Current EE position
    err2 = sigma1_d - sigma2                          # Error in Cartesian position (using EE target)
    J2 = J[:2,:3]                                     # Jacobian of the EE task
    J2bar = J2 @ P1                                   # Augmented Jacobian (projected into null space of Task 1)
    
    J2bar_dls = DLS(J2bar, 0.5)
    dq12 = dq1 + J2bar_dls@(err2 - J2@dq1)            # Velocity for both tasks

    J2bar_pseudo_inv = np.linalg.pinv(J2bar)
    product_J_2 = J2bar_pseudo_inv@J2bar
    P2 = P1 - product_J_2
    
    # Combining tasks
    q = q + dq12.reshape(1,3)[0] * dt 

    # Log the q and timestamp 
    err_log.append([np.linalg.norm(err2.copy()), err1[0][0].copy()]) 
    current_time = current_time + dt
    time_log.append(current_time)

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    return line, path, point

# Run simulation. Change the simulation function name (simulate_end_effector, simulate_first joint) for each scenario
animation = anim.FuncAnimation(fig, simulate_end_effector, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Visualize joint position change overtime
err_log = np.array(err_log) 

fig2, ax_q = plt.subplots()
ax_q.plot(time_log, err_log[:, 0], label='e1 (end-effector position)')
ax_q.plot(time_log, err_log[:, 1], label='e2: (joint 1 position)')

ax_q.set_title('Task-Priority (two tasks)')
ax_q.set_xlabel('Time[s]')
ax_q.set_ylabel('Error[1]')
ax_q.legend()
ax_q.grid(True)

plt.show()