# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Implementation method of Geometric Jacobians
control_types = ["pseudoinverse", "transpose", "dls"]

# Simulation params
max_t = 10
dt = 1.0/60.0

# Memory
err_array = []
time_array = []
time_method_log = []
err_method_log = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t, control_type):
    global d, q, a, alpha, revolute, sigma_d
    global PPx, PPy, err_array, time_array
    global control_types

    # Update robot
    T = kinematics(d, q, a, alpha)
    J = jacobian(T, revolute) 

    # Update control
    sigma = T[-1][:2,3]        # Position of the end-effector
    err = sigma_d - sigma      # Control error
    err_array.append(np.linalg.norm(err)) # Log the control error
    time_array.append(t) # Log the timestamp
    
    # Control solution
    J_cropped = J[:2,:2] # crop top-left 2x2 Jacobian matrix for plannar position control 
    if control_type == control_types[0]: 
        J_pseudo_inv = np.linalg.inv(np.transpose(J_cropped) @ J_cropped) @ np.transpose(J_cropped)
        dq = J_pseudo_inv @ err            # Pseudo-inverse Jacobian
    elif control_type == control_types[1]:
        dq = np.transpose(J_cropped) @ err # Transpose Jacobian
    elif control_type == control_types[2]: 
        lamda = 0.5
        dq = DLS(J_cropped, lamda) @ err   # DLS solution
    
    q += dt * dq # Update joints configuration
    
    # Update drawing
    P = robotPoints2D(T)
    line.set_data(P[0,:], P[1,:])
    PPx.append(P[0,-1])
    PPy.append(P[1,-1])
    path.set_data(PPx, PPy)
    point.set_data([sigma_d[0]], [sigma_d[1]]) # matplotlib 3.10 requires passing an array

    return line, path, point

# Iterate the resolved rate control simulation with pseudo-ivernse, tranpose, and dls solution of Geometric Jacobian computation
for control_type in control_types:
    # Robot definition
    d = np.zeros(2)           # displacement along Z-axis
    q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
    a = np.array([0.75, 0.5]) # displacement along X-axis
    alpha = np.zeros(2)       # rotation around X-axis 
    revolute = [True, True]
    sigma_d = np.array([0.0, 1.0]) # desired end-effector
    K = np.diag([1, 1])

    PPx = []
    PPy = []
    err_array = []
    time_array = []

    print(f"Simulating with {control_type} implementation")
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
    ax.set_title(f'Simulation with {control_type} implementation')
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2) # Robot structure
    path, = ax.plot([], [], 'c-', lw=1) # End-effector path
    point, = ax.plot([], [], 'rx') # Target

    # Run simulation
    animation = anim.FuncAnimation(fig, simulate, np.arange(0, max_t, dt), fargs=(control_type,), 
                                    interval=10, blit=True, init_func=init, repeat=False)
    plt.show()

    err_method_log.append(err_array.copy())
    time_method_log.append(time_array.copy())


# Visual norm error change over time with each method
fig2, ax_err = plt.subplots()
ax_err.plot(time_method_log[0], err_method_log[0], label=control_types[0])
ax_err.plot(time_method_log[1], err_method_log[1], label=control_types[1])
ax_err.plot(time_method_log[2], err_method_log[2], label=control_types[2])

ax_err.set_title('Resolved rate motion control')
ax_err.set_xlabel('Time[s]')
ax_err.set_ylabel('Error[m]')
ax_err.legend()
ax_err.grid(True)

plt.show()

