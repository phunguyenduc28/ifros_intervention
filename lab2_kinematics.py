# Import necessary libraries
from lab2_robotics import * # Import our library (includes Numpy)
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (planar 2 link manipulator)
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 

# Simulation params
dt = 0.01 # Sampling time
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Memory
PPx = []
PPy = []
q_log = [q.copy()] 
time_log = [0]

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    return line, path

# Simulation loop
def simulate(t):
    global d, q, a, alpha
    global PPx, PPy, q_log, time_log
    
    # Update robot
    T = kinematics(d, q, a, alpha)
    dq = np.array([1, 2])
    q = q + dt * dq

    # Log the q
    q_log.append(q.copy())
    time_log.append(t.copy())
    
    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    
    return line, path

# Simulation plot preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Kinematics')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_aspect('equal')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'r-', lw=1) # End-effector path

# Run simulation and visualize
animation = anim.FuncAnimation(fig, simulate, tt, 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()

# Visualize joint position change overtime
q_log = np.array(q_log) 

fig2, ax_q = plt.subplots()
ax_q.plot(time_log, q_log[:, 0], label='q1')
ax_q.plot(time_log, q_log[:, 1], label='q2')

ax_q.set_title('Joint Position')
ax_q.set_xlabel('Time[s]')
ax_q.set_ylabel('Angle[rad]')
ax_q.legend()
ax_q.grid(True)

plt.show()