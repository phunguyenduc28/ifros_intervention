from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model - 3-link manipulator
d = np.zeros(3)                      # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.6])    # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.5])       # displacement along X-axis
alpha = np.zeros(3)                  # rotation around X-axis
revolute = [True, True, True]        # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
obstacle_pos_1 = np.array([0.0, 1.0]).reshape(2,1)
obstacle_r_1 = 0.5

obstacle_pos_2 = np.array([0.7, -0.5]).reshape(2,1)
obstacle_r_2 = 0.3

obstacle_pos_3 = np.array([-0.5, -0.7]).reshape(2,1)
obstacle_r_3 = 0.4

tasks = [ 
          Obstacle2D("Obstacle avoidance", obstacle_pos_1, np.array([obstacle_r_1, obstacle_r_1+0.05]), 3),
          Obstacle2D("Obstacle avoidance", obstacle_pos_2, np.array([obstacle_r_2, obstacle_r_2+0.05]), 3),
          Obstacle2D("Obstacle avoidance", obstacle_pos_3, np.array([obstacle_r_3, obstacle_r_3+0.05]), 3),
          Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), 3)
        ] 

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.add_patch(patch.Circle(obstacle_pos_1.flatten(), obstacle_r_1, color='red', alpha=0.3))
ax.add_patch(patch.Circle(obstacle_pos_2.flatten(), obstacle_r_2, color='purple', alpha=0.3))
ax.add_patch(patch.Circle(obstacle_pos_3.flatten(), obstacle_r_3, color='green', alpha=0.3))
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

PPx = []
PPy = []

err_log = []
time_log = []
current_time = 0.0

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    keywords = ['position', 'configuration']

    # Reset the desired position of the end effector every repeat
    for task in tasks:
        task_name = task.name
        if "joint" in task_name.lower():
            continue
        if any(word in task_name.lower() for word in keywords):
            new_x = np.random.uniform(-1.5, 1.5)
            new_y = np.random.uniform(-1.5, 1.5)
            desired = task.getDesired()
            desired[0][0] = new_x
            desired[1][0] = new_y
            task.setDesired(desired)
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global err_log, time_log, current_time
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # Initialize null-space projector
    P = np.eye(3) # robot has 3DOF
    # Initialize output vector (joint velocity)
    dq = np.zeros((robot.dof,1))
    # Loop over tasks
    err_tasks = []
    for i in range(len(tasks)):
        # Update task state
        tasks[i].update(robot)

        task_name = tasks[i].name 
        if "position" in task_name.lower():
            if "joint" in task_name.lower():
                err_tasks.append(tasks[i].err[0]) # Joint error is a np array so need extracting to get a scalar
            else:
                pos_norm_err = np.linalg.norm(tasks[i].err)
                err_tasks.append(pos_norm_err)
        elif "orientation" in task_name.lower():
            quat_xyz_err = tasks[i].err.flatten().tolist()
            quat_w_err = tasks[i].quat_w_err.flatten().tolist()
            r = R.from_quat(quat_xyz_err + quat_w_err)
            euler = r.as_euler('zyx', degrees=True) # Convert to Euler (yaw, pitch, roll) in degrees
            yaw_err = euler[0] 
            err_tasks.append(abs(yaw_err))
        elif "configuration" in task_name.lower():
            pos_norm_err = np.linalg.norm(tasks[i].err[:2,])
            err_tasks.append(pos_norm_err)

            quat_xyz_err = tasks[i].err[2:,].flatten().tolist()
            quat_w_err =  tasks[i].quat_w_err.flatten().tolist()
            r = R.from_quat(quat_xyz_err + quat_w_err)
            euler = r.as_euler('zyx', degrees=True) # Convert to Euler (yaw, pitch, roll) in degrees
            yaw_err = euler[0] 
            err_tasks.append(abs(yaw_err))
        elif "obstacle" in task_name.lower():
            norm_dist = np.linalg.norm(tasks[i].dist_ee_obs)
            err_tasks.append(norm_dist)

        # Move to next task if not active
        if tasks[i].isActive == 0:                
            continue
        
            # Compute augmented Jacobian
        Jtask = tasks[i].getJacobian()
        Ji = np.zeros((Jtask.shape[0], robot.dof)) 
        Ji[:, :tasks[i].link] = Jtask # broadcast the link Jacobian to the task Jacobian
        Jbar_i = Ji @ P

        # Compute task velocity 
        Jbar_dls = DLS(Jbar_i, 0.2)
        feedforward_err = tasks[i].getFeedforwardVelocity() + tasks[i].getGainMatrix() @ tasks[i].err
        dq_task = Jbar_dls @ (tasks[i].isActive*feedforward_err- Ji@dq)

        # Accumulate velocity
        dq = dq + dq_task

        # Update null-space projector
        P = P - np.linalg.pinv(Jbar_i)@Jbar_i

    # Logging data for data visualization
    err_log.append(np.array(err_tasks).tolist())
    current_time += dt
    time_log.append(current_time)

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    
    point.set_data(tasks[3].getDesired()[0], tasks[3].getDesired()[1]) # get the desired position of the end-effector
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

labels = [task.name.lower() for task in tasks]

# Visualize joint position change overtime
err_log = np.array(err_log) 

fig2, ax_q = plt.subplots()
if len(tasks) == 1 and tasks[0].name == "Configuration task": # Configuration task requires special ploting
    ax_q.plot(time_log, err_log[:, 0], label=f'e1 End-effector position')
    ax_q.plot(time_log, err_log[:, 1], label=f'e2 End-effector orientation')
else:
    for i in range(len(tasks)):
        err_string = f"d_{i+1}" if "obstacle" in labels[i] else f"e_{i+1}" 
        ax_q.plot(time_log, err_log[:, i], label=f'{err_string} ({labels[i]})')

ax_q.set_title('Task-Priority')
ax_q.set_xlabel('Time[s]')
ax_q.set_ylabel('Error[1]')
ax_q.legend()
ax_q.grid(True)

plt.show()
