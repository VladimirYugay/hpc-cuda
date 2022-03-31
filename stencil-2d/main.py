import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from solver import interface as it
import threading

def runner(solver):
    time = solver.run()
    print(f'elapsed time: {time}, s')


def animate(counter, axis, solver, minmax):
    axis.cla()
    solution = solver.get_solution()
    image = axis.imshow(np.reshape(solution, field_shape), cmap='coolwarm')
    image.set_clim(minmax[0], minmax[1])


# set solver config
config = it.Configuration()
config.num_1d_grid_points = 600
config.num_iterations = 100000
config.steps_per_print = 100
config.conductivity = 0.1
config.mode = it.ExecutionMode.CPU
#config.mode = it.ExecutionMode.GPU_SHR_MEM
#config.mode = it.ExecutionMode.GPU_GLOB_MEM
config.block_size_x = 32
config.block_size_y = 10
config.eps = 1e-7
config.end_time = 0.5
it.print_config(config)


# create initial field
field_shape = (config.num_1d_grid_points, config.num_1d_grid_points)
field = np.zeros(shape=field_shape, dtype=np.float32)

# set initial and boundary conditions
frequency_x = 8
frequency_y = 1
for j in range(1, config.num_1d_grid_points):
    for i in range(0, config.num_1d_grid_points):
        factor_x = frequency_x * np.pi * i / (config.num_1d_grid_points - 1)
        factor_y = frequency_y * np.pi * j / (config.num_1d_grid_points - 1)
        field[j][i] = np.cos(factor_x) * np.cos(factor_y)

minmax = (np.amin(field), np.amax(field))

# init solver
solver = it.Solver(config)
solver.init(field.flatten(order='C'))

# run solver
runner_thread = threading.Thread(target=runner, args=(solver, ))
runner_thread.start()


#create a plot
fig, ax = plt.subplots()
fig.canvas.set_window_title('HPC-AA: 2D stencil computation')

# show the field after at the beginning
solution = solver.get_solution()
image = ax.imshow(np.reshape(solution, field_shape), cmap='coolwarm')
image.set_clim(minmax[0], minmax[1])
cbar = fig.colorbar(image, extend='both')

# run animation
animation = animation.FuncAnimation(fig, animate, interval=20, fargs=(ax, solver, minmax))
plt.show()


runner_thread.join()