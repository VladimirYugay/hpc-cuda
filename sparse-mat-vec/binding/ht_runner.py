import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import heat_transfer as ht
mpl.style.use('ggplot')


def runner(solver, num_points, settings):
    time = solver.run(num_points, settings)


def animate(counter, axis, solver, domain):
    if solver.get_plotting_status() == ht.PlottingStatus.UPDATED:
        axis.clear()
        ax.set_ylabel('Temperature')
        axis.set_ylim(bottom=-2.0, top=2.0)
        axis.set_title(f'{ht.get_device_name()}', fontdict={'fontsize': 12})
        axis.plot(domain, solver.get_solution_vector())


# settings
settings = ht.Settings()
settings.conductivity = 0.1
settings.epsilon = 1e-5
settings.steps_to_print = 2000
execution_mode = ht.ExecutionMode.HEAT_CUSPARSE
num_points = 1000


# run solver
solver = ht.PoissonSolver(settings)
runner_thread = threading.Thread(target=runner, args=(solver, num_points, execution_mode))
runner_thread.start()


# run graphics
fig, ax = plt.subplots()
fig.canvas.set_window_title('HPC-AA: 1D Heat Equation Problem')
fig.suptitle(f'Solver type: {ht.mode_to_str(execution_mode)}', fontsize=14)
domain = np.linspace(start=1, stop=num_points, num=num_points)
animation = animation.FuncAnimation(fig, animate, interval=50, fargs=(ax, solver, domain))
plt.show()


runner_thread.join()
