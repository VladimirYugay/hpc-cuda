import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from progress.bar import Bar
import h5py as h5

import dense_hpc_aa as interface
from dense_hpc_aa import KernelType as krnl


def compute_ops(matrix_size, repeats):
    return repeats * (2 * matrix_size - 1) * matrix_size * matrix_size


def compute_glops(time, config):
    operations = compute_ops(config.matrix_size, config.num_repeats)
    time /= 1000.0
    return operations / (time * 1024 * 1024 * 1024)


# base configuration
config = interface.Configuration()
config.print_matrix = True
config.print_info = True
config.matrix_size = 8
config.tile_size = 4
config.num_repeats = 100


# set up plotting style
mpl.style.use('ggplot')
mpl.rcParams['lines.markersize'] = 6.5
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 8
color_array = mpl.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots()


markers = {
    krnl.KERNEL_CPU: "+",
    krnl.KERNEL_GLOBAL: "p",
    krnl.KERNEL_TILED: "v",
    krnl.KERNEL_COALESCED: "*",
    krnl.KERNEL_COALESCED_DYM: "s",
    krnl.KERNEL_OVERLAPPED: "X",
    krnl.KERNEL_CUBLAS: "o"
}

tile_to_colors = {4: color_array[0], 8: color_array[1], 16: color_array[2], 32: color_array[3]}


# generate sizes that we're going to test
sizes = [2**i for i in range(5, 11)]


# collect performance data for cublas
with_cublas = False
if with_cublas:
    with Bar('CUBLAS...    ', max=len(sizes)) as bar:
        cublas = []
        for size in sizes:
            config.matrix_size = size
            config.kernel_type = krnl.KERNEL_CUBLAS

            time = interface.run(config)
            cublas.append(compute_glops(time, config))
            bar.next()

        ax.plot(sizes,
                cublas,
                color='green',
                marker=markers[krnl.KERNEL_CUBLAS],
                linestyle='solid',
                label=interface.kernel_type_to_str(config.kernel_type))


tiles = [2**i for i in range(2, 6)]
kernels = [krnl.KERNEL_GLOBAL, krnl.KERNEL_TILED, krnl.KERNEL_COALESCED, krnl.KERNEL_COALESCED_DYM, krnl.KERNEL_OVERLAPPED]


# prepare meta-data
db = h5.File('./database.hdf5', 'w')
db.attrs['device_name'] = interface.get_device_name()
db.attrs['num_repeats'] = config.num_repeats
db.create_dataset("matrix_sizes", data=sizes)


max = len(tiles) * len(kernels) * len(sizes)
with Bar('Processing...', max=max) as bar:
    for kernel in kernels:
        kernel_name = interface.kernel_type_to_str(kernel)
        kernel_group = db.create_group(f'{kernel_name}')
        for tile in tiles:
            performance = []
            for size in sizes:
                config.matrix_size = size
                config.kernel_type = kernel
                config.tile_size = tile

                time = interface.run(config)
                performance.append(compute_glops(time, config))
                bar.next()

            description = f'{kernel_name} - tile={tile}'
            ax.plot(sizes,
                    performance,
                    color=tile_to_colors[tile],
                    marker=markers[config.kernel_type],
                    linestyle='None',
                    label=description)

            kernel_group.create_dataset(f'{tile}', data=performance)


ax.set_xlabel('Matrix Size')
ax.set_xscale('log', basex=2)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks(sizes)


ax.set_ylabel('Performance, GFLOP/s')
legend = ax.legend(loc='upper left', bbox_to_anchor=(0, -0.65, 1, 0.5), mode = "expand", frameon=False, ncol=3)
ax.set_title(interface.get_device_name())
plt.savefig('./performance.png', bbox_extra_artists=(legend,), bbox_inches='tight', dpi=400)
