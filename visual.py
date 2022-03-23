
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc

from src.hardware.hw import HardwareModel

def plot_design_space():
    hw_model = HardwareModel("systolic", array_size=[128,128], batch_size=64, memory_bw=80, device=None)

    in_ch_range = np.arange(2,280,2)
    out_ch_range = np.arange(2,280,2)

    data = np.zeros(shape=[in_ch_range.shape[0], out_ch_range.shape[0]])

    for i, in_ch in enumerate(in_ch_range):
        for o, out_ch in enumerate(out_ch_range):
            y_shape = [32,32,in_ch,64]
            kernel_shape = [3,3]

            runtime, util = hw_model.conv_runtime(in_ch, out_ch, y_shape, kernel_shape)
            data[i,o] = util

    plt.pcolormesh(in_ch_range, out_ch_range, data, cmap=cm.get_cmap('coolwarm'), shading='auto')
    plt.xlabel("Number of output channels")
    plt.ylabel("Number of input channels")

def read_channels(filename):
    out_ch_num = {i: 0 for i in range(64,300,8)}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if 'Conv2d' in l:
                in_ch, out_ch = l.replace('\t',' ').replace('\n','').split(" ")[-1].split('->')
                out_ch = int(out_ch)
                out_ch_num[out_ch] += 1
                #print("{}".format(out_ch))

    total = sum(out_ch_num.values())
    return {k:out_ch_num[k] for k in out_ch_num}

def get_util():
    d2 = 128
    rng = list(range(4,512,4))

    util = {}
    for out_ch in rng:
        no_tiles_d2 = np.ceil(d2 / 128)
        no_tiles_d3 = np.ceil(out_ch / 128)

        u = 100*(d2 * out_ch) / ( no_tiles_d2 * no_tiles_d3 * 128 * 128 )

        util[out_ch] = u

    return util

util = get_util()

systolic = read_channels("experiments/run-2022_03_03-08_41_17/2022_03_03-08_41_17_872/ft_model.txt")
flops = read_channels("experiments/run-2022_03_03-08_41_17/2022_03_03-08_41_19_876/ft_model.txt")
#roofline = read_channels("experiments/run-2022_03_03-08_41_17/2022_03_03-08_41_21_880/ft_model.txt")
blackbox = read_channels("experiments/run-2022_03_03-08_41_17/2022_03_03-08_41_23_884/ft_model.txt")

# systolic = read_channels("experiments/tmp/optimized.txt")
# flops = read_channels("experiments/tmp/unoptimized.txt")

rc('font',**{'size':21})
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(12, 6))

#ax.hist([p[1] for p in roofline],bins=list(range(4,300,8)), alpha=0.5, color='tab:gray')
#ax.hist([p[1] for p in flops],bins=list(range(4,300,8)), alpha=0.5, color='steelblue')
#ax.hist([p[1] for p in blackbox],bins=list(range(4,300,8)), alpha=0.5, color='steelblue')
#ax.hist([p[1] for p in systolic],bins=list(range(4,300,8)), alpha=0.5, color='tab:red')

lns1 = ax.bar(x=list(systolic.keys()), height=list(systolic.values()), width=8, alpha=0.5, color='tab:red', label='U-Boost')
lns2 = ax.bar(x=list(blackbox.keys()), height=list(blackbox.values()), width=8, alpha=0.5, color='tab:gray', label='Blackbox')
lns3 = ax.bar(x=list(flops.keys()), height=list(flops.values()), width=8, alpha=0.5, color='steelblue', label='FLOPS')

ax.set_xlim([0,300])
#ax.set_ylim([0,1.1])
ax.set_ylabel(r"Histogram")
ax.set_xlabel(r"Number of output channels")

ax2=ax.twinx()
lns4 = ax2.plot(util.keys(), util.values(), linestyle='--', color='gray', label='Utilization')
ax2.set_xticks(list(range(32,257,32)))
ax2.tick_params(axis='y', colors='gray')
ax2.set_ylabel(r"Utilization (\%)", color='gray')

ax.grid(True, axis='x', alpha=0.2)
#plt.axvline(x=64, linestyle='--')
#plt.axvline(x=280, linestyle='--')


lns = [lns1]+[lns2]+[lns3]+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

plt.tight_layout()
plt.savefig('results/visual.png', bbox_inches='tight')
plt.savefig('results/visual.pdf', bbox_inches='tight')