from sim_binder import run_csim
from src.hardware.hw import HardwareModel

import numpy as np

import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib import rc

import json
import pandas as pd

import subprocess
import os 

def json_to_dict(d, prefix, val):
    if not isinstance(val, dict):
        d[prefix] = val
    else:
        for key in val:
            json_to_dict(d, prefix+'/'+key, val[key])

class Experiment:
    def __init__(self, path, s) -> None:
        self.path = path

        filenames = ['args.json', 'cs_stats.json', 'ft_stats_'+str(s)+'.json']
        prefix = ['args', 'search', 'train']
        self.json_data = {}
        for i, p in enumerate(prefix):
            f = open(path+"/"+filenames[i])
            json_to_dict(self.json_data, p, json.load(f))

    def get(self, key):
        return self.json_data[key]

def filter_by_value(exps, key, value):
    new_exps = []
    for e in exps:
        if e.get(key) == value:
            new_exps.append(e)
    return new_exps

def values_as_list(exps, key):
    val_list = []
    for e in exps:
        val_list.append(e.get(key))
    return val_list

def find_pareto_indices(points):
    pareto_indices = list(range(len(points)))
    for ind, p1 in enumerate(points):
        for p2 in points:
            if p2[0] <= p1[0] and p2[1] > p1[1]:
                pareto_indices.remove(ind)
                break

    sorted_indices = np.argsort([p[0] for p in points]).tolist()
    sorted_pareto_indices = [p for p in sorted_indices if p in pareto_indices]

    return sorted_pareto_indices

def plot_pareto(points, color=None, label=None):
    pareto_indices = find_pareto_indices(points)
    
    plot_points = [points[pareto_indices[0]]]
    for i in range(1,len(pareto_indices)):
        x = points[pareto_indices[i]][0]
        y = points[pareto_indices[i-1]][1]
        plot_points.append((x,y))
        plot_points.append(points[pareto_indices[i]])

    plt.plot([p[0] for p in plot_points], [p[1] for p in plot_points], linestyle='--', color=color)




def parse_results(exp_dirs):
    experiments = []
    for exp_dir in exp_dirs:
        subdirs = os.listdir(exp_dir)

        for sub in subdirs:
            path = exp_dir+"/"+sub

            for no_stack in [3]:
                experiments.append(Experiment(path, no_stack))

    return experiments

def hypervolume_area(ref, points):
    points_sorted = sorted(points, key=lambda tup: tup[0])

    rx, ry = ref
    x = rx
    area = 0
    for px, py in points_sorted:
        area += abs(px - x) * abs(py - ry)
        x = px
    return area

exp_dirs = []

# exp_dirs += ["experiments/run-2022_02_10-17_40_45"]
# exp_dirs += ["experiments/run-2022_02_10-22_33_40"]
# #exp_dirs += ["experiments/run-2022_02_11-10_00_05"]
# #exp_dirs += ["experiments/run-2022_02_11-16_33_59"]

# #exp_dirs += ["experiments/run-2022_02_11-23_55_16"]
# exp_dirs += ["experiments/run-2022_02_12-12_29_45"]
# exp_dirs += ["experiments/run-2022_02_13-21_08_26"]
# exp_dirs += ["experiments/run-2022_02_14-13_48_55"]
exp_dirs += ["experiments/run-2022_02_20-16_44_21"]
exp_dirs += ["experiments/run-2022_02_20-20_21_26"]
exp_dirs += ["experiments/run-2022_02_20-23_42_51"]
exp_dirs += ["experiments/run-2022_02_21-08_57_28"]


exps = parse_results(exp_dirs)

colors = {'systolic':"tab:red", 'flops':"tab:blue", 'roofline':"tab:grey", 'blackbox':"tab:green"}
markers = {'systolic':'s', 'flops':'D', 'roofline':'p', 'blackbox':'X'}
labels = {'systolic':'U-Boost', 'flops':'FLOPS', 'roofline':'Roofline', 'blackbox':'Lookup Table'}

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':14})
rc('font',**{'size':21})
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(12, 6))

hv_ref = (0,100)

df = pd.DataFrame()

data = {}
for method in ['systolic','flops','roofline','blackbox']:
    print("Method: {} \t".format(method))
    data[method] = {}
    subexps = filter_by_value(exps, 'args/hw_type', method)

    no_samples = len(subexps)
    runtime = values_as_list(subexps, 'train/test/runtime_csim/value')
    util = values_as_list(subexps, 'train/test/util_csim/value')
    acc = values_as_list(subexps, 'train/test/valid_top1/value')

    pareto_indices = find_pareto_indices([(runtime[i], acc[i]) for i in range(no_samples)])
    plot_pareto([(runtime[i], acc[i]) for i in range(no_samples)], color=colors[method])

    hp_val = hypervolume_area(hv_ref, [(runtime[i], acc[i]) for i in pareto_indices])

    print("Hypervolume: {:.3}".format(hp_val))
    plt.scatter(0, 0, s=50, color=colors[method], marker=markers[method], label=labels[method])

    for lat_coef in [0.1, 0.5, 1.0, 5.0]:
        data[method][lat_coef] = {}
        subsubexps = filter_by_value(subexps, 'args/lat_coef', lat_coef)
        
        #print(values_as_list(subsubexps, 'args/save_dir'))

        no_samples = len(subsubexps)
        runtime = values_as_list(subsubexps, 'train/test/runtime_csim/value')
        util = values_as_list(subsubexps, 'train/test/util_csim/value')
        acc = values_as_list(subsubexps, 'train/test/valid_top1/value')

        for i in range(no_samples):
            plt.scatter(runtime[i], acc[i], s=50, color=colors[method], marker=markers[method])
        

        acc_avg = np.mean(acc)
        acc_var = np.var(acc)
        runtime_avg = np.mean(runtime)
        runtime_var = np.var(runtime)
        util_avg = np.mean(util)
        util_var = np.var(util)
        print("Lat_coef: {} \t no_samples: {} \t accuracy mean: {:.3} var: {:.3} \t runtime mean: {:.3}  var: {:.3} \t util mean: {:.3} var: {:.3} ".format(
            lat_coef, 
            no_samples, 
            acc_avg,
            acc_var,
            runtime_avg,
            runtime_var,
            util_avg,
            util_var))

        data[method][lat_coef] = {"hv": "{:.3}".format(hp_val), "acc_mean": "{:.3}".format(acc_avg), "acc_var": "{:.3}".format(acc_var), "runtime mean": "{:.3}".format(runtime_avg), "runtime var": "{:.3}".format(runtime_var)}

print(pd.DataFrame(data).to_csv("table_cifar.csv"))



plt.xlim(left=0, right=0.25)
plt.ylim(bottom=82)
plt.ylabel(r'Accuracy (\%)')
plt.xlabel(r'Runtime (ms)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/cifar10.png")
plt.savefig("results/cifar10.pdf")


plt.figure()

