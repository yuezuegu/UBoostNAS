
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


data = {'systolic': {}, 'flops': {}, 'roofline': {}, 'blackbox': {}}

no_stacks = [6,7,8,9]
data['systolic'][6] = {'valid_top1': 85.7, 'runtime_csim': 0.6456 }
data['systolic'][7] = {'valid_top1': 86.3, 'runtime_csim': 0.8273 }
data['systolic'][8] = {'valid_top1': 87.0, 'runtime_csim': 0.8572 }
data['systolic'][9] = {'valid_top1': 87.04, 'runtime_csim': 0.9773 }

colors = {'systolic':"tab:red", 'flops':"tab:blue", 'roofline':"tab:grey", 'blackbox':"tab:green"}
markers = {'systolic':'s', 'flops':'D', 'roofline':'p', 'blackbox':'X'}
labels = {'systolic':'U-Boost', 'flops':'FLOPS', 'roofline':'Roofline', 'blackbox':'Lookup Table'}

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':14})
rc('font',**{'size':21})
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(12, 6))

hv_ref = (0,100)

df = pd.DataFrame()

for method in ['systolic']:
    print("Method: {} \t".format(method))

    acc = [data[method][lat_coef]['valid_top1'] for lat_coef in data[method]]
    runtime = [data[method][lat_coef]['runtime_csim'] for lat_coef in data[method]]

    pareto_indices = find_pareto_indices([(runtime[i], acc[i]) for i in range(len(runtime))])
    plot_pareto([(runtime[i], acc[i]) for i in range(len(runtime))], color=colors[method])

    hp_val = hypervolume_area(hv_ref, [(runtime[i], acc[i]) for i in pareto_indices])

    print("Hypervolume: {:.3}".format(hp_val))
    plt.scatter(0, 0, s=50, color=colors[method], marker=markers[method], label=labels[method])

    plt.scatter(runtime, acc, s=50, color=colors[method], marker=markers[method])

    # for lat_coef in no_stacks:
    #     data[method][lat_coef]['hv'] = hp_val

    #     no_samples = 1
    #     runtime = data[method][lat_coef]['runtime_csim']
    #     acc = data[method][lat_coef]['valid_top1']

    #     plt.scatter(runtime, acc, s=50, color=colors[method], marker=markers[method])
        
    #     acc_avg = np.mean(acc)
    #     runtime_avg = np.mean(runtime)
    #     print("Lat_coef: {} \t accuracy: {:.3} \t runtime: {:.3}".format(
    #         lat_coef,
    #         acc_avg,
    #         runtime_avg))

print("latex table code:")
print(pd.DataFrame(data).to_latex())
print(pd.DataFrame(data).to_csv("table_imagenet100.csv"))


#plt.xlim(left=0, right=0.25)
#plt.ylim(bottom=82)
plt.ylabel(r'Accuracy (\%)')
plt.xlabel(r'Runtime (ms)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/imagenet.png")
plt.savefig("results/imagenet.pdf")


plt.figure()

