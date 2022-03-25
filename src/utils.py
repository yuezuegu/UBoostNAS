

import torch 
import os 
import shutil 
import time 
from datetime import datetime 

from glob import glob

import matplotlib.pyplot as plt
import json 
import wandb
import numpy as np

class AvgMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt

class Stats():
	def __init__(self) -> None:
		self.metrics = {}
		self.epoch = 0
		self.new_epoch(self.epoch)

	def set_epoch(self, epoch):
		self.epoch = epoch
		if epoch not in self.metrics:
			self.new_epoch(epoch)

	def new_epoch(self, epoch):
		self.metrics[epoch] = {}

	def new_metric(self, epoch, metric):
		if epoch not in self.metrics:
			self.new_epoch(epoch)
		self.metrics[epoch][metric] = {"N":0 , "value": 0}

	def update_avg(self, metric, n, val):
		if metric not in self.metrics[self.epoch]:
			self.new_metric(self.epoch, metric)

		if isinstance(val, torch.Tensor):
			val = val.item()

		curr_val = self.metrics[self.epoch][metric]["value"]
		curr_N = self.metrics[self.epoch][metric]["N"]
		sum_ = curr_val*curr_N + val*n
		self.metrics[self.epoch][metric]["N"] = curr_N + n 
		self.metrics[self.epoch][metric]["value"] = sum_ / self.metrics[self.epoch][metric]["N"]

	def set_value(self, metric, val):
		if metric not in self.metrics[self.epoch]:
			self.new_metric(self.epoch, metric)

		if isinstance(val, torch.Tensor):
			val = val.item()

		self.metrics[self.epoch][metric]["value"] = val

	def get_metric(self, metric):
		if metric not in self.metrics[self.epoch]:
			return 0

		return self.metrics[self.epoch][metric]["value"]

	def log(self, prefix, metrics):
		for m in metrics:
			wandb.log({prefix+"/"+m: self.metrics[self.epoch][m]["value"]}) 

	def export_json(self, fname):
		with open(fname, "w") as outfile:  
			json.dump(self.metrics, outfile)

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		#correct_k = correct[:k].view(-1).float().sum(0)
		correct_k = correct[0:k,:].float().sum()
		res.append(correct_k.mul_(100.0/batch_size))
	return res


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))

	scripts_to_save=glob('*.py')+glob('src/**/*.py', recursive=True)

	if scripts_to_save is not None:
		dir_name = os.path.join(path, 'scripts')
		if not os.path.exists(dir_name):
			os.mkdir(dir_name)

		for script in scripts_to_save:
			dst_file = os.path.join(dir_name, os.path.basename(script))
			shutil.copyfile(script, dst_file)

def load(model, model_path):
	state_dict = torch.load(model_path)
	model.load_state_dict(state_dict['model_state'])

def discretize_range(out_range):
	range_step = out_range["step"]
	out_range["start"] = int(np.ceil(out_range["start"] / range_step) * range_step)
	out_range["stop"] = int(np.floor(out_range["stop"] / range_step) * range_step)
	return out_range

def plot_alphas(alphas, k, x_tick_labels=None):
	plt.figure()

	vals = alphas[k].tolist()
	if x_tick_labels is None:
		x_tick_labels = list(range(len(vals)))
	plt.stem(x_tick_labels, vals)

	plt.savefig(f"results/alphas_{k}.png")
	plt.close()




def is_proc_ended(proc):
    retcode = proc.poll()
    if retcode is not None: # Process finished.
        print("Process {} ended with code {}".format(proc.pid, retcode))
        if retcode != 0:
            print("FAILED: Return code is not 0")
        return True
    else:
        return False

def wait_for_proc_limit(running_procs, max_procs):
    while True:
        for proc in running_procs:
            if (is_proc_ended(proc)):
                running_procs.remove(proc)
                
        if len(running_procs) < max_procs: #Block if there is more than x number of running threads
            return running_procs

        time.sleep(.1)

def wait_all_finish(running_procs):
    while True:
        for proc in running_procs:
            if (is_proc_ended(proc)):
                running_procs.remove(proc)
                
        if len(running_procs) == 0:
            return running_procs

        time.sleep(.1)

def date_minute():
    return datetime.now().strftime("%Y_%m_%d-%H_%M")

def date_second():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

def date_millisecond():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")[:-3] #-3 to convert us to ms

