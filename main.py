
import logging

import numpy as np
import torch

import src.search as search
import src.vanilla as vanilla

from src.hardware.hw import HardwareModel
from src.utils import create_exp_dir
from src.data.datasets import return_dataset
from src.config import load_params
from src.logger import setup_logging
from src.device_manager import get_device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=="__main__":
	# Load arguments
	args = load_params()

	assert args.run_search==1 or args.arch_dir is not None, "Either set args.run_search=1 or provide a arch_dir!"

	# Set seeds for numpy and torch
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Setup logging and wandb
	setup_logging(args)

	# Create output directory
	create_exp_dir(args.save_dir)

	# Set-up device to use for search and training
	device = get_device(args)

	# Hardware model to estimate latency and utilization
	hw_model = HardwareModel(hw_type=args.hw_type, batch_size=args.hw_batch_size, memory_bw=args.memory_bw, array_size=args.array_size, device=device, args=args)

	# Dataloaders for search stages
	train_queue, valid_queue, test_queue, input_dims, no_classes = return_dataset(
		dataset=args.search_dataset, 
		batch_size=args.search_batch_size, 
		num_workers=args.no_dataloader_workers)

	# Init trainer object
	searcher = search.Trainer(device=device, hw_model=hw_model, parser_args=args)

	#Init MS model for microarch search stage
	searcher.init_microsearch(input_dims, no_classes)

	# If arch_dir not provided, perform MS stage. If provided, load arch state and skip MS search
	if args.run_search:
		searcher.microarch_search(
			train_dataloader=train_queue,
			valid_dataloader=valid_queue,
			epochs=args.ms_no_epoch,
		)
	else:
		ms_arch_file = args.arch_dir+"/ms_arch_state.json"
		logging.info(f"Loading MS arch state from {ms_arch_file}, skipping microarch stage.")
		searcher.ms_model.load_arch_state(ms_arch_file)
	
	# Convert MS model to CS model
	searcher.init_channel_search(input_dims, no_classes)

	# If arch_dir not provided, perform CS stage. If provided, load arch state and skip CS search
	if args.run_search:
		searcher.channel_search(
			train_dataloader=train_queue,
			valid_dataloader=valid_queue,
			epochs=args.cs_no_epoch,
		)
	else:
		cs_arch_file = args.arch_dir+"/cs_arch_state.json"
		logging.info(f"Loading CS arch state from {cs_arch_file}, skipping microarch stage.")
		searcher.cs_model.load_arch_state(cs_arch_file)

	# Dataloaders for FT stage
	train_queue, valid_queue, test_queue, input_dims, no_classes  = return_dataset(
		dataset=args.train_dataset, 
		batch_size=args.train_batch_size, 
		num_workers=args.no_dataloader_workers)

	args.steps_per_epoch = len(train_queue)

	# Init trainer object
	trainer = vanilla.trainer.Trainer(device=device, parser_args=args)

	# Convert CS model to FT model
	trainer.init_final_train(searcher.cs_model, input_dims, no_classes)
	logging.info(f"The FT model has {count_parameters(trainer.ft_model)} parameters.")
	searcher.ms_model = None
	searcher.cs_model = None
	searcher = None
	torch.cuda.empty_cache()

	if args.run_train:
		# Train final model
		trainer.train_final(train_queue, valid_queue)

	# Test final model
	trainer.test_final(test_queue)
