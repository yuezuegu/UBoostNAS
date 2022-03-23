
import os
import argparse



def load_params():
	parser = argparse.ArgumentParser()
	parser.add_argument('--search_dataset', type=str, required=False, default="cifar10",
		choices=['cifar10', 'imagenet100'])
	parser.add_argument('--train_dataset', type=str, required=False, default="imagenet",
		choices=['cifar10', 'imagenet100', "imagenet"])

	parser.add_argument('--ms_no_epoch', type=int, required=False, default=1)
	parser.add_argument('--cs_no_epoch', type=int, required=False, default=1)
	parser.add_argument('--ft_no_epoch', type=int, required=False, default=1)

	parser.add_argument('--hw_type', type=str, required=False, 
		choices=["systolic", "flops", "roofline", "blackbox"], default="systolic", 
		help='Hardware model to estimate inference latency and utilization')
	parser.add_argument('--array_size', type=int, nargs='+', required=False, default=[128,128], help='Array size')
	parser.add_argument('--hw_batch_size', type=int, required=False, default=64, help='Batch size for hw simulations')
	parser.add_argument('--memory_bw', type=int, required=False, default=80, help='Memory bandwidth in GB/s')
	parser.add_argument('--clk_freq', type=int, required=False, default=1e9, help='Clock freq for hardware model')

	parser.add_argument('--lat_coef', type=float, required=False, default=1.0,
		help='Latency coefficient in loss function')
	parser.add_argument('--util_coef', type=float, required=False, default=1.0,
		help='Utilization coefficient in loss function')
	
	parser.add_argument('--ms_stacks', type=int, nargs='+', required=False, default=[1,1,1], 
		help='No of stacks for microarch search, 1: stack with maxpool, 0: stack w/out maxpool')
	parser.add_argument('--cs_stacks', type=int, nargs='+', required=False, default=[1,0,0,1,0,0,1,0,0], 
		help='No of stacks for channel search, 1: stack with maxpool, 0: stack w/out maxpool')

	parser.add_argument('--search_batch_size', type=int, required=False, default=64)
	parser.add_argument('--search_sgd_init_lr', type=float, required=False, default=0.001)
	parser.add_argument('--search_sgd_momentum', type=float, required=False, default=0.8)
	parser.add_argument('--search_sgd_weight_decay', type=float, required=False, default=3e-4)
	parser.add_argument('--search_weight_grad_clip', type=float, required=False, default=0.5)
	parser.add_argument('--search_lr_scheduler_step', type=float, required=False, default=30)
	parser.add_argument('--search_lr_scheduler_gamma', type=float, required=False, default=0.5)

	parser.add_argument('--adam_init_lr', type=float, required=False, default=0.1)
	parser.add_argument('--adam_weight_decay', type=float, required=False, default=0)

	parser.add_argument('--train_batch_size', type=int, required=False, default=256)
	parser.add_argument('--train_sgd_init_lr', type=float, required=False, default=0.1)
	parser.add_argument('--train_sgd_momentum', type=float, required=False, default=0.9)
	parser.add_argument('--train_sgd_weight_decay', type=float, required=False, default=5e-4)
	parser.add_argument('--train_weight_grad_clip', type=float, required=False, default=0.5)
	parser.add_argument('--train_lr_scheduler_step', type=float, required=False, default=30)
	parser.add_argument('--train_lr_scheduler_gamma', type=float, required=False, default=0.1)

	parser.add_argument('--weight_vs_arch', type=float, required=False, default=0.8, 
		help='Ratio of epochs of weight training to architecture training')

	parser.add_argument('--init_tau', type=float, required=False, default=1.0)
	parser.add_argument('--tau_anneal_rate', type=float, required=False, default=0.95)
	parser.add_argument('--min_tau', type=float, required=False, default=0.001)

	parser.add_argument('--report_freq', type=int, required=False, default=50)
	parser.add_argument('--wandb_enable', type=int, required=False, default=0)

	parser.add_argument('--seed', type=int, required=False, default=0)
	parser.add_argument('--gpu_device', type=int, required=False, default=0)

	parser.add_argument('--run_search', type=int, required=False, default=1, 
		help="Set to 1 to perform architecture search, set to 0 to skip")
	parser.add_argument('--run_train', type=int, required=False, default=1,
		help="Set to 1 to train selected architecture, set to 0 to skip")

	parser.add_argument('--save_dir', type=str, required=False, default="experiments/tmp",
		help='Save directory for results')
	parser.add_argument('--arch_dir', type=str, required=False, default=None,
		help='Directory to import pre-searched architecture. If set None, search stages will be performed')
	
	args = parser.parse_args()

	args.no_dataloader_workers = os.cpu_count()

	args.channel_range = {"start": 64, "stop": 280, "step": 8}

	return args