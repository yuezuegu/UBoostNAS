
import os
import torch
import torch.backends.cudnn as cudnn


def get_device(args):
	if args.gpu_device is None:
		device = torch.device("cpu")
	else:
		# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
		# args.gpu_device = 0

		if not torch.cuda.is_available():	
			raise SystemError("No GPU device")
			
		device = torch.device("cuda:{}".format(args.gpu_device))
		cudnn.benchmark = True
		cudnn.enabled=True

	return device