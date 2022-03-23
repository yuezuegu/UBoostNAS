
import coloredlogs
import wandb 
import logging
import json 


def setup_logging(args):
	level_styles = coloredlogs.DEFAULT_LEVEL_STYLES
	level_styles['info'] = {'color': 'yellow'}

	coloredlogs.install(
		level=logging.INFO,
		fmt='%(asctime)s %(levelname)s %(message)s',
		level_styles = level_styles
	)
    
	logging.info(args)

	wandb.init(
		project='eccv-imagenet', 
		entity='dmaskingnas', 
		config=args,
		mode='online' if args.wandb_enable else 'disabled'
	)




	f = open(args.save_dir + "/args.json", "w")
	json.dump(args.__dict__, f)
	f.close()
