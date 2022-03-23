import subprocess
import os 
import time
from datetime import datetime

NO_PROCS = 1

def is_proc_ended(proc):
    retcode = proc.poll()
    if retcode is not None: # Process finished.
        print("Process {} ended with code {}".format(proc.pid, retcode))
        if retcode != 0:
            print("FAILED: Return code is not 0")
        return True
    else:
        return False

def wait_for_proc_limit(running_procs):
    while True:
        for proc in running_procs:
            if (is_proc_ended(proc)):
                running_procs.remove(proc)
                
        if len(running_procs) < NO_PROCS: #Block if there is more than x number of running threads
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

DATASET = "cifar10"

MS_NO_EPOCH = 20
CS_NO_EPOCH = 40
FT_NO_EPOCH = 200

SEARCH_SGD_INIT_LR = 0.001
TRAIN_SGD_INIT_LR = 0.025
ADAM_INIT_LR = 0.1
ADAM_WEIGHT_DECAY = 0

BATCH_SIZE = 64

START_ARCH_TRAIN = 0

NAS_MODE = 1

SETTINGS = {
    "UTIL_AWARE_SYSTOLIC": {"UTIL_COEF":1.0, "LAT_COEF":[1.0], "HW_TYPE":"systolic"},
    "FLOPS": {"UTIL_COEF":0.0, "LAT_COEF":[1.0], "HW_TYPE":"flops"},
    "ROOFLINE": {"UTIL_COEF":0.0, "LAT_COEF":[1.0], "HW_TYPE":"roofline"},
    "BLACKBOX": {"UTIL_COEF":0.0, "LAT_COEF":[1.0], "HW_TYPE":"blackbox"},
}

if __name__=="__main__":
    exp_dir = "experiments/run-{}".format(date_second())
    os.system("mkdir -p {}".format(exp_dir))

    running_procs = []
    save_dirs = []

    gpu_devices = [0,1,2,3]

    cnt = 0
    for baseline in SETTINGS:
        UTIL_COEF = SETTINGS[baseline]["UTIL_COEF"]

        HW_TYPE = SETTINGS[baseline]["HW_TYPE"]
        
        for LAT_COEF in SETTINGS[baseline]["LAT_COEF"]:
            for SEED in range(1):
                SAVE_DIR = exp_dir + "/" + date_millisecond()
                os.mkdir(SAVE_DIR)

                cmd = f"python main.py \
                    --run_search 1 \
                    --run_train 1 \
                    --dataset {DATASET} \
                    --util_coef {UTIL_COEF} \
                    --lat_coef {LAT_COEF} \
                    --search_batch_size {BATCH_SIZE} \
                    --ms_no_epoch {MS_NO_EPOCH} \
                    --cs_no_epoch {CS_NO_EPOCH} \
                    --ft_no_epoch {FT_NO_EPOCH} \
                    --train_batch_size {BATCH_SIZE} \
                    --search_sgd_init_lr {SEARCH_SGD_INIT_LR} \
                    --train_sgd_init_lr {TRAIN_SGD_INIT_LR} \
                    --gpu_device {gpu_devices[cnt%len(gpu_devices)]} \
                    --adam_init_lr {ADAM_INIT_LR} \
                    --adam_weight_decay {ADAM_WEIGHT_DECAY} \
                    --hw_type {HW_TYPE} \
                    --seed {SEED} \
                    --wandb_enable 1 \
                    --save_dir {SAVE_DIR}"

                print(cmd)
                p = subprocess.Popen(cmd, shell=True)
                running_procs.append(p)

                time.sleep(2)
                
                wait_for_proc_limit(running_procs)

                cnt += 1

    print(save_dirs, flush=True)
    wait_all_finish(running_procs)