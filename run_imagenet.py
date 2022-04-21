import os
import subprocess
import time

from src.utils import (
    date_second,
    date_millisecond,
    wait_for_proc_limit,
    wait_all_finish,
)


MAX_NO_PROC = 1

SETTINGS = {
    "UTIL_AWARE_SYSTOLIC": {"UTIL_COEF": 1.0, "LAT_COEF": [1.0], "HW_TYPE": "systolic"},
}

STACKS = [
    [1, 0, 1, 0, 1, 0],
    # [1, 0, 0, 1, 0, 0, 1, 0, 0]
]

if __name__ == "__main__":
    exp_dir = "experiments/run-{}".format(date_second())
    os.system("mkdir -p {}".format(exp_dir))

    running_procs = []
    save_dirs = []

    gpu_devices = [0, 1, 2, 3]

    cnt = 0
    for baseline in SETTINGS:
        UTIL_COEF = SETTINGS[baseline]["UTIL_COEF"]

        HW_TYPE = SETTINGS[baseline]["HW_TYPE"]

        LAT_COEF = 1.0
        for cs_stacks in STACKS:
            cs_stacks_str = " ".join([str(i) for i in cs_stacks])
            for SEED in range(1):
                SAVE_DIR = exp_dir + "/" + date_millisecond()
                os.mkdir(SAVE_DIR)

                cmd = f"python main.py \
                    --train_batch_size 256 \
                    --train_sgd_init_lr 0.06 \
                    --train_dataset imagenet\
                    --run_search 0 \
                    --run_train 1 \
                    --cs_stacks {cs_stacks_str} \
                    --ms_no_epoch {0} \
                    --cs_no_epoch {0} \
                    --ft_no_epoch {100} \
                    --arch_dir saved-models/stacks6 \
                    --util_coef {UTIL_COEF} \
                    --lat_coef {LAT_COEF} \
                    --gpu_device {gpu_devices[cnt%len(gpu_devices)]} \
                    --hw_type {HW_TYPE} \
                    --seed {SEED} \
                    --wandb_enable 1 \
                    --save_dir {SAVE_DIR}"

                print(cmd)
                p = subprocess.Popen(cmd, shell=True)
                running_procs.append(p)

                time.sleep(2)

                wait_for_proc_limit(running_procs, MAX_NO_PROC)

                cnt += 1

    print(save_dirs, flush=True)
    wait_all_finish(running_procs)
