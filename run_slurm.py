# from main import Args
# import train
import os
import dataclasses as dc

import itertools

from typing import List, Dict, Iterable

PROJECT_ROOT = "/om2/user/lindegrd/nc_cifar100/"
CONFIGS_PATH_BASE = os.path.join(PROJECT_ROOT, "slurm/conf/")
JOB_SCRIPT_STUMP = \
"""#!/bin/bash

source ~/.conda_init
conda activate nc

/bin/false
while [ $? -ne 0 ]; do
    echo "~~~ RUNNING EXPERIMENT! ~~~"
    python3 {} {}
done
"""


@dc.dataclass
class Args:
    net: str  # What nnet to use
    gpu: bool = False  # dc.field(default_factory=lambda: False)  # Whether to use GPU
    b: int = 2048  # Batch size
    warm: int = 1  # How many epochs to warm up for
    lr: float = 0.1  # Learning rate
    resume: bool = False  # Whether to resume training
    verbose: bool = False  # Print verbose debug
    nc_loss: Dict[str, float] = dc.field(default_factory=dict)  # Layers to do nc-loss on. Layername maps to weighting
    pred_loss: float = 1  # Weighting of prediction loss
    cifar10: bool = False  # Whether to use cifar10 instead of cifar100
    epochs: int = 200  # Number of train epochs
    milestones: Iterable[int] = (60, 120, 160)


def get_command(args, python_path='python3', exec_path=os.path.join(PROJECT_ROOT, 'main.py')):
    return "{} {} {}".format(python_path, exec_path, get_args_str(args))


def get_args_str(args: Args):
    ret = ""
    for attr in dir(args):
        if attr.startswith("_"):
            continue

        value = getattr(args, attr)

        if attr == 'nc_loss':
            for layer, loss in value.items():
                ret += ' -nc_loss {} {}'.format(layer, loss)
        elif attr == 'milestones':
            ret += ' -{}'.format(attr)
            for milestone in value:
                ret += ' ' + str(milestone)
        else:
            if type(value) is bool:
                if value:
                    ret += " -{}".format(attr)
                else:
                    pass
            else:
                ret += " -{} {}".format(attr, value)

    ret = ret.strip(' ')

    return ret


def write_to_configs(args_array: List[Args]) -> str:
    """Write all arg arrays to configs and return the path (missing task_id.sh)"""
    configs_path_base = os.path.join(CONFIGS_PATH_BASE, 'job_')

    print('Writing {} jobs to'.format(len(args_array)), configs_path_base+'{job_idx}.sh')

    for job_idx, args in enumerate(args_array):
        script_str = JOB_SCRIPT_STUMP.format(
            os.path.join(PROJECT_ROOT, 'train.py'),
            get_args_str(args)
        )
        # print("Writing to", filepath)
        filepath = configs_path_base + str(job_idx) + ".sh"
        with open(filepath, 'w') as f:
            f.write(script_str)

    return configs_path_base


def submit_sbatch(configs_path_base, num_tasks):

    command = "sbatch --array=0-{} ".format(num_tasks-1) \
        + "--export=configs_path_base='{}'".format(configs_path_base) \
        + " " + os.path.join(PROJECT_ROOT, "slurm/execute_array.sh")
    print("Submitting batch at {}".format(configs_path_base) + "{idx}.sh")
    print(command)
    out = os.popen(command)

    print(out.read())


def get_all_args() -> List[Args]:
    """Get all the arguments to run experiments over"""
    net = 'resnet18'

    # Specify all listst. Will do cartesian product over all lists
    nc_layers = ('avg_pool',)
    nc_loss_factor_arr = (0, 0.003, 0.01, 0.03, 0.1)
    pred_loss_arr = (0, 0.1, 1)
    cifar10_arr = (False, True)
    epochs_milestones_arr = ((200, (60, 120, 160)), (300, (120, 200, 260)))  # Each entry is (total_epochs, milestones)

    # Do cartesian product
    iter_product = itertools.product(
        nc_loss_factor_arr,
        pred_loss_arr,
        cifar10_arr,
        epochs_milestones_arr
    )

    # Add all argument possibilities to a list of Args, return it
    argument_list = []
    for nc_loss_factor, pred_loss, cifar10, (epochs, milestones) in iter_product:
        if nc_loss_factor + pred_loss <= 1E-10:  # If there is no effective loss, discard
            continue

        # Initialize base args
        args = Args(net=net, gpu=True)

        # Set arguments
        args.nc_loss = {layer_name: nc_loss_factor for layer_name in nc_layers}
        args.pred_loss = pred_loss
        args.cifar10 = cifar10
        args.epochs = epochs
        args.milestones = milestones

        argument_list.append(args)

    return argument_list


def main():
    args_array = get_all_args()
    configs_path_base = write_to_configs(args_array)
    submit_sbatch(configs_path_base, len(args_array))


if __name__ == "__main__":
    main()
