import train
import dataclasses as dc
from typing import Dict
import warnings
import time

import argparse

from run_slurm import Args


def main(debug=True):
    if debug:
        warnings.warn(64*"-" + "\n   RUNNING IN DEBUG MODE\n" + 64*"-")
        time.sleep(1)
        net = 'squeezenet'
        nc_layers = ('fire7', 'fire9')
    else:
        net = 'resnet18'
        nc_layers = ('avg_pool',)

    nc_loss_factors = (0, 0.001, 0.01, 0.1, 1)

    args = Args(net=net, gpu=True)
    print("Base args: ")
    print(args)
    print("NC layers: ", nc_layers)
    print("NC loss factors: ", nc_loss_factors)

    for loss_factor in nc_loss_factors:
        print("-"*48 + "\nAll loss factors: {}\n".format(nc_loss_factors)
              + "Starting training with NC loss factor {}\n".format(loss_factor) + "-"*48)
        args.nc_loss = {layer_name: loss_factor for layer_name in nc_layers}
        if loss_factor > 0:
            args.pred_loss = 0
        else:
            args.pred_loss = 1

        train.main(args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true', default=False, help="Whether to do debug or not")
    _args = parser.parse_args()
    main(debug=_args.debug)
