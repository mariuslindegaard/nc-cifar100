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

    nc_loss_factors = (0, 0.03, 0.01, 0.003, 0.001)

    args = Args(net=net, gpu=True)
    print("Base args: ")
    print(args)
    print("NC layers: ", nc_layers)
    print("NC loss factors: ", nc_loss_factors)

    for use_cifar10 in (False, True):
        args.cifar10 = use_cifar10
        for prediction_loss_factor in (1E-12, 0.3, 1):
            args.pred_loss = prediction_loss_factor
            for loss_factor in nc_loss_factors:
                if loss_factor == 0 and prediction_loss_factor <= 1E-10:
                    continue

                print("-"*64 + "\nAll loss factors: {}\n".format(nc_loss_factors)
                      + "Starting training with NC loss factor "
                      + "{}, cifar{}, predloss {}\n".format(loss_factor, 10 if args.cifar10 else 100, prediction_loss_factor) + "-"*64)

                args.nc_loss = {layer_name: loss_factor for layer_name in nc_layers}

                train.main(args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true', default=False, help="Whether to do debug or not")
    _args = parser.parse_args()
    main(debug=_args.debug)
