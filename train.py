# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from ForwardHookedModel import ForwardHookedModel
from Criterions import MultipleCriterions, Criterions


def train(args, net, training_loader, loss_function, optimizer, epoch, writer, warmup_scheduler=None):

    net.train()
    pbar_batch = tqdm.tqdm(training_loader, position=1, leave=False, ncols=75)
    for batch_index, (images, labels) in enumerate(pbar_batch):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        pbar_batch.set_description('Loss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


@torch.no_grad()
def eval_training(args, net, test_loader, training_loader, loss_function, epoch=0, tb_writer=None, pbar=None):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        if type(outputs) is tuple:
            assert len(outputs) == 2 and isinstance(outputs[1], dict)
            _, preds = outputs[0].max(1)
        else:
            _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.verbose:
        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(test_loader.dataset),
            correct.float() / len(test_loader.dataset),
            finish - start
        ))
        print()

    if pbar:
        pbar.set_description('Test: Avg. loss: {:.4G}, Acc.: {:.4f}'.format(
            test_loss / len(test_loader.dataset),
            correct.float() / len(test_loader.dataset),
        ))

    #add informations to tensorboard
    if tb_writer:
        tb_writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        tb_writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


def main(args):

    net = ForwardHookedModel(get_network(args), args.nc_loss)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    pred_loss_func = nn.CrossEntropyLoss()
    loss_function: MultipleCriterions = Criterions.get_CDNV_criterion(args.nc_loss, prediction_loss=pred_loss_func, prediction_weighting=1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    subfolder = os.path.join(args.net,
        "_".join(['nc_{}_{}'.format(layer_name, weight) for layer_name, weight in args.nc_loss.items()])
        if args.nc_loss else 'base'
    )

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, subfolder), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, subfolder, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, subfolder, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, subfolder, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net.base_model, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, subfolder, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, subfolder, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(args, net, cifar100_test_loader, cifar100_training_loader,
                                     loss_function, epoch=0, tb_writer=None)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, subfolder, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, subfolder, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, subfolder, recent_folder))

    pbar_epoch = tqdm.tqdm(range(1, settings.EPOCH + 1), position=0, leave=True, ncols=75)
    for epoch in pbar_epoch:
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(args, net, cifar100_training_loader, loss_function, optimizer, epoch, writer, warmup_scheduler)

        acc = eval_training(args, net, training_loader=cifar100_training_loader, test_loader=cifar100_test_loader,
                            loss_function=loss_function, epoch=epoch, tb_writer=writer, pbar=pbar_epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            if args.verbose:
                print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            if args.verbose:
                print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    # parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-b', type=int, default=2048, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-verbose', action='store_true', default=False, help='Print verbose debug')
    parser.add_argument('-nc_loss', action='append', nargs=2, default=[], help='Layers to do nc-loss on. Takes "layername loss_factor"')
    _args = parser.parse_args()
    _args.nc_loss = {layername: float(loss_weight) for layername, loss_weight in _args.nc_loss}
    main(_args)
