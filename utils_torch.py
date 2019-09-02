#!/usr/bin/env python
"""
    General useful utility functions for pytorch
    Date created: 30/8/19
    Python Version: 3.6
"""

import GPUtils.startup_guyga as gputils
import os, shutil, pickle, sys, random
from tqdm import tqdm
from time import localtime, strftime, time
import numpy as np
import torch
from torch import nn
from torchvision import transforms

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

def param_count_str(parameters):
    N = sum(p.numel() for p in parameters)
    if N > 1e6:
        return '%.2fM' % (N / 1000000.0)
    elif N > 1e3:
        return '%.1fK' % (N / 1000.0)
    else:
        return '%d' % N

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, relative=False):
        self.relative = relative
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.relative:
            if not self.count:
                self.scale = 100 / abs(val)
            val *= self.scale
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_classifier(model, trainloader, testloader, optimizer, n_epochs=10, start_epoch=0, use_cuda=False, checkpoint_filepath=None, sum_writer=None, verbose=2, loss_select=True):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0  # best test accuracy
    best_loss = np.inf
    global global_step
    global_step = 0
    if verbose == 1:
        pbar = tqdm(desc='Epochs', total=n_epochs)
        pbar.update(start_epoch)
    for epoch in range(start_epoch, n_epochs):
        if verbose > 1:
            print('\nEpoch: [%d | %d]' % (epoch + 1, n_epochs))
        train_loss, train_acc = train_test(trainloader, model, criterion, use_cuda, optimizer, sum_writer=sum_writer, verbose=verbose - 1)
        if testloader is not None:
            test_loss, test_acc = train_test(testloader, model, criterion, use_cuda, verbose=verbose - 1)
            if sum_writer:
                sum_writer.add_scalar('Test/Loss', test_loss, epoch)
                sum_writer.add_scalar('Test/Top1', test_acc, epoch)
            is_best = test_loss < best_loss if loss_select else test_acc > best_acc
            if is_best:
                best_acc = test_acc
                best_loss = test_loss
            # best_acc = max(test_acc, best_acc)
        else:
            is_best = train_loss < best_loss if loss_select else train_acc > best_acc
            if is_best:
                best_acc = train_acc
                best_loss = train_loss
        # # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        if verbose == 1:
            pbar.update()
        if checkpoint_filepath is not None:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath=checkpoint_filepath)
    if verbose == 1:
        pbar.close()
    return best_acc

def train_test(loader, model, criterion, use_cuda, optimizer=None, sum_writer=None, verbose=1):
    global global_step
    if optimizer:
        # switch to train mode
        mode = 'Train'
        model.train()
    else:
        mode = 'Test '
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    if verbose > 0:
        bar = tqdm(desc=mode, total=len(loader))
    for batch_idx, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  #targets.cuda(async=True)
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if sum_writer:
                sum_writer.add_scalar('Train/Loss', losses.val, global_step)
                sum_writer.add_scalar('Train/Top1', top1.val, global_step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose > 0:  # plot progress
            bar.set_postfix_str('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        ))
            bar.update()
    if verbose > 0:
        bar.close()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best=False, checkpoint='checkpoint', filename='checkpoint.pth.tar', filepath=None, savebestonly=False):
    if filepath is None:
        filepath = os.path.join(checkpoint, filename)
    else:
        filename = os.path.basename(filepath).split('.')[0]
        checkpoint = os.path.dirname(filepath)
    if is_best or not savebestonly:
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, '%s_best.pth.tar' % filename))

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor)

def tensor_transform(tensor, xfm):
    return torch.stack([xfm(x) for x in tensor])
