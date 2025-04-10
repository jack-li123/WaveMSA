# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import logging
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tlt.data import create_token_label_target
from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, args=None,writer=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _logger = logging.getLogger('train')
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if not args.prefetcher:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
        else:
            if args.token_label and args.token_label_data and not data_loader.mixup_enabled:
                targets = create_token_label_target(
                    targets,
                    num_classes=args.nb_classes,
                    smoothing=args.smoothing,
                    label_size=args.token_label_size)

        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)

            if args.token_label:
                loss = criterion(outputs, targets)
            else:
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            _logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    _logger.info("Averaged stats:" + str(metric_logger))
    writer.add_scalar('Train/lr',metric_logger.lr.global_avg, epoch)
    writer.add_scalar('Train/Loss',metric_logger.loss.global_avg, epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device,epoch,writer):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    _logger = logging.getLogger('train')
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    _logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    writer.add_scalar('Test/acc1',metric_logger.acc1.global_avg, epoch)
    writer.add_scalar('Test/acc5', metric_logger.acc5.global_avg, epoch)
    writer.add_scalar('Test/Loss', metric_logger.loss.global_avg, epoch)

    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
