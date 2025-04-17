from abc import abstractclassmethod
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from utils import *


class BasicTask(object):
    def __init__(self, log_dir, logger=None, device='cuda', verbose=True) -> None:
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logger
        if self.logger is None:
            log_file = os.path.join(log_dir, 'log.txt')
            self.logger = construct_logger(log_dir, log_file, verbose)
        self.writer = SummaryWriter(log_dir)

        self.device = device
        self.start_epoch = 1
        self.best_metric = None
        self.best_path = None
        self.verbose = verbose

    def launch(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def log_epoch(self, epoch, epoch_time, **kwargs):
        log_str = epoch_str(epoch, epoch_time)
        for key, value in kwargs.items():
            log_str += f'{key}:{value:.3f} '
            self.writer.add_scalar(key, value, epoch)
        self.logger.info(log_str)


    def save(self, model, optimizer, metric, epoch, best=False):
        if isinstance(model, nn.DataParallel):
            state = {
                'model': model.module.state_dict(),
                'metric': metric,
                'epoch': epoch,
                'optim': optimizer.state_dict()
            }
        else:
            state = {
                'model': model.state_dict(),
                'metric': metric,
                'epoch': epoch,
                'optim': optimizer.state_dict()
            }

        if best:
            if self.best_path is not None:
                os.remove(self.best_path)
            save_path = os.path.join(
                self.log_dir, f'best-{epoch}-{metric:.3f}.pth')
            self.best_path = save_path
        else:
            save_path = os.path.join(
                self.log_dir, f'{epoch}-{metric:.3f}.pth')
            self.best_path = save_path
        self.logger.info(f'<<< Save at {save_path}..')
        torch.save(state, save_path)

    def resume(self, model, optimizer, ckpt_file):
        ckpt_path = ckpt_file
        self.logger.info(f'>>> Resume from {ckpt_path}..')
        ckpt = torch.load(ckpt_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optim'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_metric = ckpt['metric']
        self.best_path = ckpt_path
