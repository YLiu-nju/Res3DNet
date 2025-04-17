from itertools import product
from types import SimpleNamespace
from utils import *
from torch.multiprocessing import Process, Queue, Lock

import torch
import os
import copy
import pickle


class BasicScheduler(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args_pool = list()
        self.search_params = list()
        if args.gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        copy_args = copy.deepcopy(args)
        copy_args = vars(copy_args)
        for key, value in copy_args.items():
            if not isinstance(value, list):
                copy_args[key] = [value]
            else:
                self.search_params.append(key)
        combs = product(*copy_args.values())
        for comb in combs:
            comb_arg = SimpleNamespace(**dict(zip(copy_args.keys(), comb)))
            self.args_pool.append(comb_arg)

        self.device = 'cuda'
        os.makedirs(args.log_dir, exist_ok=True)
        if len(self.search_params) > 0:
            log_file = os.path.join(args.log_dir, 'test.txt')
            self.logger = construct_logger(args.log_dir, log_file, True)
        else:
            self.logger = construct_logger(args.log_dir, None, True)

    def emit_tasks(self, Task, verbose=True, descending_metric=True):
        best_metric = 100000 if descending_metric else 0
        best_task_arg, best_path = SimpleNamespace(), ''
        for args in self.args_pool:
            args = vars(args)
            if len(self.search_params) == 0:
                sub_log_dir = ''
            else:
                tokens = [f'{k}{str(args[k]):.6}' for k in self.search_params]
                sub_log_dir = '_'.join(tokens)
            args['log_dir'] = os.path.join(args['log_dir'], sub_log_dir)
            args = SimpleNamespace(**args)
            self.logger.info('>>> Start task..')
            self.logger.info(args)

            try:
                task = Task(args.log_dir, device=self.device, verbose=verbose)
                metric, path = task.launch(args)
                self.logger.info(
                    f'<<< Task metric: {metric}, path: {path}..\n')
                if (descending_metric and metric < best_metric) or \
                        (not descending_metric and metric > best_metric):
                    best_metric = metric
                    best_task_arg = args
                    best_path = path
            except Exception as e:
                self.logger.exception('### Task FAILED!!\n')

        # self.logger.info(
        #     f'<<< Best metric: {best_metric}, path: {best_path}..')
        # self.logger.info(best_task_arg)
        # self.logger.info(f'>>> Start testing...')
        # try:
        #     test_task = Task(best_task_arg.log_dir, logger=self.logger,
        #                      device=self.device, verbose=verbose)
        #     best_task_arg.ckpt = best_path.split('/')[-1]
        #     test_task.test(best_task_arg)
        # except:
        #     self.logger.exception('### Test FAILED!!')


class ParallelScheduler(BasicScheduler):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.devices = list(range(torch.cuda.device_count()))
        # self.devices = [torch.device(i)
        #                 for i in range(torch.cuda.device_count())]
        self.args_q = Queue()
        self.task_q = Queue()
        self.lock = Lock()
        self.log_dir = args.log_dir
        assert len(self.search_params) > 0
        log_file = os.path.join(args.log_dir, 'test.txt')
        self.logger = construct_logger(args.log_dir, log_file, True, True)

    def emit_tasks(self, Task, verbose=True, descending_metric=True):
        for args in self.args_pool:
            self.args_q.put(args)
        processes = list()
        for device in self.devices:
            p = Process(target=self.run_parallel, args=(device, Task))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()

        best_metric = 100000 if descending_metric else 0
        best_task_arg, best_path = SimpleNamespace(), ''
        while not self.task_q.empty():
            args, metric, path= self.task_q.get(block=False)
            if (descending_metric and metric < best_metric) or \
                    (not descending_metric and metric > best_metric):
                best_metric = metric
                best_path = path
                best_task_arg = args
        self.logger.info(
            f'<<< Best metric: {best_metric}, path: {best_path}..')
        self.logger.info(best_task_arg)
        # self.logger.info(f'>>> Start testing...')
        try:
            test_task = Task(best_task_arg.log_dir, logger=self.logger,
                             device=self.device, verbose=verbose)
            best_task_arg.ckpt = best_path.split('/')[-1]
            test_task.test(best_task_arg)
        except:
            self.logger.exception('### Test FAILED!!\n')


    def run_parallel(self, device, Task):
        # device = torch.device(device)
        torch.cuda.set_device(device)
        log_file = os.path.join(self.log_dir, 'test.txt')
        logger = construct_logger(self.log_dir, log_file, True)

        while True:
            try:
                args = self.args_q.get(block=False)
            except:
                break
            args = vars(args)
            if len(self.search_params) == 0:
                sub_log_dir = ''
            else:
                tokens = [f'{k}{str(args[k]):.6}' for k in self.search_params]
                sub_log_dir = '_'.join(tokens)
            args['log_dir'] = os.path.join(args['log_dir'], sub_log_dir)
            args = SimpleNamespace(**args)
            self.lock.acquire()
            logger.info('>>> Start task..')
            logger.info(str(args)+'\n')
            self.lock.release()
            try:
                task = Task(args.log_dir, device='cuda', verbose=False)
                metric, path = task.launch(args)
                self.task_q.put((args, metric, path))
                self.lock.acquire()
                logger.info(f'<<< Finish task..')
                logger.info(args)
                logger.info(f'<<< Task metric: {metric}, path: {path}..\n')
            except Exception as e:
                logger.exception('### Task FAILED!!\n')
            try:
                # may release an unlocked lock
                self.lock.release()
            except ValueError as e:
                pass
        logger.info(f'<<< process:{device} finish..\n')


if __name__ == '__main__':
    args = SimpleNamespace(a=[1, 2, 3], b='abc', c=['d', 'e', 'f'])
    scheduler = BasicScheduler(args)
