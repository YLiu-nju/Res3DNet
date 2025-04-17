import argparse
import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
import time
import random
import yaml
import numpy as np
from sklearn import metrics
from .task import BasicTask, construct_logger
from .scheduler import BasicScheduler, ParallelScheduler
from .epochs import train_epoch, val_epoch, test_epoch
from .dataset import IDHDataset, create_folds
from .residual_i3d import Residual_I3D
from monai.losses import DiceLoss

class Timer(object):
    def __init__(self) -> None:
        super().__init__()
        self.t = time.time()

    def reset(self):
        self.t = time.time()

    def update(self):
        intv = time.time() - self.t
        self.t = time.time()
        return intv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False

class ProstateClsTask(BasicTask):
    def __init__(self, log_dir, logger=None, device=None, verbose=True) -> None:
        super().__init__(log_dir, logger=logger, device=device, verbose=verbose)
        self.inter_test_ids = []
        self.out_test_ids = []
        self.out_test_ids_gulou = []

    def launch(self, args):
        setup_seed(args.seed)
        self.logger.info('>>> Start training..')
        self.logger.info(args)
        self.logger.info('>>> Build model..')
        folds, folds_size = create_folds(args.root_dir, excel_name=args.excel_name, fold_num=5, alpha=2)
        print(folds_size)

        skip_fold=[]

        # cross validation
        for fold_id in range(5):
            if fold_id in skip_fold:
                print("Skip fold {} of {}.".format(fold_id + 1, 5))
                continue

            train_ids = folds[0]
            val_ids = folds[1]
            self.inter_test_ids = folds[2]
            self.out_test_tcga_ids = folds[3]
            self.out_test_ids_all = folds[4]

            model = Residual_I3D(args)
            model = model.to(self.device)
            if args.distributed:
                model = torch.nn.DataParallel(model)
            data_shape = list(map(lambda x: int(x), args.data_shape.split(',')))

            criterion = DiceLoss(square_denominator=True, set_level=True)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

            trainset = IDHDataset(args.root_dir, train_ids, data_shape, dataAug=True)
            valset = IDHDataset(args.root_dir, val_ids, data_shape, dataAug=False)

            train_loader = DataLoader(trainset, args.bs, True, num_workers=args.workers,pin_memory=True)
            val_loader = DataLoader(valset, args.bs, num_workers=args.workers,pin_memory=True)

            if args.ckpt is not None:
                self.resume(model, optimizer, args.ckpt)
            self.train(model, optimizer, criterion,
                       train_loader, val_loader, args.epochs,
                       num_classes=args.num_classes, mixup=args.mixup)
            self.test(args, self.best_metric, self.best_path, 'inter')
            self.test(args, self.best_metric, self.best_path, 'tcga')
            self.test(args, self.best_metric, self.best_path, 'all')

            del optimizer
            del model
            return self.best_metric, self.best_path

    def train(self, model, optimizer, criterion, train_loader, val_loader,
              epochs, save_intv=None, **kwargs):
        self.logger.info('>>> Start training..')
        if self.best_metric is None:
            self.best_metric = 0
        timer = Timer()
        np.set_printoptions(precision=2)
        for i in range(0, epochs-self.start_epoch+1):
            epoch = self.start_epoch + i
            timer.reset()
            train_acc_meter = metrics.accuracy_score()

            def train_metric_hook(pred, gt):
                nonlocal train_acc_meter
                train_acc_meter.add(pred, gt)

            train_loss = train_epoch(
                model, optimizer, criterion, train_loader,
                verbose=self.verbose, device=self.device,
                metric_hook=train_metric_hook, **kwargs)
            epoch_time = timer.update()
            train_acc = train_acc_meter.value()

            val_auc_meter = metrics.roc_auc_score(multi='ovr')
            val_prec_meter = metrics.precision_score()
            val_rec_meter = metrics.recall_score()
            val_acc_meter = metrics.accuracy_score()
            val_spec_meter = metrics.specificity_score()

            def val_metric_hook(pred, gt):
                nonlocal val_auc_meter, val_prec_meter
                nonlocal val_rec_meter, val_acc_meter
                nonlocal val_agree_meter,val_spec_meter
                val_auc_meter.add(pred, gt)
                val_prec_meter.add(pred, gt)
                val_rec_meter.add(pred, gt)
                val_acc_meter.add(pred, gt)
                val_spec_meter.add(pred, gt)

            val_loss = val_epoch(
                model, criterion, val_loader, metric_hook=val_metric_hook,
                verbose=self.verbose, device=self.device)
            val_f1_score = metrics.f1_score()

            self.log_epoch(epoch, epoch_time, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc_meter.value(), val_prec=val_prec_meter.value(), val_spec=val_spec_meter.value(), val_rec=val_rec_meter.value(), val_auroc=val_auc_meter.auc(), val_auprc=val_auc_meter.auprc(), val_f1=val_f1_score)

            val_metric = val_rec_meter.value() + val_prec_meter.value() + val_acc_meter.value()
            if val_metric > self.best_metric:
                self.save(model, optimizer, val_loss, epoch, best=True)
                self.best_metric = val_metric
            if save_intv is not None and epoch % save_intv == 0:
                self.save(model, optimizer, val_loss, epoch)

    def test(self, args, best_metirc, best_path, type='inter'):
        setup_seed(args.seed)

        model = Residual_I3D(args)
        model = model.to(self.device)

        self.resume(model, None, best_path)
        model.eval()

        self.logger.info('>>> Start testing..')
        print(os.path.join(args.log_dir, best_path))
        self.logger.info(os.path.join(args.log_dir, best_path))

        data_shape = list(map(lambda x: int(x), args.data_shape.split(',')))

        if type == 'inter':
            print(len(self.inter_test_ids))
            testset = IDHDataset(args.root_dir, self.inter_test_ids, data_shape, dataAug=False)
        elif type == 'tcga':
            print(len(self.out_test_tcga_ids))
            testset = IDHDataset(args.root_dir, self.out_test_ids, data_shape, dataAug=False)
        elif type == 'all':
            print(len(self.out_test_ids_all))
            testset = IDHDataset(args.root_dir, self.out_test_ids_gulou, data_shape, dataAug=False)

        test_loader = DataLoader(testset, args.bs, num_workers=args.workers, pin_memory=True)

        test_auc_meter = metrics.roc_auc_score()
        test_prec_meter = metrics.precision_score(multi=args.multi)
        test_rec_meter = metrics.recall_score(multi=args.multi)
        test_acc_meter = metrics.accuracy_score(multi=args.multi)
        test_spec_meter = metrics.specificity_score(multi=args.multi)

        def metric_hook(pred, gt):
            nonlocal test_auc_meter, test_prec_meter
            nonlocal test_rec_meter, test_acc_meter, test_spec_meter
            test_auc_meter.add(pred, gt)
            test_prec_meter.add(pred, gt)
            test_rec_meter.add(pred, gt)
            test_acc_meter.add(pred, gt)
            test_spec_meter.add(pred, gt)

        test_epoch(model, test_loader, metric_hook=metric_hook, verbose=self.verbose, device=self.device)
        test_f1_score = metrics.F1(test_prec_meter.value(), test_rec_meter.value())
        self.logger.info('>>> ' +
                         f'test_acc:{test_acc_meter.value():.3f} ' +
                         f'test_prec:{test_prec_meter.value():.3f} ' +
                         f'test_rec:{test_rec_meter.value():.3f} ' +
                         f'test_spec:{test_spec_meter.value():.3f} ' +
                         f'test_auc:{test_auc_meter.auc():.3f} ' +
                         f'test_auprc:{test_auc_meter.auprc():.3f} ' +
                         f'test_f1:{test_f1_score:.3f} '
                         )


def parse_args():
    parser = argparse.ArgumentParser(description='IDH Classification')
    parser.add_argument('--cfg', default=None, type=str)

    parser.add_argument('--bs', '--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=[0.00003], type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--log-dir', default=None, type=str)
    parser.add_argument('--num-classes', default=2, type=int)
    parser.add_argument('--num-experts', default=3, type=int)
    parser.add_argument('--wd', default=0.02, type=float)
    parser.add_argument('--growth-rate', default=16, type=int)
    parser.add_argument('--data-shape', default='32,96,96', type=str)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--multi',default=False, action='store_true')

    parser.add_argument('--root-dir', default='The path where data and excel is stored')
    parser.add_argument('--excel-name', default='excel name')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='i3d_residual', type=str)
    parser.add_argument('--inChannel', default=4, type=str)
    parser.add_argument('--workers', default=12, type=str)
    parser.add_argument('--verbose', default=True,action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--distributed', action='store_true')

    args = parser.parse_args()
    if args.cfg is not None:
        args_dict = vars(args)
        with open(args.cfg) as file:
            cfg_dict = yaml.load(file, Loader=yaml.FullLoader)
            if cfg_dict is not None:
                args_dict.update(cfg_dict)
        return
    return args

def main():
    args = parse_args()
    setup_seed(args.seed)
    if not args.test:
        if args.parallel:
            mp.set_start_method('spawn')
            scheduler = ParallelScheduler(args)
        else:
            scheduler = BasicScheduler(args)
        scheduler.emit_tasks(ProstateClsTask, args.verbose)
    else:
        log_file = os.path.join(args.log_dir, 'test.txt')
        logger = construct_logger(args.log_dir, log_file, args.verbose)
        task = ProstateClsTask(args.log_dir, logger,
                               device='cuda', verbose=args.verbose)
        task.test(args)

if __name__ == '__main__':
    main()
