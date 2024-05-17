import os.path
import time

import pandas as pd
import torch
from torch import optim, nn
from ray.experimental.tqdm_ray import tqdm as ray_tqdm
from tqdm import trange
from tqdm import tqdm as real_tqdm
# from torch.optim import lr_scheduler

import numpy as np
from task.util import os_makedirs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logger, patience=7, delta=0):
        """
        Args:
            logger: log the info to a .txt
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 cfg,
                 logger):
        super(Trainer, self).__init__()

        self.epochs_stats = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.train_acc_list = None
        self.train_loss_list = None
        self.val_acc = None
        self.val_loss = None
        self.train_acc = None
        self.best_monitor = None
        self.lr_list = None
        self.train_loss = None
        self.t_s = None
        self.early_stopping = None
        self.criterion = None
        self.optimizer = None

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.model = model.to(self.cfg.device)

        self.iter = 0

    def loop(self):
        
        self.checkpoint_folder = os.path.join(
            self.cfg.model_fit_dir, 'checkpoint')
        os_makedirs(self.checkpoint_folder)
        
        self.before_train()
        
        for self.iter in trange(0, self.cfg.epochs):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

        last_model_name = self.cfg.data_name + '_' + \
            f'{self.cfg.model_name}' + '.last.pt'
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_folder,last_model_name))

    @staticmethod
    def adjust_learning_rate(optimizer, gamma):
        """Sets the learning rate when we have to"""
        lr = optimizer.param_groups[0]['lr'] * gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.cfg.device)
        self.early_stopping = EarlyStopping(
            self.logger, patience=self.cfg.patience)

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def before_train_step(self):
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")
        
    def cal_loss_acc(self, sig_batch, lab_batch):
        logit = self.model(sig_batch)
        loss = self.criterion(logit, lab_batch)

        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ).item() / lab_batch.size(0)
        
        return loss, acc

    def run_train_step(self, ray = False):
        
        if ray:
            for step, (sig_batch, lab_batch) in ray_tqdm(enumerate(self.train_loader),  total=len(self.train_loader)):
                sig_batch = sig_batch.to(self.cfg.device)
                lab_batch = lab_batch.to(self.cfg.device)
                loss, acc = self.cal_loss_acc(sig_batch, lab_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_loss.update(loss.item())
                self.train_acc.update(acc)
        else:
            # pass
            with real_tqdm(total=len(self.train_loader),
                    desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                    postfix=dict,
                    mininterval=0.3) as pbar:
                for step, (sig_batch, lab_batch) in enumerate(self.train_loader):
                    sig_batch = sig_batch.to(self.cfg.device)
                    lab_batch = lab_batch.to(self.cfg.device)
                    loss, acc = self.cal_loss_acc(sig_batch, lab_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.train_loss.update(loss.item())
                    self.train_acc.update(acc)

                    pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                        'train_acc': self.train_acc.avg})
                    pbar.update(1)
                
        return self.train_loss.avg, self.train_acc.avg
                    

    def after_train_step(self):
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info('====> Epoch: {} Time: {:.2f} Train Loss: {:.6E} Train Acc: {:.3f}% lr: {:.5f}'.format(
            self.iter, time.time() - self.t_s, self.train_loss.avg, self.train_acc.avg * 100, self.lr_list[-1]))
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)
        


    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter()
        self.val_acc = AverageMeter()
        self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self, ray = False):
        if ray:
            for step, (sig_batch, lab_batch) in ray_tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                with torch.no_grad():
                    sig_batch = sig_batch.to(self.cfg.device)
                    lab_batch = lab_batch.to(self.cfg.device)

                    loss, acc = self.cal_loss_acc(sig_batch, lab_batch)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)
        else:
            with real_tqdm(total=len(self.val_loader),
                    desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                    postfix=dict,
                    mininterval=0.3,
                    colour='blue') as pbar:
                for step, (sig_batch, lab_batch) in enumerate(self.val_loader):
                    with torch.no_grad():
                        sig_batch = sig_batch.to(self.cfg.device)
                        lab_batch = lab_batch.to(self.cfg.device)

                        loss, acc = self.cal_loss_acc(sig_batch, lab_batch)

                        self.val_loss.update(loss.item())
                        self.val_acc.update(acc)

                        pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                            'val_acc': self.val_acc.avg})
                        pbar.update(1)

        return self.val_loss.avg, self.val_acc.avg
                        

    def after_val_step(self, checkpoint = True):
        if self.val_acc.avg >= self.best_monitor:
            self.best_monitor = self.val_acc.avg
            self.best_epoch = self.iter
            # toDo: change to annother location.
            if checkpoint:
                best_model_name = self.cfg.data_name + '_' + \
                    f'{self.cfg.model_name}' + '.best.pt'
                torch.save(self.model.state_dict(), os.path.join(
                    self.checkpoint_folder, best_model_name))
                
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Val Loss: {:.6E} Val Acc: {:.3f}%'.format(self.iter, time.time() - self.t_s, self.val_loss.avg, self.val_acc.avg * 100))
        self.logger.info('Best Epoch: {} \t Best Val Acc: {:.3f}%'.format(self.best_epoch, self.best_monitor * 100 ))

        self.early_stopping(self.val_loss.avg)

        if self.early_stopping.counter != 0 and self.early_stopping.counter % self.cfg.milestone_step == 0:
            self.adjust_learning_rate(self.optimizer, self.cfg.gamma)

        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)

        self.epochs_stats = pd.DataFrame(
            data={"epoch": range(self.iter + 1),
                  "lr_list": self.lr_list,
                  "train_loss": self.train_loss_list,
                  "val_loss": self.val_loss_list,
                  "train_acc": self.train_acc_list,
                  "val_acc": self.val_acc_list}
        )