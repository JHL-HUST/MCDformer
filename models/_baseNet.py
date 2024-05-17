import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

from models._baseTrainer import Trainer


class BaseNet(nn.Module):
    def __init__(self, hyper = None, logger = None):
        '''
        self.hyper = hyper\n
        self.logger = logger
        '''
        super(BaseNet, self).__init__()
        
        self.hyper = hyper
        self.logger = logger

        for (arg, value) in hyper.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x
    
    def load_pretraing_file(self, file_path = None, tag = 'pretraining'):
        try:
            device_id = next(self.parameters()).get_device()
            
            model_state = torch.load(file_path, map_location = f'cuda:{device_id}') if device_id > -1 else torch.load(file_path)
            for key,value in self.state_dict().items():
                print(key)
            for key,value in model_state.items():
                print(key)
            
            self.load_state_dict(model_state)
            self.logger.info(f'Successfully loading the {tag} file in the location: {file_path}!')
        except:
            self.logger.exception(
                    '{}\nGot an error on loading the {} file in the location: {}.\n{}'.format('!'*50, tag, file_path, '!'*50))
            raise SystemExit()

    def xfit(self, train_loader, val_loader):
        """
        If self.hyper has pretraining_file, then directly loading the pretraining states;
        else, training the model with the Traniner in models._baseTrainer.
        
        Return: fit_info
        """
        pretraining_tag = False
        fit_info = None
        if 'pretraining_file' in self.hyper.dict and self.hyper.pretraining_file is not None and os.path.exists(self.hyper.pretraining_file):
            
            self.logger.info(f'Finding pretraining file in the location {self.hyper.pretraining_file}')
            self.load_pretraing_file(file_path=self.hyper.pretraining_file)
            pretraining_tag = True
        
        if pretraining_tag is False:
            fit_info = self._xfit(train_loader, val_loader)
            
            checkpoint_file = os.path.join(self.hyper.model_fit_dir, 'checkpoint', self.hyper.data_name + '_' + f'{self.hyper.model_name}' + '.best.pt')
            
            self.load_pretraing_file(file_path=checkpoint_file, tag='checkpoint') 
        
        return fit_info
    

    def _xfit(self, train_loader, val_loader):
        net_trainer = Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info

    def predict(self, sample):
        """
        Return: prediction label of each sample as torch.tensor.
        """        
        sample = sample.to(self.hyper.device)
        logit = self.forward(sample)
        pre_lab = torch.argmax(logit, 1).cpu()
        return pre_lab
    
# if __name__ == '__main__':
#     model = AMC_Net(11, 128, 3)
#     x = torch.rand((4, 2, 128))
#     y = model(x)
