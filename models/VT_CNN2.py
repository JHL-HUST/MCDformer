import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models._baseNet import BaseNet

class VTCNN(BaseNet):
    '''Refer to https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb'''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
                    
        output_dim = hyper.num_classes
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            # nn.ZeroPad2d(padding = (2, 2)),
            nn.Conv2d(in_channels = 1, out_channels = 256, padding=(0,2), kernel_size = (1, 3)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            # nn.ZeroPad2d(padding = (0, 2)),
            nn.Conv2d(in_channels = 256, out_channels = 80, padding=(0,2), kernel_size=(2, 3)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # fc_input_dim = (self.hyper.sig_len + 4 - 2 + 4 - 2 ) * 80
        
        # assert fc_input_dim == 10560
        
        self.fc1 = nn.Sequential(
            nn.Linear(10560, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, output_dim)
        )        
        
        self.initialize_weight()
        self.to(self.hyper.device)
                
    def forward(self, x):
        x = x.view(x.shape[0],1, 2, 128)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_uniform_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.kaiming_normal_(m.bias)
                # nn.init.constant_(m.bias, 0)        