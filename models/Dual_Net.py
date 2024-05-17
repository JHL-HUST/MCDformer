import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models._baseNet import BaseNet


class Stream(nn.Module):
    def __init__(self):
        super(Stream, self).__init__()
        # input(batch, 1, 2, 128)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = (1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.conv3 = nn.Sequential(
             nn.BatchNorm2d(256),
             nn.ZeroPad2d((2, 2, 0, 0)),
             nn.Conv2d(in_channels= 256, out_channels= 80, kernel_size=(1,3)),
             nn.ReLU(inplace=True),
             nn.Dropout(0.5)
        )

        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size= 80, hidden_size= 100, num_layers= 1, batch_first= True, dropout = 0.5),
        )
        # self.dropout_1 = nn.Dropout(0.5)
        self.lstm2 = nn.Sequential(
            nn.LSTM(input_size= 100, hidden_size= 50, num_layers= 1, batch_first= True, dropout = 0.5),
        )
        # self.dropout_2 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        # x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], 80, -1)
        x = x.transpose(1,2)
        x, (h,c) = self.lstm1(x)
        # x = self.dropout_1(x)
        x, (h,c) = self.lstm2(x)
        # x = self.dropout_2(x)
        x = x[:,-1:,:]
        return x

class DualNet(BaseNet):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
        output_dim = hyper.num_classes
        self.steam_iq = Stream()
        self.steam_pa = Stream()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 2500, out_features= output_dim),
        )
        self.initialize_weight()
        self.to(self.hyper.device)

    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)

        # x_t = x[:,:,0,:] + x[:,:,1,:]*1j
        # x_fft = torch.fft.fft(x_t)

        x_pa = torch.zeros_like(x)
        # x_pa[:,:,0,:] = torch.real(x_fft)
        # x_pa[:,:,1,:] = torch.imag(x_fft)
        x_pa[:,:,0,:] = torch.sqrt(torch.sum(torch.square(x),dim=-2))
        x_pa[:,:,1,:] = torch.arctan(x[:,:,1,:]/x[:,:,0,:])

        iq_feat = self.steam_iq(x)
        pa_feat = self.steam_pa(x)
        mul_feat = torch.matmul(iq_feat.transpose(-1,-2),pa_feat)
        mul_feat = mul_feat.reshape([mul_feat.shape[0],-1])
        # mul_feat = torch.cat([iq_feat,pa_feat],dim=-1)

        out = self.fc1(mul_feat)
        # out = F.softmax(out)
        return out

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

if __name__ == '__main__':
    model = DualNet(11)
    x = torch.rand((4,1, 2, 128))
    y = model(x)
    print(y.shape)

