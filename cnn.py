import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self, num_input, num_classes, action):

        action=[action[0][0][x:x+4] for x in range(0, len(action[0][0]),4)]
        print(action)
        cnn_filter_size = [c[0] for c in action]
        cnn_num_filters = [c[1] for c in action]
        max_pool_ksize = [c[2] for c in action]
        dropout_rate = [c[3] for c in action]

        modules = []
        cin = num_input
        flag = 0
        for idd, filter_size in enumerate(cnn_filter_size):
            conv_out = torch.nn.Conv1d(cin, cnn_num_filters[idd],filter_size,stride=1,padding="SAME")
            init.xavier_uniform_(conv_out.weight)
            init.constant_(conv_out.bias,0.1)

            conv = torch.nn.ReLU(conv_out)

            maxpool = torch.nn.MaxPool1d(max_pool_ksize[idd], stride=1, padding=max_pool_ksize[idd]//2)


            dropout = nn.Dropout(dropout_rate[idd]/100.0)
            dropout = dropout(maxpool)

            modules.extend([conv,maxpool,dropout])

        print(modules)
        for i in range(6):
            print(i)
            print(modules[i])
        #self.model = nn.Sequential(*modules)
        #print(self.model)

        self.nas_cell = nn.LSTM(8,100,num_layers=2,bias=True)



    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':


    cnn = CNN(784,10,[[[ 5 , 5 , 5 , 5 , 5 , 5 , 5 ,17]]])



