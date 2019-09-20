import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self, num_input, num_classes, action):
        super().__init__()

        self.num_classes = num_classes
        self.num_input = num_input
        self.action = action

        cnn_filter_size = [c[0] for c in action] # filter size
        cnn_num_filters = [c[1] for c in action] #num of filters
        max_pool_ksize = [c[2] for c in action] # max pool k size
        dropout_rate = [c[3] for c in action] # drop out rate

        modules = []

        for i, filter_size in enumerate(cnn_filter_size):
            if filter_size.item() % 2 ==0:
                for_pd = filter_size.item() + 1
            else :
                for_pd = filter_size.item()
            padding = int(for_pd/2 - 0.5)
            conv = torch.nn.Conv2d(self.num_input, cnn_num_filters[i].item(),filter_size.item(),stride=1, padding = padding)
            #init.xavier_uniform_(conv_out.weight)
            #init.constant_(conv_out.bias,0.1)
            relu = torch.nn.ReLU(conv)
            maxpool = torch.nn.MaxPool2d(max_pool_ksize[i].item(), stride=1, padding=max_pool_ksize[i].item()//2)

            dropout = nn.Dropout(dropout_rate[i].item()/100.0)

            modules.append(conv)
            modules.append(relu)
            modules.append(maxpool)
            modules.append(dropout)

            self.num_input = cnn_num_filters[i].item()

        self.net = nn.Sequential(*modules)

    def forward(self, input):
        print("net \n",self.net)
        x = self.net(input)
        x = torch.flatten(x)
        classifier = nn.Linear(len(x), self.num_classes)
        logit = classifier(x)
        return logit

'''
# for test,
if __name__ == '__main__':
    cnn = CNN(784,10,[[[ 1 , 2 , 3 , 4 , 5 , 6 , 7 ,8]]])

    print(cnn)

'''

