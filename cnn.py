import torch
import torch.nn as nn


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
                for_pd = filter_size.item()+1
            else :
                for_pd = filter_size.item()
            padding = int(for_pd/2 - 0.5)
            padding_pool = int((max_pool_ksize[i].item()+1)/2 - 0.5)
            conv = torch.nn.Conv2d(self.num_input, cnn_num_filters[i].item(),filter_size.item(),stride=1, padding = padding)
            #init.xavier_uniform_(conv_out.weight)
            #init.constant_(conv_out.bias,0.1)
            relu = torch.nn.ReLU(conv)
            maxpool = torch.nn.MaxPool2d(max_pool_ksize[i].item(), stride=None, padding=padding_pool)

            dropout = nn.Dropout(dropout_rate[i].item()/100.0)

            modules.append(conv)
            modules.append(relu)
            modules.append(maxpool)
            modules.append(dropout)

            self.num_input = cnn_num_filters[i].item()

        self.net = nn.Sequential(*modules)
        self.classifier = nn.Linear(cnn_num_filters[-1], self.num_classes)

    def forward(self, input):
        #print("net \n",self.net)
        x = self.net(input)
        input_shape = x.shape
        x = x.view(input_shape[0], -1)
        logit = self.classifier(x)
        return logit

