import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self,num_input,num_classes,action):
        self.num_input = num_input
        self.num_classes = num_classes
        self.action = action
