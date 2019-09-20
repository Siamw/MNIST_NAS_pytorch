import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import random
import torch.nn.functional as F
import keyword

import tensorflow as tf
from torch.autograd import Variable


def main():
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
         normalize])

    _mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform = train_transform) # have to add transform
    _mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform = valid_transform) # have to add transform

    train_valid = list(range(len(_mnist_train)))
    split = int(np.floor(0.2 * len(_mnist_train)))

    np.random.seed(1)
    np.random.shuffle(train_valid)

    train_idx, valid_idx = train_valid[split:], train_valid[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
    mnist_train = torch.utils.data.DataLoader(
        _mnist_train, batch_size=100,num_workers=1, sampler = train_sampler)
    mnist_valid = torch.utils.data.DataLoader(
        _mnist_train, batch_size=100, num_workers=1, sampler=valid_sampler)

    print(len(mnist_train))
    print(len(mnist_valid))

    print("ww")

    #for steps, (input, target) in enumerate(mnist_valid):
    for steps, (input, target) in enumerate(mnist_valid):
        input = torch.FloatTensor(input)
        target = torch.FloatTensor(target)

        input = Variable(input, volatile=True)
        target = Variable(target, volatile=True)

def policy_network(state,max_lay):

    nas_cell = nn.LSTM(input_size=8, hidden_size=128, num_layers=2)
    fc = nn.Linear(128,8, bias=None)
    print(state)

    h_0 = torch.randn(2,1,128)
    c_0 = torch.randn(2,1,128)

    output, (hx, cx) = nas_cell(state.unsqueeze(0), (h_0,c_0))

    output =output[:, -1:, :]


    output = F.sigmoid(fc(output))

    #output = F.softmax(output, 2)
    #output = F.softmax(output,2)

    output = output * 100
    print("after softmax:", output)


    output = output.to(dtype=torch.int64)
    return output

if __name__ == '__main__':
    state=torch.tensor([[10.0, 128.0, 1.0, 1.0] * 2])
    max_layer = 2
    #state = torch.tensor(state, requires_grad = False)
    action = policy_network(state,max_layer)
    print("action : ", action[0])


