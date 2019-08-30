import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np

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
        print(input)
        print(target)
        input = Variable(input, volatile=True)
        target = Variable(target, volatile=True)

def policy_network(max_layers):
    #encoder = nn.Embedding(4*max_layers, 100)
    nas_cell = nn.LSTMCell(8, 100)
    #input  = torch.randn(8,3,8) # seq_len, batch, input_size
    input= torch.randn(8,1,8)
    h0 = torch.zeros(1, 100)
    c0 = torch.zeros(1, 100)

    out = []

    for i in range(8):
        hx, cx = nas_cell(input[i], (h0,c0)) # output = tensor of shape (batch_size, seq_length, hidden_size)
        out.append(hx)

    print(out)
    print(np.shape(out))


    lstm = nn.LSTM(input_size=8, hidden_size=100, num_layers=1, batch_first=True)
    h0 = torch.randn(1, 1, 100)
    c0 = torch.randn(1,1,100)

    output,(ht,ct) = lstm(out)

    print(output.data)


    print(out[:,-1:,:])
    return out[:,-1:,:]

if __name__ == '__main__':
    state = np.array([[10.0, 128.0, 1.0, 1.0]*2], dtype=np.float64)
    print(state)
    print(np.shape(state))
    max_layer = 2
    state = torch.tensor(state, requires_grad = False)
    policy_network(max_layer)


    # main()