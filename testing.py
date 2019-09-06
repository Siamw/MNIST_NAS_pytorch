import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import random

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
        #print(input)
        #print(target)
        input = Variable(input, volatile=True)
        target = Variable(target, volatile=True)

def policy_network(state,max_layers):

    nas_cell = nn.LSTMCell(8, 100)
    fc = nn.Linear(100,8, bias=None)
    state = state[0][:]
    print(state)
    state = state.reshape(1,8)

    input = torch.stack((state,state,state,state,state,state,state,state), dim = 0)

    output = torch.empty((8,100,8))
    for i in range(8):
        hx, cx = nas_cell(input[i])
        print(hx)
        print(cx)
        print(" edfdsfasedfasefasefsef")
        output[i] = fc(hx)

    print("finding...")
    output_=output[:,-1:,:]
    print(output[:,-1:,:])
    #print(output_[-1:,:,:]*100) #******************* have to do this

    #print(out[:,-1:,:])

    return output[:,-1:,:]

if __name__ == '__main__':
    print(np.array([[random.sample(range(1,35),8)]]))
    # one episode = result of controller.. 그게 아니면 gradient가 action이 되어서 조금씩 optimal로 움직여야함..

    state=np.array([[10.0, 128.0, 1.0, 1.0] * 2], dtype=np.float32)
    max_layer = 2
    state = torch.tensor(state, requires_grad = False)
    policy_network(state,max_layer)


    # main()