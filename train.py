import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import numpy as np
from reinforce import Reinforce
from reinforce import Reward

#def model():

def policy_network(state, max_layers):
    nas_cell = nn.LSTM(state,100,4*max_layers)
    outputs, state = nas_cell(state) # output = tensor of shape (batch_size, seq_length, hidden_size)

    bias = np.array([0.05]*4*max_layers)
    print("bias")
    print(bias)
    print(outputs)
    outputs = outputs+bias
    print("outputs and bias")
    print(bias)
    print(outputs)


def train(dataset):
    max_layers = 2
    global_step = 500

    # self, optimizer, policy_network, max_layers, global_step,
    # division_rate=100.0, reg_param=0.001, discount_factor=0.99, exploration=0.3):
    reinforce = Reinforce(policy_network, max_layers, global_step)

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0,1.0,1.0]*max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0

    for episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)



def main():
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform = None) # have to add transform
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform = None) # have to add transform

    train(mnist_train)

if __name__ == '__main__':
    main()

