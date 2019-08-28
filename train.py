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


def train_arch(trainset, validset):
    max_layers = 2
    global_step = 500
    MAX_EPISODES = 2500
    step = 0
    pre_acc = 0.0
    total_rewards = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0] * max_layers], dtype=np.float32)

    input_size = 4 * max_layers
    hidden_size = 100
    #output_size =
    #batch_size =
    #length =

    _finder = nn.LSTM(input_size, hidden_size, num_layers=max_layers, bias = True)

    reinforce = Reinforce(_finder,max_layers, global_step) # only initialize

    pre_acc = 0.0 # 이전 세대의 accuracy
    for episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai>0 for ai in action[0][0]):
            reward, pre_acc = Reward.get_reward(action,pre_acc,validset)
            print("====>", reward, pre_acc)
        else:
            reward = -1.0
        total_rewards +=reward

        state = action[0]
        reinforce.storeRollout(state,reward) # butter에 저장

        # loss 출력 위한 코드
        """
        step += 1
        ls = reinforce.train_step(1)
        log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)
        """



def main():
    _mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform = None) # have to add transform
    _mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform = None) # have to add transform

    train_valid = list(range(_mnist_train))
    split = int(np.floor(0.2 * len(_mnist_train)))

    mnist_train = torch.utils.data.DataLoader(
      _mnist_train, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_valid[:split]),batch_size=100, num_workers=1)
    mnist_valid = torch.utils.data.DataLoader(
        _mnist_test, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_valid[split:len(_mnist_train)]), batch_size=100, num_workers=1)

    train_arch(mnist_train, mnist_valid)

if __name__ == '__main__':
    main()

