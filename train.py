import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
from reinforce import Reinforce
from reinforce import Reward

#def model():

def policy_network(state, max_layers):

    nas_cell = nn.LSTMCell(8, 100)
    fc = nn.Linear(100,8, bias=None)
    state = state[0][:]
    state = state.reshape(1,8)
    input = torch.stack((state,state,state,state,state,state,state,state), dim = 0)
    output = torch.empty((8,100,8))

    for i in range(8):
        hx, cx = nas_cell(input[i])
        output[i] = fc(hx)

    print("finding...")
    output_=output[:,-1:,:]
    print(output[:,-1:,:])
    #print(output_[-1:,:,:]*100) #******************* have to do this

    #print(out[:,-1:,:])
    return output[:,-1:,:]


def train_arch(trainset, validset):
    max_layers = 2
    global_step = 500
    MAX_EPISODES = 2500
    step = 0
    pre_acc = 0.0
    total_rewards = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0] * max_layers], dtype=np.float32)

    reinforce = Reinforce(policy_network,max_layers, global_step) # only initialize

    pre_acc = 0.0 # 이전 세대의 accuracy
    for episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai>0 for ai in action[0][0]):
            reward, pre_acc = Reward.get_reward(action,pre_acc,trainset,validset)
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
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    _mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                                  transform=train_transform)  # have to add transform
    _mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                                 transform=valid_transform)  # have to add transform

    train_valid = list(range(len(_mnist_train)))
    split = int(np.floor(0.2 * len(_mnist_train)))

    np.random.seed(1)
    np.random.shuffle(train_valid)

    train_idx, valid_idx = train_valid[split:], train_valid[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
    mnist_train = torch.utils.data.DataLoader(
        _mnist_train, batch_size=100, num_workers=1, sampler=train_sampler)
    mnist_valid = torch.utils.data.DataLoader(
        _mnist_train, batch_size=100, num_workers=1, sampler=valid_sampler)

    train_arch(mnist_train, mnist_valid)

if __name__ == '__main__':
    main()

