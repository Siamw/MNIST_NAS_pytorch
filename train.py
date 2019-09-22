import torch
import torch.nn as nn
import torchvision
import argparse
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
from reinforce import Reinforce
from reinforce import Reward
from torch.distributions import Bernoulli
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default ='./data' )
parser.add_argument('--batch_size', type = int, default=100, help="batch size")
parser.add_argument('--lr', type=float, default= 5e-2, help="weights learning rate")
#parser.add_argument('--gpus', default='0,1,2,3', help = "gpu device id")
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--max_layers', type = int, default = 2)
parser.add_argument('--channels', type = int, default = 16)
args = parser.parse_args()

args.max_layers = int(args.max_layers)
#args.gpus = [int(s) for s in args.gpus.split(',')]



def train_arch(trainset, validset):
    max_layers = 2
    global_step = 500
    MAX_EPISODES = 100
    step = 0
    pre_acc = 0.0 # 이전 세대의 accuracy

    state = torch.tensor([[10.0, 128.0, 1.0, 1.0] * max_layers])

    total_rewards = 0

    state_h, action_h, reward_h  =[],[],[]

    reinforce = Reinforce(state, max_layers, global_step) # only initialize
    Rewards = Reward(num_input= 784, num_classes=10, learning_rate=0.001, batch_size=100)
    action = state
    for episode in range(MAX_EPISODES):
        print("episode: ",episode)
        step += 1
        action = reinforce.get_action(action)
        #m = Bernoulli(action)
        print("ca:", action)
        # 선택한 행동으로 환경에서 한 타임스탭 진행 후 샘플 수집

        if all(ai>0 for ai in action[0][0]):

            reward, pre_acc = Rewards.get_reward(action,pre_acc,trainset,validset)
            # 행동과 타임스탭진행후의 샘플이 동일하기 때문에 따로 진행하지 않고 바로 또 넘어가고 ~...
            print("====>", reward, pre_acc)
        else:
            reward = -1.0


        score = round(total_rewards,2)
        print(" score: ", reward, "time_step: ", step)

        state = action[0]
        reinforce.storeRollout(state,reward) # butter에 저장

        # 에피소드마다 정책신경망 업데이트
        reinforce.update_policy(episode, args.batch_size, step)


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
        _mnist_train, batch_size=args.batch_size, num_workers=4, sampler=train_sampler)
    mnist_valid = torch.utils.data.DataLoader(
        _mnist_train, batch_size=args.batch_size, num_workers=4, sampler=valid_sampler)

    train_arch(mnist_train, mnist_valid)

if __name__ == '__main__':\
    # for test
    #state = np.array([[10.0, 128.0, 1.0, 1.0] * 2], dtype=np.float32)
    #state = torch.tensor(state, requires_grad=False)
    #policy_network(state,2)
    main()

