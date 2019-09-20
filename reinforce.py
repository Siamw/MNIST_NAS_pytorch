import torch
import torch.nn as nn
import random
import numpy as np
from cnn import CNN
from torch.autograd import Variable
import torch.nn.functional as F


class policy_network(nn.Module):
    def __init__(self, state, max_layers):
        super().__init__()
        self.state = state
        self.max_layers = max_layers
        self.nas_cell = nn.LSTM(input_size=8, hidden_size=100, num_layers=self.max_layers)
        self.fc = nn.Linear(100, 8)

        self.h_0 = torch.randn(2, 1, 100)
        self.c_0 = torch.randn(2, 1, 100)

    def forward(self):
        output, (hx, cx) = self.nas_cell(self.state.unsqueeze(0), (self.h_0, self.c_0))

        output = output[:, -1:, :]
        output = F.sigmoid(self.fc(output))

        print("finding...")

        print(output)
        return output


class Reinforce(nn.Module):
    def __init__(self,state, max_layers, global_step,
                 division_rate=100.0, reg_param=0.001, discount_factor=0.99, exploration=0.3):
        super().__init__()
        self.state = state
        self.max_layers = max_layers
        self.global_step = global_step
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor =discount_factor
        self.exploration=exploration
        self.fc = nn.Linear(100,8)

        self.criterion = nn.CrossEntropyLoss()
        self.reward_buffer = []
        self.state_buffer = []

        ################## have to change
        self.optimizer = torch.optim.RMSprop(policy_network.parameters(self), lr=0.99)# eps, weight_decay, momentum, centered=False
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,0.96) # last_epoch = -1

        self.policy_network = policy_network(self.state, self.max_layers)
        #self.policy_outputs =



    def update_policy(self,episode, batch_size, step): # = train_step
        if episode > 0 and episode % batch_size == 0:
                # discount reward
            running_add = 0
            for i in reversed(range(step)):
                if self.reward_buffer[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * self.discount_factor + self.reward_buffer[i]
                    self.reward_buffer[i] = running_add

            # Normalize reward
            reward_mean = np.mean(self.reward_buffer)
            reward_std = np.mean(self.reward_pool)
            for i in range(step):
                self.reward_buffer[i] = (self.reward_buffer[i] - reward_mean) /reward_std

            # Gradient Descent
            self.optimizer.zero_grad()

            for i in range(step):
                state = self.state_buffer[i]
                action = Variable(torch.FloatTensor([self.state_buffer[i]])) # in here, state = action
                reward = self.reward_buffer[i]

                probs = self.policy_net(self.state,self.max_layers)
                loss = -probs.log_brop(action) * reward
                loss.backward()

            self.optimizer.step()

            self.reward_buffer = []
            self.state_buffer = []


    def get_action(self): # action 반환

        # to tensor
        output = self.policy_network()
        print(output)
        output = torch.tensor(output)#, requires_grad=False)
        output = torch.mul(output,100)
        predicted_action = output.to(dtype=torch.int64)
        #state로 계산
        return predicted_action
    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])


class Reward(nn.Module): # net_manater.py
    def __init__(self, num_input, num_classes, learning_rate, batch_size,
                 max_step_per_action=5500*3,
                 dropout_rate=0.85):
        super().__init__()
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        #self.mnist = mnist
        self.batch_size = batch_size
        self.max_step_per_action = max_step_per_action
        self.dropout_rate=dropout_rate

    def get_reward(self,action,pre_acc,trainset,validset):
        action=[action[0][0][x:x+4] for x in range(0, len(action[0][0]),4)]
        cnn_drop_rate = [c[3] for c in action]

        # criterion, optimizer, train(loss minimize)
        model = CNN(num_input=1,num_classes=10,action=action)
        # model.cuda
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),self.learning_rate)

        for steps, (input, target) in enumerate(trainset):
            model.train()
            n = input.size(0)
            print("ddddddddd")
            print(input.shape)
            print(target.shape)
            input = Variable(input, requires_grad = False)
            target = Variable(target, requires_grad = False)

            print(input.shape)
            #print(input)
            logits = model(input)
            print(logits.shape)
            optimizer.zero_grad()
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            if steps % 100 == 0 :
                acc = model.accuracy
                print("Step " + str(steps) +
                                  ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                  ", Current accuracy= " + "{:.3f}".format(acc))

        for steps, (input, target) in enumerate(validset):
            input = Variable(input, requires_grad=False)
            target = Variable(target, requires_grad=False)

            logits = model(input)
            acc_ = model.accuracy

        if acc_ - pre_acc <= 0.01:
            return acc_, acc_
        else:
            return 0.01, acc_



    '''
      # cross entropy + reg_param * regularization loss . reg 헷갈리니 일단 빼고 진행
    def get_loss(self):
        output = self.fc(self.policy_outputs)
        output = F.softmax(output,2)
        #
        pg_loss = self.criterion(output, state) # policy gradient loss

        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        self.loss = pg_loss + l2_reg * self.reg_param

        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.discount_rewards = torch.placeholder
        for i, (grad, var) in enumerate(self.gradients):
            if grad is not None:
                self.gradients[i] = (grad + self.discounted_rewards, var)
    '''







