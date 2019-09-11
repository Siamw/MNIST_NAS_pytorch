import torch
import torch.nn as nn
import random
import numpy as np
from cnn import CNN
from torch.autograd import Variable
import torch.nn.functional as F


class Reinforce(nn.Module):
    def __init__(self,policy_network, max_layers, global_step,
                 division_rate=100.0, reg_param=0.001, discount_factor=0.99, exploration=0.3):

        self.policy_network=policy_network
        self.max_layers = max_layers
        self.global_step = global_step
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor =discount_factor
        self.exploration=exploration
        self.fc = nn.Linear(100,8)

        self.reward_buffer = []
        self.state_buffer = []

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.99)# eps, weight_decay, momentum, centered=False
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,0.96) # last_epoch = -1

        self.policy_outputs = self.policy_network(self.states, self.max_layers)

        policy_network_variables =
        self.discounted_rewards = np.array([None, ])
        self.logprobs = self.policy_network(self.states, self.max_layers)

        # compute policy loss and regularization loss
        self.cross_entropy_loss = F.nll_loss(F.softmax(self.logprobs[:, -1, :]),self.states)
        self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
        self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])  # Regularization
        self.loss = self.pg_loss + self.reg_param * self.reg_loss

        # compute gradients
        self.gradients = self.optimizer.compute_gradients(self.loss)

        # compute policy gradients
        for i, (grad, var) in enumerate(self.gradients):
            if grad is not None:
                self.gradients[i] = (grad * self.discounted_rewards, var)

        # training update
        with tf.name_scope("train_policy_network"):
            # apply gradients to update policy network
            self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

    def get_action(self, state): # action 반환
        output = self.fc(self.policy_outputs)
        output = output * 100
        self.predicted_action = output.to(dtype=torch.int64)
        #state로 계산
        return self.predicted_action


    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])


class Reward(nn.Module): # net_manater.py
    def __init__(self, num_input, num_classes, learning_rate, mnist, batch_size,
                 max_step_per_action=5500*3,
                 dropout_rate=0.85):
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist
        self.batch_size = batch_size
        self.max_step_per_action = max_step_per_action
        self.dropout_rate=dropout_rate

    def get_reward(self, action,pre_acc,trainset,validset):

        action=[action[0][0][x:x+4] for x in range(0, len(action[0][0]),4)]
        cnn_drop_rate = [c[3] for c in action]

        # criterion, optimizer, train(loss minimize)
        model = CNN(num_input=784,num_classes=10,action=action)
        # model.cuda
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),self.learning_rate)

        for steps, (input, target) in enumerate(trainset):
            model.train()
            n = input.size(0)
            input = Variable(input, requires_grad = False)
            target = Variable(target, requires_grad = False)

            logits = model(input)
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











