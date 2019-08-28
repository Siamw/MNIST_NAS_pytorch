import torch
import torch.nn as nn
import random
import numpy as np
from cnn import CNN

class Reinforce(nn.Module):
    def __init__(self,_finder, max_layers, global_step,
                 division_rate=100.0, reg_param=0.001, discount_factor=0.99, exploration=0.3):

        self._finder=_finder
        self.max_layers = max_layers
        self.global_step = global_step
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor =discount_factor
        self.exploration=exploration

        self.reward_buffer = []
        self.state_buffer = []

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.99)# eps, weight_decay, momentum, centered=False
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,0.96) # last_epoch = -1

        #model input
        self.states = np.array([[10.0, 128.0, 1.0, 1.0] * max_layers], dtype=np.float32)
        # predict_actions
        self.policy_outputs = self._finder(self.states, self.max_layers)
        self.action_scores = tf.identify(self.policy_outputs)
        self.predicted_action = tf.case(tf.scalar_mul(self.division_rate, self.action_scores))

        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        #compute discounts
        self.discounted_rewards = np.array([None,])
        #policy network
        self.logprobs = self._finder(self.states, self.max_layers)
        #compute policy loss and regularization loss

    def cal_



    def create_variables(self):
        self.states = torch.randn(None,self.max_layers*4)


    def get_action(self, state): # action 반환
        return

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

    def get_reward(self, action,pre_acc,validset):

        action=[action[0][0][x:x+4] for x in range(0, len(action[0][0]),4)]
        cnn_drop_rate = [c[3] for c in action]

        # criterion, optimizer, train(loss minimize)
        model = CNN(self.num_input, self.num_classes, action)
        # model.cuda
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            self.learning_rate)

        for steps in range(self.max_step_per_action):
           batch_X, batch_y =





