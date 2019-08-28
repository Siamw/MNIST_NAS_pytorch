import torch
import torch.nn as nn
import random
import numpy as np

class Reinforce(nn.Module):
    def __init__(self, policy_network, max_layers, global_step,
                 division_rate=100.0, reg_param=0.001, discount_factor=0.99, exploration=0.3):

        self.policy_network=policy_network
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

        self.create_variables()
        #var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #self.sess.run(tf.variables_initializer(var_lists))

    def create_variables(self):
        self.states = torch.randn(None,self.max_layers*4)


    def get_action(self, state): # action 반환
        return


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

    def get_reward(self, action,step,pre_acc):
        action=[action[0][0][x:x+4] for x in range(0, len(action[0][0]),4)]
        cnn_drop_rate = [c[3] for c in action]


