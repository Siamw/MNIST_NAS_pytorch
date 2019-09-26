import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
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

    def forward(self,new_state):

        output, (hx, cx) = self.nas_cell(new_state.unsqueeze(0), (self.h_0, self.c_0))

        output = output[:, -1:, :]
        output = F.sigmoid(self.fc(output))

        print("finding...")

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


        self.optimizer = torch.optim.RMSprop(policy_network.parameters(self), lr=0.99)# eps, weight_decay, momentum, centered=False
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,0.96) # last_epoch = -1

        self.policy_network = policy_network(self.state, self.max_layers)



    def update_policy(self,episode): # = train_step
        batch_size = 2
        if episode > 0 and (episode+1) % batch_size == 0: # 원래는 % batch_size ... step이 없는거 아닌가 ! ?
            step = batch_size
                # discount reward
            running_add = 0
            for i in reversed(range(step)):
                if self.reward_buffer[i] == 0:
                    running_add = 0
                else: # using discount factor,
                    running_add = running_add * self.discount_factor + self.reward_buffer[i]
                    self.reward_buffer[i] = running_add

            # Normalize reward
            reward_mean = np.mean(self.reward_buffer)
            reward_std = np.std(self.reward_buffer)
            for i in range(step):
                self.reward_buffer[i] = (self.reward_buffer[i] - reward_mean) /reward_std

            # Gradient Descent
            self.optimizer.zero_grad()

            for i in range(step):
                print("i i i i i i : ",i )
                state = self.state_buffer[i]
                reward = self.reward_buffer[i]
                #reward = torch.FloatTensor([self.reward_buffer[i]])
                action = self.state_buffer[i]
                #action = torch.FloatTensor([self.state_buffer[i]]) # in here, state = action
                print("floatTensor : ",action)
                print("reward : ", reward)

                probs = torch.log(self.policy_network(self.state))
                loss = -reward * probs

                print("probs : ", probs)
                print("loss : ", loss)

                loss.mean().backward()

            self.optimizer.step()

            self.reward_buffer = []
            self.state_buffer = []


    def get_action(self,new_state): # action 반환
        output = self.policy_network(new_state)
        output = torch.tensor(output)#, requires_grad=False)
        output = torch.mul(output,100)
        predicted_action = output.to(dtype=torch.int32)
        #predicted_action = predicted_action.numpy()
        #state로 계산
        return predicted_action

    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

class Reward(nn.Module): # net_manater.py
    def __init__(self, num_input, num_classes, learning_rate, batch_size,
                 max_step_per_action=3,
                 dropout_rate=0.85):
        super().__init__()
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        #self.mnist = mnist
        self.batch_size = batch_size
        self.max_step_per_action = max_step_per_action # change to epoches
        self.dropout_rate=dropout_rate

    def get_reward(self,action,pre_acc,trainset,validset):

        action=[action[0][0][x:x+4] for x in range(0, len(action[0][0]),4)]

        model = CNN(num_input=1,num_classes=10,action=action)
        criterion = nn.CrossEntropyLoss()

        model.cuda()
        criterion.cuda()
        cudnn.benchmark = True

        optimizer = torch.optim.Adam(model.parameters(),self.learning_rate)

        model.train()

        for i in range(self.max_step_per_action):

            for steps, (input, target) in enumerate(trainset):

                input = Variable(input, requires_grad=False)
                target = Variable(target, requires_grad=False)
                logits = model(input)
                loss = criterion(logits,target)
                #loss = criterion(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if(steps % 200 == 0):
                    stepss = steps + i*500
                    print("Step " + str(stepss) + ", Minibatch Loss= " + "{:.4f}".format(loss))
                #if (steps == 3):
                #    break;


        test_loss =0
        correct =0
        stepp = 0

        for steps, (input, target) in enumerate(validset):

            input = Variable(input, requires_grad=False)
            target = Variable(target, requires_grad=False)

            output =model(input)
            test_loss += F.nll_loss(output,target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # 몇개나 같은지,,,

            stepp = steps # validset 개수

            #if(steps == 2):
            #    break;
        test_loss /= (stepp+1)
        acc_ = correct / ((stepp+1)*100) # step별 100개 있으니까..

        print("validation accuracy : "+ str(acc_))

        if acc_ - pre_acc <= 0.01:
            return acc_, acc_
        else:
            return 0.01, acc_

