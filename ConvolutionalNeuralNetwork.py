from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define all layers in model
        # First two are convolutional layers
        self.convolutional1 = nn.Conv2d(1, 10, kernel_size=5)
        self.convolutional2 = nn.Conv2d(10, 20, kernel_size=5)
        # Here we drop out of 2D layers (images) ...
        self.convolutional2_dropout = nn.Dropout2d()
        #  Next 2 are full connected linear layers
        self.full_connected1 = nn.Linear(320, 50)
        self.full_connected2 = nn.Linear(50, 10)

    def forward(self, x):
        # Define connection between next layers in model
        # Uses torch.nn.functional
        x = F.relu(F.max_pool2d(self.convolutional1(x), 2))
        x = F.relu(F.max_pool2d(self.convolutional2_dropout(self.convolutional2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.full_connected1(x))
        x = F.dropout(x, training=self.training)
        x = self.full_connected2(x)
        return F.log_softmax(x)


class ConvolutionalNeuralNetwork:
    def __init__(self, learning_rate=0.01, momentum_rate=0.5):
        # Loading pictures the best way for the PyTorch conv nets
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=64, shuffle=True, **kwargs)

        self.model = Net()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum_rate)

    def train(self, epochs=10):
        # Training session for neural network organized in epochs
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                # negative log likelihood loss
                loss = F.nll_loss(output, target)
                # back propagation
                loss.backward()
                self.optimizer.step()
                if batch_idx % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.data[0]))

    def evaluate(self):
        # Evaluates model - predict number (between 0 and 9) and prints confusion matrix
        self.model.eval()
        test_loss = 0
        correct = 0
        confusion = torch.zeros(10, 10)
        for data, target in self.test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for i in range(0, len(target)):
                confusion[int(target[i]), int(pred[i])] += 1

        test_loss /= len(self.test_loader.dataset)
        print('Accuracy: ({:.0f}%, Average loss: {:.4f},)'.format(
            100. * correct / len(self.test_loader.dataset),
            test_loss))

        print('confusion matrix:')
        print(confusion)
