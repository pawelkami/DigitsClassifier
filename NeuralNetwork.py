from __future__ import print_function
from classificationalgo import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(81, 50)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x)


class NeuralNetworkHoG(ClassificationAlgorithm):
    def __init__(self, learning_rate=0.005, momentum_rate=0.5):
        self.model = Net()
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum_rate)

    def train(self, samples, responses, epoch_size=1):
        images = torch.from_numpy(samples)
        labels = responses.astype(np.long)
        labels = torch.from_numpy(labels)
        train_dataset = torch.utils.data.TensorDataset(images, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset)

        for epoch in range(1, epoch_size + 1):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data[0]))

    def predict(self, samples):
        images = torch.from_numpy(samples)
        if torch.cuda.is_available():
            images = images.cuda()

        self.model.eval()
        test_loss = 0
        correct = 0
        predictions = torch.FloatTensor()
        for image in images:
            if torch.cuda.is_available():
                image = image.cuda()

            data = Variable(image.unsqueeze(0), volatile=True)
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)))
        return predictions.numpy().squeeze()
