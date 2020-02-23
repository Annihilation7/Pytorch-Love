import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
import torch.optim as optim


if __name__ == '__main__':

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(
        np.random.normal(0, 0.01, size=labels.size()), dtype=labels.dtype
    )  #

    batch_size = 10
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    for X, y in data_iter:
        print(X, y)
        break

    # 定义网络的几种方法
    # 方法1
    class LinearNet1(nn.Module):
        def __init__(self, n_feature):
            super(LinearNet1, self).__init__()
            self.linear = nn.Linear(n_feature, 1)

        def forward(self, x):
            y = self.linear(x)
            return y

    linear1 = LinearNet1(num_inputs)
    print(linear1)
    print('-' * 50)

    # 方法2
    linear2 = nn.Sequential(
        nn.Linear(num_inputs, 1)
    )
    print(linear2)
    print(linear2[0])
    print('-' * 50)

    # 方法3
    linear3 = nn.Sequential()
    linear3.add_module('linear', nn.Linear(num_inputs, 1))
    print(linear3)
    print(linear3[0])
    print('-' * 50)

    # 方法4
    linear4 = nn.Sequential(
        OrderedDict([
            ('linear', nn.Linear(num_inputs, 1))
        ])
    )
    print(linear4)
    print(linear4[0])
    print('-' * 50)

    # 就用linear4了
    for param in linear4.parameters():
        print(param)

    # initialize
    # 要比linear4[0].weight表意更明确
    init.normal_(linear4.linear.weight, mean=0, std=0.01)
    init.constant_(linear4.linear.bias, val=0)

    # loss func
    loss = nn.MSELoss()

    # optimizer
    lr = 0.03
    optimizer = optim.SGD(linear4.parameters(), lr=lr, momentum=0.9)
    print(optimizer)

    # 手动调整学习率也很方便
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

    # train stage
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            output = linear4(X)
            train_loss = loss(output, y.view(-1, 1))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        print('epoch {}, loss {}'.format(epoch, train_loss))

    print(
        'after train stage:\ntrue_w: {}, '
        'pred_w: {}\ntrue_b: {}, pred_b: {}'.format(
            true_w, linear4.linear.weight.tolist(),
            true_b, linear4.linear.bias.tolist()
        )
    )