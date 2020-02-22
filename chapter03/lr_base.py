import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import torch
import numpy as np
import utils
import matplotlib.pyplot as plt


if __name__ == '__main__':

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(
        np.random.normal(0, 0.01, size=labels.size()), dtype=labels.dtype
    )  # 加一点高斯噪声

    print(features[0], labels[0])

    utils.set_figsize()
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    # plt.show()

    batch_size = 10
    for X, y in utils.data_iter(batch_size, features, labels):
        print(X, y)
        break

    # initialize
    w = torch.tensor(
        np.random.normal(0, 0.01, size=(num_inputs, 1)), dtype=torch.float32
    )
    b = torch.zeros(1, dtype=torch.float32)

    # grad
    w.requires_grad_(True)
    b.requires_grad_(True)

    # train
    lr = 0.04
    num_epochs = 3
    net = utils.linreg
    loss = utils.squared_loss

    for epoch in range(num_epochs):
        for X, y in utils.data_iter(batch_size, features, labels):
            loss = utils.squared_loss(net(X, w, b), y).sum()
            loss.backward()  # 为每个可训练的参数产生梯度
            utils.sgd([w, b], lr, batch_size)
            # 对于pytorch，不要忘记梯度要清零
            w.grad.data.zero_()
            b.grad.data.zero_()
        # 每个epoch看一下训练集loss
        train_loss = utils.squared_loss(net(features, w, b), labels)
        print('epoch {}, loss {}'.format(epoch, train_loss))

    print(
        'after train stage:\ntrue_w: {}, '
        'pred_w: {}\ntrue_b: {}, pred_b: {}'.format(
            true_w, w.view(-1).tolist(), true_b, b.tolist()
        )
    )
