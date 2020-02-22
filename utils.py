from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import torch


def use_svg_display():
    """用矢量图显示"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """设置图的尺寸"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    shuffle_idxes = np.random.permutation(num_examples)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(shuffle_idxes[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(X, w, b):
    """定义一个线性模型"""
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    """MSE，还没有除以样本数量进行平均操作"""
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    """实现一个sgd"""
    for param in params:
        param.data -= lr * param.grad / batch_size


