from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.utils.data as Data


"""lr_base"""
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


"""torchvision intro"""
def get_fashion_mnist_labels(labels):
    """labels是一些列label index"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    use_svg_display()

    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for fig, img, label in zip(figs, images, labels):
        fig.imshow(img.view(28, 28).numpy())
        fig.set_title(label)  # 所以这里的label是应该已经取好类别的集合
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_xaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size, num_workers=4):
    mnist_train = torchvision.datasets.FashionMNIST(
        root='/data/mazhenyu/Code_2rd/torch_love/data', train=True,
        transform=torchvision.transforms.ToTensor(), download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='/data/mazhenyu/Code_2rd/torch_love/data', train=False,
        transform=torchvision.transforms.ToTensor(), download=True
    )

    train_iter = Data.DataLoader(
        mnist_train, batch_size, True, num_workers=num_workers
    )
    test_iter = Data.DataLoader(
        mnist_test, batch_size, False, num_workers=num_workers
    )

    return train_iter, test_iter


"""softmax_base"""
def softmax(X):
    """对shape=[B, n]的tensor求softmax"""
    exp_x = torch.exp(X)
    return exp_x / exp_x.sum(1, keepdim=True)  # 利用了broadcast机制


def cross_entropy(y_hat, y):
    """y_hat shape: [B, n], y shape: [B]"""
    return -torch.log(y_hat.gather(1, y.view(y_hat.size(0), -1)))  # shape=[B, 1]


def accuracy(y_hat, y):
    """y_hat shape: [B, n], y shape: [B]"""
    # bool matrix -> float matrix -> mean -> scalar
    return (y_hat.argmax(1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_num = 0
    total_num = 0
    for X, y in data_iter:
        acc_num += (net(X).argmax(1) == y).sum().item()
        total_num += X.size(0)
    return acc_num / total_num


def train_ch3(
        net, train_iter, test_iter, loss, num_epochs, batch_size,
        params=None, lr=None, optimizer=None
):
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.zero_()

            l.backward()
            if optimizer is not None:
                optimizer.step()
            else:
                sgd(params, lr, batch_size)

            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(1) == y).sum().item()
            n += X.size(0)

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {}, loss {:.4f}, train_acc {:.4f}, test_acc {:.4f}'.format(
            epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc
        ))

