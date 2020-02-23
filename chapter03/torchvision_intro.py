

"""
torchvision主要包含以下几个模块：
1. dataset
2. models
3. transform
4. utils
"""


import torchvision
import utils
import torch.utils.data as Data
import time


if __name__ == '__main__':

    # 下面两个都是torch.utils.data.dataset类
    mnist_train = torchvision.datasets.FashionMNIST(
        root='/data/mazhenyu/Code_2rd/torch_love/data', train=True,
        transform=torchvision.transforms.ToTensor(), download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='/data/mazhenyu/Code_2rd/torch_love/data', train=False,
        transform=torchvision.transforms.ToTensor(), download=True
    )
    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))  # 60000, 10000

    # sample one
    feature, label = mnist_train[0]
    # ToTensor通道变成torch想要的格式了，label是idx而不是onehot
    print(feature.size(), label)

    # 看一下数据
    X, y = [], []
    show_num = 10
    for i in range(show_num):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    # utils.show_fashion_mnist(X, utils.get_fashion_mnist_labels(y))

    # dataloader
    batch_size = 256
    train_iter = Data.DataLoader(mnist_train, batch_size, True, num_workers=4)
    test_iter = Data.DataLoader(mnist_test, batch_size, False, num_workers=4)

    # 看一下读取数据集所需要的时间
    st = time.time()
    for X, y in train_iter:
        continue
    print('{:.2f} sec.'.format(time.time() - st))
