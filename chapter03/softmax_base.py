import utils
import torch


if __name__ == '__main__':

    batch_size = 256
    # 获取dataloader
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    num_inputs = 784  # softmax回归是fc结构，所以28*28=784
    num_outputs = 10  # fashion mnist是10类

    # 定义trainable parameters
    w = torch.normal(mean=0, std=0.01, size=(784, 10))
    b = torch.zeros(10)
    w.requires_grad_(True)
    b.requires_grad_(True)

    # 测试softmax
    X = torch.rand(2, 5)
    x_prob = utils.softmax(X)
    print(x_prob)
    print(x_prob.sum(1))

    # 在做softmax分类/回归的时候，有一个节省显存的小技巧，就是不用将label展开成onehot
    # 再做crossentroyp，用某个tensor的gather方法可以很优雅的实现根据label idx来筛
    # 选pred中对应位置的值，再乘积。因为其他的位置label都是0，对loss是没有贡献的


    def net(X):
        return utils.softmax(torch.mm(X.view(-1, num_inputs), w) + b)
    num_epochs = 6
    lr = 0.01

    utils.train_ch3(
        net, train_iter, test_iter, utils.cross_entropy, num_epochs, batch_size,
        [w, b], lr
    )

    # evaluate and show
    X, y = iter(test_iter).next()
    true_labels = utils.get_fashion_mnist_labels(y.numpy())
    pred_labels = utils.get_fashion_mnist_labels(net(X).argmax(1).numpy())
    labels = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    utils.show_fashion_mnist(X[:9], labels[:9])