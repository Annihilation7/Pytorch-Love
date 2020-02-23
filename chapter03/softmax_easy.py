import utils
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    class Net(nn.Module):
        def __init__(self, in_features, out_features):
            super(Net, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

            # initialize
            nn.init.normal_(self.linear.weight, mean=0, std=0.01)
            nn.init.zeros_(self.linear.bias)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)

    net = Net(num_inputs, num_outputs)

    # loss
    # 无需对label进行onehot，只需label idx即可
    # 无需对output进行softmax，直接往里塞即可，非常方便
    loss = nn.CrossEntropyLoss()

    # optimizer
    lr = 0.1
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # train
    num_epochs = 6
    utils.train_ch3(
        net, train_iter, test_iter, loss, num_epochs, batch_size,
        optimizer=optimizer
    )

