import torch.nn as nn


# get from https://github.com/kimiandj/gsw
class MLP(nn.Module):
    """MNIST Encoder from Original Paper's Keras based Implementation.
    Args:
        init_num_filters (int): initial number of filters from encoder image channels
        lrelu_slope (float): positive number indicating LeakyReLU negative slope
        inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
        embedding_dim (int): embedding dimensionality
    """

    def __init__(self, din=2, dout=10, num_filters=32, depth=3):
        super(MLP, self).__init__()
        self.din = din
        self.dout = dout
        self.init_num_filters = num_filters
        self.depth = depth

        self.features = nn.Sequential()

        for i in range(self.depth):
            if i == 0:
                self.features.add_module("linear%02d" % (i + 1), nn.Linear(self.din, self.init_num_filters))
            else:
                self.features.add_module(
                    "linear%02d" % (i + 1), nn.Linear(self.init_num_filters, self.init_num_filters)
                )
            self.features.add_module("activation%02d" % (i + 1), nn.LeakyReLU(inplace=True))

        self.features.add_module("linear%02d" % (i + 2), nn.Linear(self.init_num_filters, self.dout))

    def forward(self, x):
        return self.features(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def reset(self):
        self.features.apply(self.init_weights)
