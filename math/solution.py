import math
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

NUM_EPOCHS = 500000


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float64), requires_grad=True)
        # self.b = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=True)

    def forward(self, f) -> torch.Tensor:
        return torch.matmul(f, self.w)


def train(a_data):
    a_val = []
    features = []
    labels = []
    for i in range(len(a_data)):
        a_val.append(a_data["a_seq"][i])

    for i in range(3, len(a_data)):
        features.append([a_val[i - 2], a_val[i - 3]])
        labels.append([a_val[i]])

    features = torch.tensor(features, dtype=torch.float64)
    labels = torch.tensor(labels, dtype=torch.float64)

    dataset = data.TensorDataset(*(features, labels))
    data_iter = data.DataLoader(dataset, batch_size=4, shuffle=True)

    net = Net()
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(NUM_EPOCHS):
        # print(features, labels)
        l = loss(net(features), labels.T[0])
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # for x, y in data_iter:
        #     l = loss(net(x), y.T[0])
        #     optimizer.zero_grad()
        #     l.backward()
        #     optimizer.step()

    return net


t1 = time.time()
dt = pd.read_csv("a_seq_train.csv")
print(dt)
ret = train(dt)
print(ret.w)
print(time.time() - t1)
