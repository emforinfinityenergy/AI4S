import torch
import torch.nn as nn
import pandas as pd


class ExpDataset(torch.utils.data.Dataset):
    def __init__(self, c1, l1, c2, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.c1 = c1
        self.c2 = c2

    def __len__(self):
        return len(self.l1)

    def __getitem__(self, idx):
        return self.c1[idx], self.l1[idx], self.c2[idx], self.l2[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(1., requires_grad=True, dtype=torch.float64), requires_grad=True)

    def forward(self):
        ret = (self.k + 1) / (self.k * 5 + 3)
        return ret


def train(data: pd.DataFrame):
    num_epochs = 5000

    net = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    c1_data = torch.tensor(data['c1'].values, dtype=torch.float64).view(-1, 1)
    l1_data = torch.tensor(data['l1'].values, dtype=torch.float64).view(-1, 1)
    c2_data = torch.tensor(data['c2'].values, dtype=torch.float64).view(-1, 1)
    l2_data = torch.tensor(data['l2'].values, dtype=torch.float64).view(-1, 1)
    dataset = ExpDataset(c1_data, l1_data, c2_data, l2_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)

    for epoch in range(num_epochs):
        for c1, l1, c2, l2 in loader:
            optimizer.zero_grad()
            pred = net()
            loss = criterion(pred, l1 * c1 / l2 / c2)
            loss.backward()
            optimizer.step()
        if epoch % 500 == 0:
            with torch.no_grad():
                pred = net()
                loss = criterion(pred, l1_data * c1_data / l2_data / c2_data)
                print("Epoch", epoch, "loss", loss.item())
    return net
