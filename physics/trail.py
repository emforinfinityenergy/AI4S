import math
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


def load_data(file_path):
    # 读取CSV文件
    # read the CSV file
    df = pd.read_csv(file_path)
    # 获取时间t, x位置和y位置，并转换为Tensor，同时确保数据类型为torch.float32
    # Retrieve time t, x, y, and convert them to Tensors, ensuring that the data type is torch.float32
    t_obs = torch.tensor(df['Time'].values, dtype=torch.float64).view(-1, 1)
    x_obs = torch.tensor(df['X'].values, dtype=torch.float64).view(-1, 1)
    y_obs = torch.tensor(df['Y'].values, dtype=torch.float64).view(-1, 1)
    t_start = t_obs[0].detach().numpy().item()  # 训练集数据和测试集数据的时间都是从0开始的
    t_end = t_obs[-1].detach().numpy().item()

    return t_start, t_end, t_obs, x_obs, y_obs


class PointData(Dataset):
    def __init__(self, t_seq, p_seq):
        self.t_seq = t_seq
        self.p_seq = p_seq

    def __len__(self):
        return len(self.t_seq)

    def __getitem__(self, idx):
        return self.t_seq[idx], self.p_seq[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32, dtype=torch.float64), nn.Tanh(),
            nn.Linear(32, 64, dtype=torch.float64), nn.Tanh(),
            # nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(64, 2, dtype=torch.float64)
        )

    def forward(self, x):
        return self.net(x)


def train(loader: DataLoader) -> nn.Module:
    t1 = time.time()

    # Train the model
    net = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    NUM_EPOCHS = 1000

    for epoch in range(NUM_EPOCHS):
        for t, p in loader:
            optimizer.zero_grad()
            loss = criterion(net(t), p)
            loss.backward()
            optimizer.step()
        # if epoch % 50 == 0:
        #     with torch.no_grad():
        #         loss = criterion(net(t_seq), p_seq)
        #         print(f"Epoch {epoch}, loss = {loss}")

    print("Time Elapsed: ", time.time() - t1)
    return net


def calculate(data_path: str):
    # Process the data
    t_s, t_e, t_seq, x_seq, y_seq = load_data(data_path)
    p_seq = torch.tensor(torch.zeros([len(t_seq), 2]), dtype=torch.float64)
    for i in range(len(t_seq)):
        p_seq[i] = torch.tensor([x_seq[i], y_seq[i]], dtype=torch.float64)
    dataset = PointData(t_seq, p_seq)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(p_seq)

    net = train(loader)

    # Evaluate the model
    eva_t = np.linspace(t_s, t_e, 100000)
    xs = []
    ys = []

    for t in eva_t:
        pre = net(torch.tensor([t], dtype=torch.float64))
        xs.append(pre[0].item())
        ys.append(pre[1].item())

    plt.figure(figsize=(16, 16))
    for i in range(len(t_seq)):
        plt.plot(x_seq[i], y_seq[i], 'o')
    plt.plot(x_seq, y_seq, color="red")

    plt.plot(xs, ys, color="blue")
    plt.show()

    sx = 0
    sy = 0
    sm = 0
    cnt = 0
    samples = np.linspace(t_s, t_e, 5000)
    for i in range(5000):
        time_tensor = torch.tensor([samples[i]], dtype=torch.float64, requires_grad=True)
        op = net(time_tensor)
        # print(op)

        dy_dt = torch.autograd.grad(op[1], time_tensor, grad_outputs=torch.ones_like(op[1]),
                                    create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, time_tensor, grad_outputs=torch.ones_like(dy_dt),
                                      create_graph=True)[0]
        dx_dt = torch.autograd.grad(op[0], time_tensor, grad_outputs=torch.ones_like(op[1]),
                                    create_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt, time_tensor, grad_outputs=torch.ones_like(dy_dt),
                                      create_graph=True)[0]

        # if abs(dy_dt.item()) > 10 or abs(dx_dt.item()) > 10 or abs(d2y_dt2.item()) > 10 or abs(d2x_dt2.item()) > 10:
        #     continue
        # print("dx", dx_dt)
        # print("dy", dy_dt)
        # print(d2x_dt2, d2y_dt2)
        v_gs = math.sqrt(dx_dt ** 2 + dy_dt ** 2)
        kx = d2x_dt2 / (dx_dt * v_gs)
        ky = (d2y_dt2 + 9.8) / (dy_dt * v_gs)
        # print(kx, ky)
        sx += kx.item()
        sy += ky.item()
        sm += (kx.item() + ky.item()) / 2
        # print((kx + ky) / 2)
        cnt += 1

    k = sx / cnt * -1
    print(f"{k=}")

    eva_samples = np.linspace(t_s, t_e, 100000)
    max_height = -1
    max_point = None
    max_v = None
    for sample in eva_samples:
        sample_tensor = torch.tensor([sample], dtype=torch.float64, requires_grad=True)
        pred = net(sample_tensor)
        if pred[1].item() > max_height:
            max_point = (pred[0].item(), pred[1].item())
            dx_dt = torch.autograd.grad(pred[0], sample_tensor, grad_outputs=torch.ones_like(pred[1]),
                                        create_graph=True)[0]
            dy_dt = torch.autograd.grad(pred[1], sample_tensor, grad_outputs=torch.ones_like(pred[1]),
                                        create_graph=True)[0]
            max_v = (dx_dt, dy_dt)
            max_height = pred[1].item()

    print(max_point)
    print(max_v)

    end_tensor = torch.tensor([t_e], dtype=torch.float64, requires_grad=True)
    pred = net(end_tensor)
    dx_dt = torch.autograd.grad(pred[0], end_tensor, grad_outputs=torch.ones_like(pred[1]),
                                create_graph=True)[0]
    dy_dt = torch.autograd.grad(pred[1], end_tensor, grad_outputs=torch.ones_like(pred[1]),
                                create_graph=True)[0]
    slope = dy_dt / dx_dt
    offset = pred[1].item() - slope * pred[0].item()
    print(dx_dt, dy_dt)
    print(f"y = {slope.item()}x + {offset.item()}")
    s = -1 * offset / slope

    return k, max_height, s.item()
