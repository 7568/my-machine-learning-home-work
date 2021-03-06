# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/13
Description:
"""
import torch
from torch.utils import data
import w6.use_pytorch.d2l as d2l
import os
# `nn` is an abbreviation for neural networks
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK']='True'

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


net = nn.Sequential(nn.Linear(2, 1))
nn.MaxPool2d
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print(w)
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print(b)
print('error in estimating b:', true_b - b)