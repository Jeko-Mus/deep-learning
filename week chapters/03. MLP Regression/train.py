#import the needed libraries

import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import Net

x_train, x_test, y_train, y_test = dh.load_data('data/turkish_stocks.csv')

model = Net()

lr = 1e-1
n_epochs = 100

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    y_hat = model(x_train)
    
    loss = loss_func(y_train, y_hat)
    loss.backward()    
    optimizer.step()
    

# Remember to validate your model: with torch.no_grad() ...... model.eval .........model.train
