import torch
import torch.nn.functional as F
from torch import nn
#from torchsummary import summary

# YOUR CODE HERE

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = torch.nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.layer(x)
        return x

model = Net()
print(model)
