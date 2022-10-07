import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.preprocessingLayer = None
        
        
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Tanh()
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.ReLU()
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.ReLU()
        self.fc7 = nn.Linear(100, 1)
        
        torch.nn.init.normal_(self.fc1.weight, 0, sigma)
        torch.nn.init.normal_(self.fc3.weight, 0, sigma)
        torch.nn.init.normal_(self.fc5.weight, 0, sigma)
        torch.nn.init.normal_(self.fc7.weight, 0, sigma)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)

        return out