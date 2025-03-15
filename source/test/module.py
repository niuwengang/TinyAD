import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, x):
        x=x+1
        return x
    
test_model= TestModel()
x=torch.tensor(0.1)
y=test_model(x)
print(y)