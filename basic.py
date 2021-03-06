import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph
y = w * x + b

# Compute gradients
y.backward()

# Print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)
