import numpy as np
import matplotlib.pyplot as plt
import torch
from torchviz import make_dot

x_np = np.arange(-2, 2.1, 0.25)

print(x_np)

x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)

print(x)

y = 2 * x**2 + 2

print(y)

plt.plot(x.data, y.data)
plt.show()
z = y.sum()
g = make_dot(z, params={'x': x})
display(g)
