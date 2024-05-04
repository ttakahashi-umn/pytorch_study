import numpy as np
import torch

r0 = torch.tensor(1.0).float()

print("r0")
print(type(r0))
print(r0.dtype)
print(r0.shape)
print(r0.data)

r1_np = np.array([1, 2, 3, 4, 5])
print(r1_np.shape)

r1 = torch.tensor(r1_np).float()
print("r1")
print(r1.dtype)
print(r1.shape)
print(r1.data)

r2_np = np.array([[1, 5, 6], [4, 3, 2]])
print("r2")
print(r2_np.shape)

r2 = torch.tensor(r2_np).float()
print(r2.dtype)
print(r2.shape)
print(r2.data)

torch.manual_seed(123)

r3 = torch.randn((3, 2, 2))

print("r3")
print(r3.dtype)
print(r3.shape)
print(r3.data)

r4 = torch.ones((2, 3, 2, 2))

print("r4")
print(r4.dtype)
print(r4.shape)
print(r4.data)

r5 = r1.long()

print("r5")
print(r5.dtype)
print(r5.shape)
print(r5.data)

r6 = r3.view(3, -1)

print("r6")
print(r6.dtype)
print(r6.shape)
print(r6.data)

r7 = r3.view(-1)

print("r7")
print(r7.dtype)
print(r7.shape)
print(r7.data)

print(f"requires_grad: {r1.requires_grad}")
print(f"device: {r1.device}")

item = r0.item()

print(type(item))
print(item)

print(r2)
print(r2.max())
print(torch.max(r2, 1))
