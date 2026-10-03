import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("Original:")
print(x)
print(x.shape)

y = x.reshape(3, 2)

print("\nReshaped:")
print(y)
print(y.shape)