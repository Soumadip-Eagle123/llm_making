import torch


class LinearLayer:
    '''A simple linear layer that performs a linear transformation on the input tensor.'''
    def __init__(self, input_size, output_size):
        self.W = torch.randn(output_size, input_size)
        self.b = torch.randn(output_size)
    def input(self, x):
        return self.W @ x + self.b
    def calculateMSELoss(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

input_size = int(input("Enter the input size: "))
output_size = int(input("Enter the output size: "))
linear_layer = LinearLayer(input_size, output_size)

x = []
for i in range(input_size):
    x.append(float(input(f"Enter value for input {i + 1}: ")))
x = torch.tensor(x)
y = linear_layer.input(x)
y_desired = torch.tensor([6.7, 6.9])

print(f"Output of the linear layer: {y}")
loss = linear_layer.calculateMSELoss(y, y_desired)
print(f"Mean Squared Error Loss: {loss}")

'''
print(x)
print(x.shape)
W = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
print(W)
print(W.shape)
b = torch.tensor([1.0, 2.0])
print(b)
print(b.shape)
y = W @ x + b
print(y)
print(y.shape)
'''