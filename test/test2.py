import numpy as np
import torch
from torch.autograd import Variable as v
import torch.nn as nn

print("hello world!")

# f = np.array([[1,2,3],[4,5,6]])
# print(f.sum(axis=-1).sum(axis=-1).shape)

# print (np.array([3])*3)
# m = v(torch.FloatTensor([[2, 3]]), requires_grad=True)
# print(type(j.data[0].item()))

# ## matrix multiplication
# j = torch.Tensor([[2,3]])
# k = torch.Tensor([[4],[5]])
# print(j.size(), k.size())
# print(k@j)

# j = torch.Tensor([[2,3]]).requires_grad_(True)
# k = j*2
# k.sum().backward()
# print(j.grad)

# def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
#     # Build a feedforward neural network.
#     layers = []
#     for j in range(len(sizes)-1):
#         act = activation if j < len(sizes)-2 else output_activation
#         layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#     return nn.Sequential(*layers)

# logit_net = mlp([4,32,2])

# for i in range(2):
#     print(i)

# b = np.array([2 + 1j, 3 + 4j, 5 + 2j])
# a = np.array([[1+2j, 3+4j], [1+3j,2+1j]])
# print(np.matmul(a,a))

# b = np.random.rand(2,3,4)
# a = np.arange(24).reshape([2,3,4])
# c = np.arange(12).reshape([3,4])
# print(c)
# print(c.transpose().reshape(-1))

# real_array = np.arange(8).reshape([2,4])
# print(real_array)

# # 将实数数组视图转换为复数数组
# complex_array = real_array.view(np._ComplexValue)

# # 显示转换后的复数数组
# print("Complex array:", complex_array)


# a = np.random.normal(1,2,4)
# b = a.view(np.complex128)
# print(np.sqrt(a[0]**2 + a[1]**2 +a[2]**2 + a[3]**2))
# print(np.linalg.norm(b))

a = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
print(a.mean(-2))