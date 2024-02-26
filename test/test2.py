import numpy as np
import torch
from torch.autograd import Variable as v

print("hello world!")

# f = np.array([[1,2,3],[4,5,6]])
# print(f.sum(axis=-1).sum(axis=-1).shape)

print (np.array([3])*3)
m = v(torch.FloatTensor([[2, 3]]), requires_grad=True)
# print(type(j.data[0].item()))

# ## matrix multiplication
# j = torch.Tensor([[2,3]])
# k = torch.Tensor([[4],[5]])
# print(j.size(), k.size())
# print(k@j)

j = torch.Tensor([[2,3]]).requires_grad_(True)
k = (j*2).retain_grad()
k.sum().backward()
print(j.grad, k.grad)