import numpy as np
import torch

print("hello world!")
# #

# a = torch.tensor([1])
# print(a.item())

d = []
b = [1, 2, 3]
c = [1, 2]
# c = torch.tensor([[2,2],[2,2],[2,2]])
# print(len(c))
d.append(b.copy())
d.append(c.copy())
c = [1]
# print(d)

# e = np.array([[[ 0,  1,  2 , 3],
#   [ 4 , 5 , 6,  7],
#   [ 8 , 9 ,10, 11]],

#  [[12, 13, 14 ,15],
#   [16 ,17 ,18, 19],
#   [20, 21, 22, 23]]])
# f = np.array([[1,2,3],[4,5,6]])
# print(f.sum(axis=-1).sum(axis=-1).shape)

print (np.array([3])*3)