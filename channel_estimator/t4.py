import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal

# t = torch.tensor([0.0, 0.01, 2.0, 0.01])
# m = Normal(t[0:len(t):2], t[1:len(t):2])
# print(m.sample())
# print(t[0:len(t):2])

ct = []
t1 = torch.tensor([1, 1, 1])
n2 = np.array([2, 2, 2])

ct.append([t1,n2])  
ct.append([t1,n2])
print(ct)

ct = [[row[0] for row in ct], [row[1] for row in ct]]
print(np.array(ct[0])-np.array(ct[1]))
print(np.array(ct[0]))
# print(np.linalg.norm())
# print(ct)

# print(ct[0:len(ct):2])
# print(np.linalg.norm(ct, axis=2))

# ct = torch.cat((t1, t2), dim=0)
# print("Concatenated tensor:", ct)

# ct = torch.cat((t1, t2), dim=-2)
# print("Concatenated tensor:", ct)