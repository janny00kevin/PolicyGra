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
# print(ct)

ct = [[row[0] for row in ct], [row[1] for row in ct]]
# print(np.array(ct[0])-np.array(ct[1]))

# print(np.array(ct[0]))
# print(np.linalg.norm())
# print(ct)

# print(ct[0:len(ct):2])
# print(np.linalg.norm(ct, axis=2))

# ct = torch.cat((t1, t2), dim=0)
# print("Concatenated tensor:", ct)

# ct = torch.cat((t1, t2), dim=-2)
# print("Concatenated tensor:", ct)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# print(device)
n_R = 2; n_T = 2; T = 2

# print(H)
H = torch.view_as_complex(torch.normal(1, 0.01, size=(n_R, n_T, 2))).to(device)
# print(H.type())
# x = torch.randn(2,2, dtype=torch.cfloat).to(device)
# print(x.type())
# print(H)
# H = torch.normal(0, 1, size=(2, 2*3))
# H = np.random.normal(1, 0.01, [2,2*3]).view(np.complex128)
# print(H)


# X = torch.eye(n_T).tile(int(T/n_T)).to(device)
X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
# print(H)
# print(H)
# print(H.matmul(X))
# a = torch.flatten(torch.view_as_real(H))
# print(a)

# print(torch.cat((torch.zeros(0), a), 0))

a = torch.tensor([1, 2, 3, 4, 5, 6])
# print(type(a.size(dim=0)/2))
# print(a[:int(a.size(dim=0)/2)])
# print(a[int(a.size(dim=0)/2):])
# print(a[:len(a)//2])
a = b = c = torch.zeros(0)