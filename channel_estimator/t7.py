import torch
# import time
import torch.nn as nn

n_T = 2
T = n_T*1
data_size = 3
H_mean = 0
H_sigma = 0.01
n_R = 2
W_mean = 0
W_sigma = 0.001
hidden_sizes = [64,32]

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Softplus):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T])
print(logits_net)

# print([2*n_R*T]+hidden_sizes+[2*2*n_R*n_T])

# X = torch.complex(torch.eye(2,2).tile(2), torch.zeros(n_T, T))
# H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R, n_T, 2)))
# W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R, T, 2)))
# Y = H.matmul(X) + W
# print(torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)))
# print(torch.complex(torch.eye(n_T).tile(T//n_T*data_size), torch.zeros(n_T, T, data_size)))

# print(torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).repeat(1,data_size))
# print(torch.complex(torch.eye(n_T).tile(T//n_T*data_size), torch.zeros(n_T, T*data_size)))
X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T))
H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T, 2)))
W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*data_size, T, 2)))
Y = H.matmul(X) + W

a1 = torch.Tensor(range(data_size*n_T*n_R))
# a2 = torch.Tensor(range(-12,0))
# A = torch.complex(a1, a2).reshape(data_size,n_R,T)
# print(A)
# k = 1
# print(A[k,:])
# ak = torch.flatten(torch.view_as_real(A[k,:]))
# print(ak)
# a = torch.flatten(torch.view_as_real(A))
# print(a[k*n_R*T*2:(k+1)*n_R*T*2])

# a = a1.reshape(data_size,n_T*n_R)
# print(a)
# print(a.mean(dim=0))
# print(a.mean(dim=1))

# a = [1, 2, 3]
# b = 4
# print(a[0],b)


# n = 10**8
# m = 10**4
# start = time.time()
# a = torch.zeros(n)
# for i in range(n//m):
#     a[i*m:m+i*m] = torch.rand(m)
# print(f'{time.time() - start} s')
# # print(a)

# start = time.time()
# a = torch.zeros(0)
# for i in range(n//m):
#     a = torch.cat((a, torch.rand(m)), 0)
# print(f'{time.time() - start} s')
# print(a)


# a = torch.zeros(10)
# b = torch.Tensor(range(3))
# a[5:5+3] = b
# print(a)