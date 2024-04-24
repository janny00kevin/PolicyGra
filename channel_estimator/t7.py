import torch

n_T = 2
T = n_T*1
data_size = 4
H_mean = 0
H_sigma = 0.01
n_R = 3
W_mean = 0
W_sigma = 0.001

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
# print(H)
# print(X)
# print(W)
print(Y)
Y_shuffle = Y.reshape(data_size, n_R ,T)
print(Y_shuffle)
idx = torch.randperm(Y_shuffle.shape[0])
print(idx)
Y_shuffle = Y_shuffle[idx,:,:]
print(Y_shuffle)
# y = torch.flatten(torch.view_as_real(Y))
# print(y.unsqueeze(0))
# print(Y.size())
# for i in range(Y.size(1)//T): ## != num_minibatch because here num_epoch = 1
#     # print(i)
#     print(Y[:,T*i:T*i+T])