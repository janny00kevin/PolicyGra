import torch
from torch.distributions.normal import Normal

device = torch.device("cpu")

num_minibatch = 2
num_trajectories = 3
data_size = num_trajectories*num_minibatch
n_R = 2
n_T = 8
T = n_T
H_mean,H_sigma,W_mean,W_sigma = [0,0.01,0,0.01]

# # g = torch.Tensor(range(num_t*n_R*n_T)).reshape(num_t,n_R,n_T)
# Y0 = torch.view_as_complex(torch.Tensor(range(n_R*data_size*T*2)).reshape(n_R*data_size, T, 2)).to(device) # size: 12x4 48
# y0 = torch.view_as_real(Y0).reshape(data_size, n_R*T*2) # vectorize size: 6x16 96
# # print(Y0)
# # print(y0.size())
# for j in range(num_minibatch): # 2
#     tau_y = y0[j*num_trajectories:(j+1)*num_trajectories, :] # size: 3x16
#     # print(tau_y)
#     y1 = torch.view_as_complex((tau_y).reshape(num_trajectories, n_T*n_R, 2)) # take the norm line64
#     # print(y1)
#     # print(torch.norm(y1,dim=1))
# Y_shuffle = Y0.reshape(data_size, n_R ,T)
# idx = torch.randperm(Y_shuffle.shape[0])
# Y_shuffle = Y_shuffle[idx,:,:]
# print(Y_shuffle)
# print(Y_shuffle.reshape(n_R*data_size, T))

sqrt2 = 2**0.5
X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/sqrt2).to(device)
W_sigma = torch.Tensor([0, 10, 3])
W_mean = W_mean*torch.ones(data_size*len(W_sigma), n_R, T, 2)
# print(W_mean.size())
# W_sigma = torch.tile(torch.Tensor([0, 10]),(data_size,n_R,T), dims=1)

########## datasize stay the same but every snr has datasize/len(snr) size
W_sigma = W_sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
W_sigma = W_sigma.repeat(data_size//len(W_sigma), n_R, T, 2)

W_sigma = torch.Tensor([0, 10, 3])
W_sigma = W_sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
W_sigma = W_sigma.repeat(data_size, n_R, T, 2)
# print(W_sigma.shape)

W = torch.view_as_complex(torch.normal(W_mean, W_sigma)/sqrt2).to(device)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)

MC = A.shape[1]  # 这里 MC = 3

# 计算协方差矩阵
print(torch.mean(A, dim=1))
mean_A = torch.reshape(torch.mean(A, dim=1), (-1, 1))  # (2, 1)
print(mean_A)

# print(torch.normal(0, torch.tensor([1.0,2.0,3.0])))

# print(W)

# a = torch.Tensor([[1,2,3],[4,5,6]])
# print(a.shape)
# print(torch.mean(a,dim=1))
# print(torch.reshape(torch.mean(a, dim=1), (-1, 1)))

# Y = H.matmul(X) + W
# h = H.reshape(data_size, n_R*n_T)
# y = Y.reshape(data_size, n_R*T)
# print("Y:",Y)
# print("y:",y)
# print("y_:",y.reshape(data_size, n_R, n_T))

# x = torch.tensor([[1,2,3],[4,5,6]])
# y = torch.tensor([[7,8,9],[5,3,1]])
# # print(torch.stack([x,y],dim=1))
# for a,b in x,y:
#     print(a,b)

# print(torch.stack([Y.real,Y.imag,Y.abs()],dim=1))
# print(torch.stack([Y.real,Y.imag,Y.abs()],dim=1).reshape(num_trajectories,3,4,4))

# print(5e4)