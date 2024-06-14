import torch
from torch.distributions.normal import Normal

device = torch.device("cpu")
logit_net = torch.load('./simulation/model/Rlr1e-05_[144, 1024, 1024, 288]_ep1000_mu0.pt').to(device)
# for para in logit_net.parameters():
#     print(para.size())

# n_T, n_R, T = [1, 1, 1]
# n_R, n_T, T = [4, 4, 4]
n_R, n_T, T = [4, 36, 36]  ###
H_mean = 0  ###
H_sigma, W_mean, W_sigma = [1, 0, 0.1]

data_size = 100

X = torch.eye(n_T).tile(T//n_T).to(device)
H = torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T)).to(device)
# H = torch.view_as_complex(torch.zeros(n_R*data_size, n_T, 2)).to(device)
# H1 = torch.view_as_complex(torch.normal(1, H_sigma, size=(data_size, n_T, 2))).to(device)
# H2 = torch.view_as_complex(torch.normal(2, H_sigma, size=(data_size, n_T, 2))).to(device)
# H3 = torch.view_as_complex(torch.normal(3, H_sigma, size=(data_size, n_T, 2))).to(device)
# H4 = torch.view_as_complex(torch.normal(4, H_sigma, size=(data_size, n_T, 2))).to(device)
# H[::4,:] = H1; H[1::4,:] = H2; H[2::4,:] = H3; H[3::4,:] = H4

W = torch.normal(W_mean, W_sigma, size=(n_R*data_size, T)).to(device)
Y = H.matmul(X) + W
y = Y.reshape(data_size, n_R*T)
h = H.reshape(data_size, n_R*n_T)
# h_ = h.reshape(n_R*data_size, n_T, 2)
# print(H_ == h_)

print("y:",y[0,0:10])
print("h:",h[0,0:10])
logits = logit_net(y)
h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
h0 = torch.zeros(data_size, n_R*n_T*2)+H_mean
print("h_hat:",h_hat[0,0:10])
print("mean:",logits[0,:20:2])
print("var:",logits[0,1:20:2])
print((torch.norm((h_hat-h).reshape(data_size,n_R*n_T),dim=1)**2).mean()/(n_R*n_T))
print((torch.norm((h-y).reshape(data_size,n_R*n_T),dim=1)**2).mean()/(n_R*n_T))