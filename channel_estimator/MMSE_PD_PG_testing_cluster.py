import torch
from torch.distributions.normal import Normal

device = torch.device("cpu")
logit_net = torch.load('./simulation/result/lr0.0001_[288, 64, 32, 576]_ep500_mu5.0.pt').to(device)
# for para in logit_net.parameters():
#     print(para.size())

# n_T, n_R, T = [1, 1, 1]
# n_R, n_T, T = [4, 4, 4]
n_R, n_T, T = [4, 36, 36]  ###
H_mean = 5  ###
H_sigma, W_mean, W_sigma = [1, 0, 0.1]

data_size = 2

X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T, 2))).to(device) ###
# H = torch.view_as_complex(torch.zeros(n_R*data_size, n_T, 2)).to(device)
# H1 = torch.view_as_complex(torch.normal(1, H_sigma, size=(data_size, n_T, 2))).to(device)
# H2 = torch.view_as_complex(torch.normal(2, H_sigma, size=(data_size, n_T, 2))).to(device)
# H3 = torch.view_as_complex(torch.normal(3, H_sigma, size=(data_size, n_T, 2))).to(device)
# H4 = torch.view_as_complex(torch.normal(4, H_sigma, size=(data_size, n_T, 2))).to(device)
# H[::4,:] = H1; H[1::4,:] = H2; H[2::4,:] = H3; H[3::4,:] = H4

W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*data_size, T, 2))).to(device)
Y = H.matmul(X) + W
y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)
h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)

print("y:",y)
print("h:",h)
logits = logit_net(y)
h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
h0 = torch.zeros(data_size, n_R*n_T*2)+H_mean
print("h_hat:",h_hat)
print("mean:",logits[:,::2])
print("var:",logits[:,1::2])
# print(torch.norm(h_hat-h)**2/torch.norm(h)**2)
# print(torch.norm(h_hat-h)**2)
# print(torch.norm((h_hat-h)[:,0:64])**2/32)
print(torch.norm((h_hat-h))**2/(n_R*n_T))
print(torch.norm((h-y))**2/(n_R*n_T))