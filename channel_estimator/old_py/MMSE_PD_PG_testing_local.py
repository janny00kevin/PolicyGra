import torch
from torch.distributions.normal import Normal

device = torch.device("cpu")
logit_net = torch.load('./simulation/result/lr0.0001_[32, 64, 32, 64]_ep500.pt').to(device)
# for para in logit_net.parameters():
#     print(para.size())

n_T, n_R, T = [4, 4, 4]
H_mean, H_sigma, W_mean, W_sigma = [0, 1, 0, 0.1]

data_size = 1

X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(n_R*data_size, n_T, 2))).to(device)
W = torch.view_as_complex(torch.normal(W_mean, W_sigma, size=(n_R*data_size, T, 2))).to(device)
Y = H.matmul(X) + W
y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)

print(y)
logits = logit_net(y)
h_hat = Normal(logits[:,::2], logits[:,1::2]).sample()
print(h_hat)
print(y.size(), h_hat.size())
print(logits[:,::2])
print(logits[:,1::2])