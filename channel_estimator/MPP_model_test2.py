import torch
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import LMMSE

device = torch.device("cpu")
# logits_net = torch.load('./simulation/model/lr5e-06_[288, 2048, 2048, 576]_ep6000_SNR0.pt').to(device)
# logits_net = torch.load('./simulation/model/lr5e-06_[288, 2048, 2048, 576]_ep8000_SNR0.pt').to(device)

filename = 'lr5e-06_[288, 4096, 576]_ep6000_SNR[-10, -5, 5, 10].pt'

filename1 = 'lr1e-05_[64, 1024, 128]_ep1000_SNR[10].pt'
checkpoint1 = torch.load('./simulation/result/model/'+filename1)
filename2 = 'lr1e-05_[64, 1024, 128]_ep1000_SNR[-10].pt'
checkpoint2 = torch.load('./simulation/result/model/'+filename2)
filename3 = 'lr1e-05_[64, 1024, 1024, 1024, 1024, 128]_ep400_SNR[-10, -5, 5, 10].pt'
checkpoint3 = torch.load('./simulation/result/model/'+filename3)

logits_net1 = checkpoint1['logits_net'].to(device)
logits_net2 = checkpoint2['logits_net'].to(device)
logits_net3 = checkpoint3['logits_net'].to(device)
n_R, n_T, T = [checkpoint1['n_R'], checkpoint1['n_T'], checkpoint1['T']]
H_mean, H_sigma, W_mean = [0, 1, 0]

data_size = 2000

sqrt2 = 2**.5
SNR_dB = torch.arange(-10,10.1,5)
# SNR_dB = torch.Tensor([0])
SNR = 10**(SNR_dB/10.0)
NMSE_1 = torch.zeros_like(SNR)
NMSE_2 = torch.zeros_like(SNR)
NMSE_3 = torch.zeros_like(SNR)
NMSE2 = torch.zeros_like(SNR)
NMSE3 = torch.zeros_like(SNR)
NMSE4 = torch.zeros_like(SNR)

# print(logits_net.parameters())

pbar = tqdm(total = len(SNR_dB))
for idx, snr in enumerate(SNR):
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/(sqrt2)).to(device)
    S = H.matmul(X)
    Ps = S.norm()**2/torch.ones_like(S.real).norm()**2
    Pn = Ps / snr
    W = torch.view_as_complex(torch.normal(W_mean, torch.sqrt(Pn), size=(data_size, n_R, T, 2))/(sqrt2)).to(device) # torch.sqrt(var_n*2)/2
    
    Y = S + W
    y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)
    h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)

    logits1 = logits_net1(y)
    h_hat1 = Normal(logits1[:,::2], torch.exp(logits1[:,1::2])).sample()
    norm_1 = torch.norm((h - h_hat1).reshape(data_size, n_T*n_R*2), dim=1)**2
    logits2 = logits_net2(y)
    h_hat2 = Normal(logits2[:,::2], torch.exp(logits2[:,1::2])).sample()
    norm_2 = torch.norm((h - h_hat2).reshape(data_size, n_T*n_R*2), dim=1)**2
    logits3 = logits_net3(y)
    h_hat3 = Normal(logits3[:,::2], torch.exp(logits3[:,1::2])).sample()
    norm_3 = torch.norm((h - h_hat3).reshape(data_size, n_T*n_R*2), dim=1)**2

    norm2 = torch.norm((h - y).reshape(data_size, n_T*n_R*2), dim=1)**2
    norm3 = torch.norm((h - H_sigma**2/(H_sigma**2+Pn)*y).reshape(data_size, n_T*n_R*2), dim=1)**2
    # sample_Chh = H.reshape(data_size, n_R*n_T).T.matmul(torch.conj(H.reshape(data_size, n_R*n_T)))
    # # print(sample_Chh)
    # sample_LMMSE = sample_Chh.matmul((sample_Chh + Pn*torch.eye(n_R*n_T)).inverse()).matmul(Y.reshape(data_size, n_R*n_T).T).T
    # lmmse = torch.zeros_like(H)
    ## calculate h hat lmmse col by col
    # for i in range(n_T):
    #     Y_i = Y[:,:,i].T
    #     H_i = H[:,:,i].T
    #     A = torch.complex(torch.eye(n_R), torch.zeros(n_R, n_R)).to(device)
    #     # print(H_i.shape)
    #     lmmse[:,:,i] = LMMSE.LMMSE_solver(Y_i, A, H_i, Pn*torch.eye(n_R), data_size).T
    #     pbar.set_description('NMSE:%s' %(format(float(NMSE[idx]), '.3f')))
    #     pbar.update(1)
    X_tild = torch.complex(torch.eye(n_R*n_T), torch.zeros(n_R*n_T, n_R*n_T)).to(device)
    lmmse = LMMSE.LMMSE_solver(Y.reshape(data_size, n_R*T).T, X_tild, H.reshape(data_size, n_R*n_T).T, Pn*torch.eye(n_R*T), data_size).T
    norm4 = torch.norm((h - torch.view_as_real(lmmse).reshape(data_size, n_R*n_T*2)), dim=1)**2
    NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm((h), dim=1)**2).mean())
    NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm((h), dim=1)**2).mean())
    NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm((h), dim=1)**2).mean())

    NMSE2[idx] = 10*torch.log10((norm2 / torch.norm((h), dim=1)**2).mean())
    NMSE3[idx] = 10*torch.log10((norm3 / torch.norm((h), dim=1)**2).mean())
    NMSE4[idx] = 10*torch.log10((norm4 / torch.norm((h), dim=1)**2).mean())
    
    # plt.text(10*torch.log10(snr)-1,NMSE3[idx]-1, f'({10**(NMSE3[idx].item()/10):.2f})')
    pbar.update(1)
    
    

plt.plot(SNR_dB, NMSE_1,'-o', label='1 layer MLP trained w/ SNR:10')
plt.plot(SNR_dB, NMSE_2,'-o', label='1 layer MLP trained w/ SNR:-10')
plt.plot(SNR_dB, NMSE_3,'-o', label='4 layers MLP w/ SNR:-10,-5,5,10')

plt.plot(SNR_dB, NMSE2,'-o', label='LS')
plt.plot(SNR_dB, NMSE3,'-o', label='ideal LMMSE')
plt.plot(SNR_dB, NMSE4,'-o', label='sample LMMSE')
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
plt.suptitle("NMSE of primal dual DRL MIMO chest vs SNR")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title(' MLP trained with size:%s, lr:%s SNR:%s' %([2*n_R*T]+checkpoint['hidden_sizes']+[2*2*n_R*n_T], checkpoint['lr'], checkpoint['SNR_dB'].tolist()))
plt.xlabel('SNR(dB)')
plt.ylabel('NMSE(dB)')
plt.legend()
plt.grid(True)
plt.savefig('./simulation/result/snr/3mlp_Chest_.png')


# plt.savefig('./simulation/result/snr/dire_Y_snr[-50,25].png' )
# plt.close()

# print(b.dtype)
# print(SNRdB)
# print(torch.view_as_complex(torch.normal(0, 1, size=(data_size, n_R, T, 2))))

# print("y:",y)
# print("h:",h)
# logits = logits_net(y)
# h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
# h0 = torch.zeros(data_size, n_R*n_T*2)+H_mean
# print("h_hat:",h_hat)
# print("mean:",logits[:,::2])
# print("var:",logits[:,1::2])
# # print(torch.norm(h_hat-h)**2/torch.norm(h)**2)
# # print(torch.norm(h_hat-h)**2)
# # print(torch.norm((h_hat-h)[:,0:64])**2/32)
# print(torch.norm((h_hat-h))**2/(n_R*n_T))
# print(torch.norm((h-y))**2/(n_R*n_T))