import torch
import matplotlib.pyplot as plt


file_path = './result/240527_adam_spLMMSE/lr1e-06_[288, 4096, 576]_ep6000_SNR[-10,  -5,   5,  10].pt'
# loss = torch.load('./result/240527_adam_spLMMSE/lr5e-06_[288, 2048, 2048, 2048, 576]_ep5800_SNR0_loss.pt')
checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
loss = checkpoint['iter_loss_N'].to('cpu')

# print(len(loss))
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label='training loss')
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, SNR:%s' 
#                 %(n_R,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T], snr))
for i in range(0, len(epochs), 1000):
    plt.text(i,loss[i],f'({loss[i]:.2f})')
plt.text(len(loss),loss[len(loss)-1],f'({loss[len(loss)-1]:.2f})')
plt.xlabel('epochs')
plt.ylabel('NMSE')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./simulation/result/N_lr%s_%s_ep%s_SNR:%s.png' 
#                 %(lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T],num_epochs,SNR_dB))
# plt.close()