import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(size, hidden_sizes=[64, 32], lr=1e-3, channel_information = [0, 1, 0, 0.1],
           num_epochs = 3, num_trajectories = 1e4, num_minibatch = 10, cuda = 1, snr = 0):
    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    n_T, n_R, T = size
    data_size = num_minibatch*num_trajectories
    lamb, t = torch.Tensor([1, 500]).to(device)              ###
    lr_l = 5e-3                                              ###
    lr_t = lr_l*5e4
    H_mean, H_sigma, W_Mean = channel_information
    test_size = 2000
    # SNR_dB = torch.arange(-10,10.1,5).to(device)
    SNR_dB = snr
    SNR_lin = 10**(SNR_dB/10.0)
    
    sqrt2 = 2**0.5
    ## generate communication data to train the parameterized policy
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    # SNR_lin = torch.Tensor([-10, 0, 10, 20]).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/sqrt2).to(device)
    S = H.matmul(X)
    Ps = S.norm()**2/torch.ones_like(S.real).norm()**2
    # Px = S.norm()**2/torch.ones_like(S.real).norm()**2
    Pn = Ps / SNR_lin
    print(torch.sqrt(Pn))
    # Pn = Pn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(data_size//len(Pn), n_R, T, 2)
    # W = torch.view_as_complex(torch.normal(W_mean, Pn)/sqrt2).to(device)
    W = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn), size=(data_size, n_R, T, 2))/(sqrt2)).to(device)
    Y = S + W
    h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)
    y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)

    ## generate testing data
    # Pn = Ps / SNR_lin
    # Pn = Pn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(test_size//len(Pn), n_R, T, 2)
    H_test = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(test_size, n_R, n_T, 2))/sqrt2).to(device)
    # W_test = torch.view_as_complex(torch.normal(W_mean, torch.sqrt(Pn))/sqrt2).to(device)
    W_test = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn), size=(test_size, n_R, T, 2))/(sqrt2)).to(device)
    Y_test = H_test.matmul(X) + W_test
    h_test = torch.view_as_real(H_test).reshape(test_size, n_R*n_T*2)
    y_test = torch.view_as_real(Y_test).reshape(test_size, n_R*T*2)
    h_test_norm = torch.norm((h_test).reshape(test_size, n_T*n_R*2), dim=1)**2

    
    ## generate the multilayer peceptron
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*2*n_R*n_T]).to(device)
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # iter_loss_M = torch.zeros(num_epochs*num_minibatch).to(device)
    iter_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    # testing_loss_M = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    t_rec = torch.zeros(num_epochs*num_minibatch).to(device)
    lamb_rec = torch.zeros(num_epochs*num_minibatch).to(device)
    mse = torch.zeros(num_epochs*num_minibatch).to(device)
    ## plot progress bar
    pbar = tqdm(total = num_epochs*num_minibatch)
    for i in range(num_epochs):
        for j in range(num_minibatch):
            itr = j+i*num_minibatch+1
            ## trajectories training data
            tau_y = y[j*num_trajectories:(j+1)*num_trajectories, :]
            tau_h = h[j*num_trajectories:(j+1)*num_trajectories, :]
            logits = logits_net(tau_y)
            tau_h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
            
            ## compute loss and update
            optimizer.zero_grad()
            ## 2-norm of h - h_hat square at each realization (num_trajectories by 1) 
            norm = torch.norm(torch.view_as_complex((tau_h - tau_h_hat).reshape(num_trajectories, n_T*n_R, 2)), dim=1)**2
            ## lambda update
            lamb = max(0, lamb + lr_l/itr*(norm.mean() - t)) 
            ## model parameters update with policy gradient
            batch_loss = lamb*(norm.unsqueeze(1)*Normal(logits[:, ::2], torch.exp(logits[:, 1::2])).log_prob(tau_h_hat)).mean() ###
            batch_loss.backward()
            optimizer.step()
            ## t update
            t -= lr_t/itr*(1-lamb)
            ## save the loss to plot
            # iter_loss_M[itr-1] = norm.mean() / (n_T*n_R)  ### nRnT
            iter_loss_N[itr-1] = (norm / torch.norm((tau_h).reshape(num_trajectories, n_T*n_R*2), dim=1)**2).mean()
            
            ## validation
            logits = logits_net(y_test)
            h_hat_test = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
            # testing_loss_M[itr-1] = (torch.norm((h_test - h_hat_test).reshape(test_size, n_T*n_R*2), dim=1)**2).mean() / (n_R*n_T)  ### nRnT
            testing_loss_N[itr-1] = torch.norm(h_test - h_hat_test)**2 / torch.norm(h_test)**2
            t_rec[itr-1] = t
            lamb_rec[itr-1] = lamb
            mse[itr-1] = norm.mean()
            # try mse
            pbar.set_description('NMSE:%s, MSE:%s, t:%s, l:%s' 
                    %(format(float(iter_loss_N[itr-1]), '.3f'),format(norm.mean(), '.1f'),format((t), '.3f'),format((lamb), '.3f')))
            # pbar.set_description('norm:%s, t:%s, l:%s' %(format(norm.mean(), '.1f'),format((t), '.3f'),format((lamb), '.3f')))
            pbar.update(1)
        # after 1 epoch, shuffle the dataset, using the same index to shuffle H and Y
        idx = torch.randperm(Y.shape[0])
        Y = Y[idx,:,:]
        H = H[idx,:,:]

    ## plot the loss
    epochs = range(1, num_epochs+1)
    plt.subplot(311)
    plt.plot(epochs, iter_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    plt.suptitle("MMSE based PD with PG channel estimator")
    plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, SNR:%s' 
                 %(n_R,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T], SNR_dB))
    plt.xlabel('epochs')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(312)
    plt.plot(epochs,t_rec.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='t')
    plt.plot(epochs,mse.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"),label='MSE')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(313)
    plt.plot(epochs,lamb_rec.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"),label='$\\lambda$')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.legend()

    plt.savefig('./simulation/result/lr%s_%s_ep%s_SNR_%s.png' 
                  %(lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T],num_epochs,SNR_dB))
    plt.close()

    # plt.plot(epochs, iter_loss_M.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training loss')
    # plt.plot(epochs, testing_loss_M.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')
    # plt.suptitle("MMSE based PD with PG channel estimator")
    # plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, SNR:%s' 
    #              %(n_R,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T], snr))
    # plt.xlabel('epochs')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('./simulation/result/M_lr%s_%s_ep%s_SNR%s.png' %(lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T],num_epochs,snr))

    ## save the model
    torch.save(logits_net, './simulation/result/lr%s_%s_ep%s_SNR%s.pt' %(lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T],num_epochs,SNR_dB))
    torch.save(iter_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), 
               './simulation/result/lr%s_%s_ep%s_SNR%s_loss.pt' %(lr, [2*n_R*T]+hidden_sizes+[2*2*n_R*n_T],num_epochs,SNR_dB))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs', type=float, default=1) # sigma of H
    parser.add_argument('--hm', type=float, default=0) # mean of H
    parser.add_argument('--lr', type=float, default=1e-5) # learning rate
    parser.add_argument('--ep', type=int, default=10)    # num of epochs
    parser.add_argument('--tau', type=int, default=4)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=100)  # 100 number of mini-batch
    parser.add_argument('--nR', type=int, default=4)  
    parser.add_argument('--nT', type=int, default=36)  
    parser.add_argument('--hsz', type=int, default=1, nargs='+')  # hidden layer size
    parser.add_argument('--cuda', type=int, default=0)  # cuda
    parser.add_argument('--snr', type=int, default=0)  # cuda
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available())
    print(args.hsz)

    if args.hsz == 1:
        hidden_sizes = [64, 32]
    else:
        hidden_sizes = args.hsz

    T = args.nT*1
    # H_mean = 0
    W_mean = 0
    channel_information = [args.hm, args.hs, W_mean]

    train([args.nT, args.nR, T], hidden_sizes, args.lr, channel_information, args.ep, 10**args.tau, args.nmb, args.cuda, args.snr)
