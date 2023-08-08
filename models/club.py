'''MI module
    Modified from: https://github.com/Linear95/CLUB
'''

import torch 
import torch.nn as nn


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :     provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())


    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar


    def forward(self, x_samples, y_samples): 

        mu, logvar = self.get_mu_logvar(x_samples.unsqueeze(1))
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples.unsqueeze(2))**2 /2./logvar.exp()  
        
        prediction_1 = mu
        y_samples_1 = y_samples.unsqueeze(2)

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples.unsqueeze(1))
        return (-(mu - y_samples.unsqueeze(2))**2 /logvar.exp()-logvar).mean()
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample_group(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_group, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples) # mu/logvar: (bs, y_dim)
        mu = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, mu.shape[-1]) # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        logvar = logvar.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1]) # (bs, T, y_dim) -> (bs*T, y_dim)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0) / 2 

    def mi_est(self, x_samples, y_samples): # x_samples: (bs, x_dim); y_samples: (bs, T, y_dim)
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        # log of conditional probability of positive sample pairs
        mu_exp1 = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1) # (bs, y_dim) -> (bs, T, y_dim)
        positive = - ((mu_exp1 - y_samples)**2).mean(dim=1) / logvar.exp() # mean along T
        negative = - ((mu_exp1 - y_samples[random_index])**2).mean(dim=1) / logvar.exp() # mean along T

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean() / 2
