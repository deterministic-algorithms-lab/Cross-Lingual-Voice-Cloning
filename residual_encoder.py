import torch
import torch.nn as nn


class residual_encoder(nn.Module) :
    '''
    Neural network that can be used to parametrize q(z_{l}|x) and q(z_{o}|x)
    '''
    def __init__(self, hparams):
        super(residual_encoder, self).__init__()
        self.conv1 = nn.Conv1d(hparams.n_mel_channels, 512, 3, 1)
        self.bi_lstm = nn.LSTM(512, 256, 2, bidirectional = True, batch_first=True)
        self.linear = nn.Linear(512, 32)
        self.residual_encoding_dim = int(hparams.residual_encoding_dim/2)
        
    def forward(self, x):
        '''
        x.shape = [batch_size, seq_len, n_mel_channels]
        returns single sample from the distribution q(z_{l}|X) or q(z_{o}|X) of size [batch_size, 16]
        '''
        x = self.conv1(x.transpose(2,1)).transpose(2,1)
        output, (_,_) = self.bi_lstm(x)
        seq_len = output.shape[1]
        output = output.sum(dim=1)/seq_len
        x = self.linear(output)
        mean, log_variance = x[:,:self.residual_encoding_dim], x[:,self.residual_encoding_dim:]
        return torch.distributions.normal.Normal(mean, log_variance)    #Check here if scale_tril=log_variance ?
        
class continuous_given_discrete(nn.Module) :
    '''
    Class for p(z_{o}|y_{o}) and p(z_{l}|y_{l})
    n_disc :- number of discrete possible values for y_{o/l}
    distrib_lis[i] :- is the distribution over z , p(z|y=i). Total n_disc distribuitons
    distribs :- p(z|y) for all y. Can be used to sample n_disc z's {1 from each of the n_disc distribution of prev line}, simultaneously. 
    '''
    def __init__(self, hparams, n_disc) :
        super(continuous_given_discrete, self).__init__()
        self.n_disc = n_disc
        self.residual_encoding_dim  = int(hparams.residual_encoding_dim/2)

        self.cont_given_disc_mus    = nn.Parameter(torch.randn((self.n_disc, self.residual_encoding_dim)))
        self.cont_given_disc_sigmas = nn.Parameter(torch.ones((self.n_disc, self.residual_encoding_dim)))
        
        self.distrib_lis  = self.make_normal_distribs(self.cont_given_disc_mus, self.cont_given_disc_sigmas, make_lis=True)
        self.distribs     = self.make_normal_distribs(self.cont_given_disc_mus, self.cont_given_disc_sigmas, make_lis=False)

    def make_normal_distribs(self, mus, sigmas, make_lis = False) :
        if make_lis :
            return [torch.distributions.normal.Normal(mus[i], sigmas[i]) for i in range(mus.shape[0])]
        return torch.distributions.normal.Normal(mus, sigmas)
    
    def after_optim_step(self) :
        sigmas = self.cont_given_disc_sigmas.data 
        sigmas = sigmas.clamp(torch.exp(torch.tensor(-2.)))
        self.cont_given_disc_sigmas.data = sigmas

        self.cont_given_disc_mus.detach_()
        self.cont_given_disc_sigmas.detach_()
        
        self.cont_given_disc_mus.requires_grad=True
        self.cont_given_disc_sigmas.requires_grad=True
        
        self.distrib_lis  = self.make_normal_distribs(self.cont_given_disc_mus, self.cont_given_disc_sigmas, make_lis=True)
        self.distribs     = self.make_normal_distribs(self.cont_given_disc_mus, self.cont_given_disc_sigmas, make_lis=False)

    
class residual_encoders(nn.Module) :
    def __init__(self, hparams) :
        super(residual_encoders, self).__init__()
        
        #Variational Posteriors
        self.q_zl_given_X = residual_encoder(hparams)        #q(z_{l}|X)
        self.q_zo_given_X = residual_encoder(hparams)        #q(z_{o}|X)
        self.q_zl_given_X_at_x = None
        self.q_zo_given_X_at_x = None
        
        self.residual_encoding_dim = hparams.residual_encoding_dim
        self.mcn = hparams.mcn
        
        #Priors
        self.y_l_probs = torch.ones((hparams.dim_yl))
        self.y_l = torch.distributions.categorical.Categorical(self.y_l_probs)
        self.p_zo_given_yo = continuous_given_discrete(hparams, hparams.dim_yo)
        self.p_zl_given_yl = continuous_given_discrete(hparams, hparams.dim_yl)
        
        self.q_yl_given_X = None
    
    def calc_q_tilde(self, sampled_zl) :
        '''
        Caculates approximation to q_yl_given_X using monte carlo sampling, for each element in a batch.
        Supposed to be recalculated for each batch.
        '''
        K = self.p_zl_given_yl.n_disc
        sampled_zl = sampled_zl.repeat_interleave(K,-2)
        sampled_zl = sampled_zl.reshape(sampled_zl.shape[0], -1, K, sampled_zl.shape[-1])
        probs = self.p_zl_given_yl.distribs.log_prob(sampled_zl).exp()                        #[mcn, batch_size, K, residual_encoding_dim/2]
        p_zl_givn_yl = probs.prod(dim=-1)                                                     #[mcn, batch_size, K] 
        ans = p_zl_givn_yl*self.y_l.probs 
        normalization_consts = ans.sum(dim=-1)                                                #[mcn, batch_size]
        ans = ans.permute(2,0,1)/normalization_consts                                         #[K, mcn, batch_size]
        self.q_yl_given_X = ans.sum(dim=1)/self.mcn                                           #[K, batch_size]
                                                                                    
    def forward(self, x) :
        '''
        x.shape = [seq_len, batch_size, n_mel_channels]
        z_l.shape, z_o.shape == [hparams.mcn, batch_size, hparams.residual_encoding_dim/2]
        returns concatenation of z_{o} and z_{l} sampled from respective distributions
        '''
        x = x.transpose(1,0)
        self.q_zl_given_X_at_x, self.q_zo_given_X_at_x = self.q_zl_given_X(x), self.q_zo_given_X(x)         
        z_l, z_o = self.q_zl_given_X_at_x.rsample((self.mcn, )), self.q_zo_given_X_at_x.rsample((self.mcn,)) #[mcn, batch_size, residual_encoding_dim/2]
        self.calc_q_tilde(z_l)        
        return torch.cat([z_l,z_o], dim=-1).reshape(-1, self.residual_encoding_dim)
    
    def after_optim_step(self) :
        '''
        The parameters :- cont_given_disc_mus, sigmas, y_l_probs are altered, so their distributions need to be made again.
        '''
        self.y_l_probs.detach_()
        self.y_l_probs.requires_grad=True
        self.y_l = torch.distributions.categorical.Categorical(self.y_l_probs)
        self.p_zo_given_yo.after_optim_step()
        self.p_zl_given_yl.after_optim_step()
    
    def infer(self, y_o_idx, y_l_idx=None) :
        if y_l_idx is None :
            y_l_idx = self.y_l.sample()
        z_l = self.p_zl_given_yl.distrib_lis[y_l_idx].sample()
        z_o = self.p_zo_given_yo.distrib_lis[y_o_idx].sample()
        return torch.cat([z_l,z_o], dim=-1).unsqueeze(dim=0)
