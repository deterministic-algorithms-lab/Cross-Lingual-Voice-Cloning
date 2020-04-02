class residual_encoder(nn.Module) :
  '''
  Neural network that can be used to parametrize q(z_{l}|x) and q(z_{o}|x)
  '''
    def __init__(self, hparams):
        super(residual_encoder, self).__init__()
        self.conv1 = nn.Conv1d(hparams.n_mel_channels, 512, 3, 1)
        self.bi_lstm = nn.LSTM(512, 256, 2, bidirectional = True, batch_first=True)
        self.linear = nn.Linear(hparams.n_mel_channels, 32)
        self.epsilon = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(16), torch.eye(16))

    def forward(self, x):
        '''
        x.shape = [batch_size, seq_len, n_mel_channels]
        returns single sample from the distribution q(z_{l}|X) or q(z_{l}|X) of size [batch_size, 16]
        '''
        x = self.conv1(x)
        output, (_,_) = self.bi_lstm(x)
        seq_len = output.shape[1]
        output = output.sum(dim=1)/seq_len
        x = self.linear(x)
        mean, log_variance = x[:,:16], x[:,16:]
        return mean + log_variance*self.epsilon.sample((x.shape[1],))

class residual_encoders(nn.Module) :
    def __init__(self, hparams) :
        super(residual_encoders, self).__init__()
        self.latent_encoder = residual_encoder(hparams)         #q(z_{l}|X)
        self.observe_encoder = residual_encoder(hparams)        #q(z_{o}|X)
    
    def forward(self, x) :
        '''
        x.shape = [batch_size, seq_len, n_mel_channels]
        returns concatenation of z_{o} and z_{l} sampled from respective distributions
        '''
        z_l, z_o = self.latent_encoder(x), self.observe_encoder(x)
        return torch.cat([z_l,z_o], dim=1)