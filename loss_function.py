import torch
from torch import nn 
from torch.distributions.kl import kl_divergence as kld 
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class speaker_classifier(nn.Module):
    
    def __init__(self, hparams) :
        super(speaker_classifier, self).__init__()
        self.model = nn.Sequential(nn.Linear(hparams.encoder_embedding_dim, hparams.hidden_sc_dim),
                                   nn.Linear(hparams.hidden_sc_dim, hparams.n_speakers), 
                                   nn.Softmax(dim=-1))
    
    def parse_outputs(self, out, text_lengths) :
        mask = torch.arange(out.size(1), device=out.device).expand(out.size(0), out.size(1)) < text_lengths.unsqueeze(1)
        out = out.permute(2,0,1)
        out = out*mask
        out = out.permute(1,2,0)
        return out

    def forward(self, encoder_outputs, text_lengths) :
        '''
        input :-
        encoder_outputs = [batch_size, seq_len, encoder_embedding_size]
        text_lengths = [batch_size]
        
        output :-
        log probabilities of speaker classification
        '''
        out = torch.log( self.model(encoder_outputs) )
        out = self.parse_outputs( out, text_lengths )
        return out


class Tacotron2Loss(nn.Module):
    
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.speaker_classifier = speaker_classifier(hparams)

    def forward(self, model_output, targets, re, batched_speakers):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, encoder_outputs, text_lengths = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        
        means, stddevs = re.q_zo_given_X_at_x.mean, re.q_zo_given_X_at_x.stddev
        kl_loss = kld(re.p_zo_given_yo.distrib_lis[batched_speakers[0]], Normal(means[0], stddevs[0])).sum()
        for i, speaker in enumerate(batched_speakers[1:], 1) :
            kl_loss += kld(re.p_zo_given_yo.distrib_lis[speaker], Normal(means[i], stddevs[i])).sum()
        for i in range(re.p_zl_given_yl.n_disc) :
            kl_loss += ( re.q_yl_given_X[i]*kld(re.q_zl_given_X_at_x, re.p_zl_given_yl.distrib_lis[i]).sum(dim=1) ).sum()
        for i in range(re.q_yl_given_X.shape[1]) :
            kl_loss += kld( Categorical(re.q_yl_given_X[:,i]), re.y_l)

        speaker_log_probs = self.speaker_classifier(encoder_outputs, text_lengths)
        speaker_loss = torch.sum(speaker_log_probs)
        
        return (mel_loss + gate_loss) + 0.02*speaker_loss -kl_loss
