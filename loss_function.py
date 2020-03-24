import torch
from torch import nn 

class speaker_classifier(nn.Module):
    
    def __init__(self, hparams) :
        self.model = nn.Sequential(nn.Linear(hparams.encoder_embedding_dim, hparams.hidden_sc_dim),\
                                   nn.Linear(hparams.hidden_sc_dim, n_speakers), \ 
                                   nn.Softmax(dim=-1))
    
    def parse_outputs(self, out, text_lengths) :
        mask = torch.arange(out.size(1)).expand(out.size(0), out.size(1)) < text_lengths.unsqueeze(1)
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
        out = torch.log( self.model(text_embedding) )
        out = parse_outputs( out, text_lengths )
        return out


class Tacotron2Loss(nn.Module):
    
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        speaker_classifier = speaker_classifier(hparams)

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, encoder_outputs, text_lengths = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        
        encoder_outputs.register_hook(lambda x : return -x)
        speaker_log_probs = speaker_classifier(encoder_outputs, text_lengths)
        speaker_loss = torch.sum(speaker_log_probs)
        return (mel_loss + gate_loss) + 0.02*speaker_loss
