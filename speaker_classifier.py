import torch.nn as nn
import torch

class speaker_classifier(nn.Module):
    
    def __init__(self, hparams) :
        super(speaker_classifier, self).__init__()
        self.model = nn.Sequential(nn.Linear(hparams.encoder_embedding_dim, hparams.hidden_sc_dim),
                                   nn.Linear(hparams.hidden_sc_dim, hparams.n_speakers))
    
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
        log probabilities of speaker classification = [batch_size, seq_len, n_speakers]
        '''
        out = self.model(encoder_outputs) 
        out = self.parse_outputs( out, text_lengths )
        return out
