
import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Conformer(nn.Module):
    def __init__(self, feature_size=108, num_layers=6, max_len=3000, in_channels=12, dropout=0.3):
        super(Conformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'

        self.src_mask = None

        self.inconv = nn.Sequential(nn.Conv1d(max_len, feature_size, 1),
                                        nn.LayerNorm((feature_size, in_channels)))
        self.pos_encoder = PositionalEncoding(feature_size, max_len)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=6, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)


    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.inconv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = torch.mean(output, dim=1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
