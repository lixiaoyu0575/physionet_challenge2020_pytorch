
import torch
from torch import nn
import math
import torch.nn.functional as F


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
    def __init__(self, num_classes, num_layers, in_channels, dropout):
        super(Conformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'

        self.src_mask = None

        self.conv1 = nn.Conv1d(in_channels, 128, 14, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(128, 256, 14, 3, 0, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(256, 256, 10, 2, 0, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv1d(256, 256, 10, 2, 0, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv1d(256, 256, 10, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv1d(256, 256, 10, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU(inplace=True)

        # self.inconv = nn.Sequential(nn.Conv1d(58, feature_size, 1),
        #                                 nn.LayerNorm((feature_size, 256)))
        self.pos_encoder = PositionalEncoding(58, 512)
        self.encoder_layer = TransformerEncoderLayer(d_model=58, nhead=2, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.conv1(src)
        src = self.bn1(src)
        src = self.relu1(src)

        src = self.conv2(src)
        src = self.bn2(src)
        src = self.relu2(src)

        src = self.conv3(src)
        src = self.bn3(src)
        src = self.relu3(src)

        src = self.conv4(src)
        src = self.bn4(src)
        src = self.relu4(src)

        src = self.conv5(src)
        src = self.bn5(src)
        src = self.relu5(src)

        src = self.conv6(src)
        src = self.bn6(src)
        src = self.relu6(src)

        # src = self.inconv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = F.adaptive_avg_pool1d(output, 1)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # output = torch.mean(output, dim=1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",in_channels=64,out_channels=108):
        super(TransformerEncoderLayer, self).__init__()
        try:
            from torch.nn import MultiheadAttention
        except:
            raise ImportError('MultiheadAttention module does not exist in PyTorch 1.1 or lower.')
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.conv = Convolution(d_model, in_channels,out_channels)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):

        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout1(src2)
        #
        # src2 = self.conv(src2)
        # src = src + self.dropout1(src2)
        # src = self.norm(src)

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)


        # src2 = self.conv(src2)
        # src = src + self.dropout(src2)

        src = self.norm(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm(src)

        return src

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Convolution(nn.Module):

    def __init__(self, d_model, in_channels,out_channels):
        super(Convolution, self).__init__()
        self.layerNorm = nn.LayerNorm(d_model)
        self.batchNorm = nn.BatchNorm1d(out_channels)
        self.glu = nn.GLU()
        self.swish = Act_op()
        self.depthWise_conv = nn.Conv1d(out_channels, out_channels, 54, stride= 1, padding=0, dilation=1,groups=out_channels,bias=True)
        self.pointWise_conv1 = nn.Conv1d(in_channels, out_channels, 1, stride= 1, padding=0, dilation=1,groups=1,bias=True)
        self.pointWise_conv2 = nn.Conv1d(out_channels, in_channels, 1, stride=1, padding=0, dilation=1,groups=1,bias=True)
    def forward(self, src):
        src = self.layerNorm(src)
        src = self.pointWise_conv1(src)
        src = self.glu(src)
        src = self.depthWise_conv(src)
        src = self.batchNorm(src)
        src = self.swish(src)
        src = self.pointWise_conv2(src)
        return src

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
