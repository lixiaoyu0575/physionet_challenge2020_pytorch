from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
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

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
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

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src2,attn_output_weights = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SeparableConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv1d,self).__init__()

        self.conv1 = nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv1d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.ca = ChannelAttention(in_channels, ratio= 16)
        self.sa = SpatialAttention()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self,x):
        x = self.conv1(x)
        # if self.in_channels == self.out_channels:
        x = self.ca(x) * x
        x = self.pointwise(x)
        # x = self.sa(x)*x
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides,start_with,grow_first,ratio):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv1d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm1d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv1d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm1d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv1d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm1d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv1d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm1d(out_filters))

        if not start_with:
            rep = rep[1:]
        elif start_with=='Relu':
            rep[0] = nn.ReLU(inplace=False)
        else:
            rep[0] = Act_op()

        if strides != 1:
            rep.append(nn.MaxPool1d(3,strides,1))
        self.rep = nn.Sequential(*rep)


    def forward(self,inp):

        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel: int, ratio: int):
        super(ChannelAttention, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Conv1d(channel, channel // ratio, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // ratio, channel, 1, padding=0, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_avg = self.shared_mlp(self.avg_pool(x))
        feat_max = self.shared_mlp(self.max_pool(x))

        return self.sigmoid(feat_avg + feat_max)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv1d(2, 1, 7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_avg = torch.mean(x, dim=1, keepdim=True)
        feat_max = torch.max(x, dim=1, keepdim=True)[0]

        feature = torch.cat((feat_avg, feat_max), dim=1)

        return self.sigmoid(self.conv(feature))

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, in_channels=12, num_classes=108,src_mask=None):
        """ Constructor
        Args:
            in_channels: channels of input data
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        try:
            from torch.nn import TransformerEncoder
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.src_mask = src_mask

        self.conv1 = nn.Conv1d(in_channels,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with=False,grow_first=True,ratio = 16)
        self.block2=Block(128,256,2,2,start_with='Relu',grow_first=True,ratio = 16)
        self.block3=Block(256,728,2,2,start_with='Relu',grow_first=True,ratio = 16)

        self.block4=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)
        self.block5=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)
        self.block6=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)
        self.block7=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)

        self.block8=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)
        self.block9=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)
        self.block10=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)
        self.block11=Block(728,728,3,1,start_with='Relu',grow_first=True,ratio = 16)

        self.block12=Block(728,1024,2,1,start_with='Relu',grow_first=False,ratio = 16)

        self.conv3 = SeparableConv1d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm1d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv1d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm1d(2048)

        # # Implementation of attention model
        self.pos_encoder = PositionalEncoding(188, 256)
        self.encoder_layer = TransformerEncoderLayer(d_model=188, nhead=4, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=2)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def blocks(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

    def attention(self, features):
        x = nn.ReLU(inplace=True)(features)
        output = self.pos_encoder(x)
        output = self.transformer_encoder(output, self.src_mask)
        return output

    def forward(self, input):
        x = self.features(input)
        x = self.blocks(x)
        att = self.attention(x)
        x = x + att
        x = self.logits(x)
        x = self.fc(x)
        return x


class GroupConv1d(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False, groups=32):
        super(GroupConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class GceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 5, 7],stride=1,padding=0,dilation=1,bias=False, groups=32, group_size=4):
        super(GceptionModule, self).__init__()

        inner_channels = group_size * groups
        self.conv1 = nn.Conv1d(in_channels, inner_channels, 1,1,0,1, bias=False)
        self.conv2 = GroupConv1d(inner_channels, inner_channels, kernel_size[0], stride, padding, dilation, bias, groups=groups)
        self.conv3 = GroupConv1d(inner_channels, inner_channels, kernel_size[1], stride, padding, dilation, bias, groups=groups)
        self.conv4 = GroupConv1d(inner_channels, inner_channels, kernel_size[2], stride, padding, dilation, bias, groups=groups)
        self.conv5 = nn.Conv1d(3 * inner_channels, out_channels, 1,1,0,1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class GceptionBlock(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,groups=[32,32],group_size=[4,4],start_with=True,grow_first=True):
        super(GceptionBlock, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv1d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm1d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(GceptionModule(in_filters,out_filters,kernel_size=[3,5,7],stride=1,padding=1,bias=False,groups=groups[0],group_size=group_size[0]))
            rep.append(nn.BatchNorm1d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(GceptionModule(filters,filters,kernel_size=[3,5,7],stride=1,padding=1,bias=False,groups=groups[1],group_size=group_size[1]))
            rep.append(nn.BatchNorm1d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(GceptionModule(in_filters,out_filters,kernel_size=[3,5,7],stride=1,padding=1,bias=False,groups=groups[0],group_size=group_size[0]))
            rep.append(nn.BatchNorm1d(out_filters))

        if not start_with:
            rep = rep[1:]
        elif start_with == 'Relu':
            rep[0] = nn.ReLU(inplace=False)
        else:
            rep[0] = Act_op()

        if strides != 1:
            rep.append(nn.MaxPool1d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Gception(nn.Module):

    def __init__(self, in_channels=12, num_classes=108):
        """ Constructor
        Args:
            in_channels: channels of input data
            num_classes: number of classes
        """
        super(Gception, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=GceptionBlock(64,128,2,2,groups=[32,32],group_size=[4,4],start_with=False,grow_first=True)
        self.block2=GceptionBlock(128,256,2,2,groups=[32,32],group_size=[4,4],start_with='Relu',grow_first=True)
        self.block3=GceptionBlock(256,512,2,2,groups=[32,32],group_size=[4,4],start_with='Relu',grow_first=True)

        self.block4=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with='Swish',grow_first=True)
        self.block5=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with='Swish',grow_first=True)
        self.block6=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with='Swish',grow_first=True)
        self.block7=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with='Swish',grow_first=True)

        self.block8=GceptionBlock(512,768,2,2,groups=[64,64],group_size=[8,8],start_with='Swish',grow_first=False)

        self.conv3 = GceptionBlock(768,1024,3,1,groups=[64,64],group_size=[8,8],start_with='Swish',grow_first=True)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc = nn.Linear(1024, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------
    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x