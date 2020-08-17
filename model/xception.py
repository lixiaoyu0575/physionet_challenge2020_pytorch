from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class SeparableConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv1d,self).__init__()

        self.conv1 = nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv1d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
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

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

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


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, in_channels=12, num_classes=108):
        """ Constructor
        Args:
            in_channels: channels of input data
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv1d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm1d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv1d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm1d(2048)

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

        x = self.block1(x)
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
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
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
    def __init__(self,in_filters,out_filters,reps,strides=1,groups=[32,32],group_size=[4,4],start_with_relu=True,grow_first=True):
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

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

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

        self.block1=GceptionBlock(64,128,2,2,groups=[32,32],group_size=[4,4],start_with_relu=False,grow_first=True)
        self.block2=GceptionBlock(128,256,2,2,groups=[32,32],group_size=[4,4],start_with_relu=True,grow_first=True)
        self.block3=GceptionBlock(256,512,2,2,groups=[32,32],group_size=[4,4],start_with_relu=True,grow_first=True)

        self.block4=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with_relu=True,grow_first=True)
        self.block5=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with_relu=True,grow_first=True)
        self.block6=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with_relu=True,grow_first=True)
        self.block7=GceptionBlock(512,512,3,1,groups=[32,32],group_size=[8,8],start_with_relu=True,grow_first=True)

        self.block8=GceptionBlock(512,768,2,2,groups=[64,64],group_size=[8,8],start_with_relu=True,grow_first=False)

        self.conv3 = GceptionBlock(768,1024,3,1,groups=[64,64],group_size=[8,8],start_with_relu=True,grow_first=True)
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