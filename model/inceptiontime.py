import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import torch.nn.init as init

def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes

def pass_through(X):
    return X

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1   = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class InceptionTimeTransposeModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 activation=nn.ReLU()):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        """
        super(InceptionTimeTransposeModule, self).__init__()
        self.activation = activation
        self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.conv_to_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv1d(
            in_channels=3 * bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X, indices):
        Z1 = self.conv_to_bottleneck_1(X)
        Z2 = self.conv_to_bottleneck_2(X)
        Z3 = self.conv_to_bottleneck_3(X)
        Z4 = self.conv_to_maxpool(X)

        Z = torch.cat([Z1, Z2, Z3], dim=1)
        MUP = self.max_unpool(Z4, indices)
        BN = self.bottleneck(Z)
        # another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution

        return self.activation(self.batch_norm(BN + MUP))

class InceptionTimeTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 use_residual=True, activation=nn.ReLU()):
        super(InceptionTimeTransposeBlock, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.inception_1 = InceptionTimeTransposeModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_2 = InceptionTimeTransposeModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_3 = InceptionTimeTransposeModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=out_channels
                )
            )

    def forward(self, X, indices):
        assert len(indices) == 3
        Z = self.inception_1(X, indices[2])
        Z = self.inception_2(Z, indices[1])
        Z = self.inception_3(Z, indices[0])
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        return Z

class InceptionTimeMoudule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, bottleneck=True, activation=nn.ReLU(), stride=1,
                 return_indices=False):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(InceptionTimeMoudule, self).__init__()
        self.return_indices = return_indices
        if in_channels > 1 and bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = in_channels

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=stride,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=stride,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=stride,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, X):
        # step 1
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3
        Z = torch.cat([Z1, Z2, Z3, Z4], dim=1)
        Z = self.activation(self.batch_norm(Z))
        if self.return_indices:
            return Z, indices
        else:
            return Z

class InceptionTimeBlockV1(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False):
        super(InceptionTimeBlockV1, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = InceptionTimeMoudule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            return_indices=return_indices
        )
        self.inception_2 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            return_indices=return_indices
        )
        self.inception_3 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_1(X)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
        else:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z

class InceptionTimeBlockV2(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, input_bottleneck=True, use_residual=True,
                 attention=None, activation=nn.ReLU(), return_indices=False):
        super(InceptionTimeBlockV2, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.attention = attention

        if input_bottleneck:
            self.inception_1 = InceptionTimeMoudule(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                activation=activation,
                stride=2,
                return_indices=return_indices
            )
        else:
            self.inception_1 = InceptionTimeMoudule(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                activation=activation,
                stride=2,
                return_indices=return_indices,
                bottleneck=False
            )
        self.inception_2 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            return_indices=return_indices
        )
        self.inception_3 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=2,
            return_indices=return_indices
        )
        self.inception_4 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            return_indices=return_indices
        )
        self.inception_5 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=2,
            return_indices=return_indices
        )
        self.inception_6 = InceptionTimeMoudule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=1,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=8,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )
        if self.attention:
            if self.attention == 'SENet':
                self.att_1 = SELayer(channel=4 * n_filters)
                self.att_2 = SELayer(channel=4 * n_filters)
                self.att_3 = SELayer(channel=4 * n_filters)
                self.att_4 = SELayer(channel=4 * n_filters)
                self.att_5 = SELayer(channel=4 * n_filters)
                self.att_6 = SELayer(channel=4 * n_filters)
            elif self.attention == 'CBAM_Channel':
                self.att_1 = ChannelAttention(in_planes=4 * n_filters)
                self.att_2 = ChannelAttention(in_planes=4 * n_filters)
                self.att_3 = ChannelAttention(in_planes=4 * n_filters)
                self.att_4 = ChannelAttention(in_planes=4 * n_filters)
                self.att_5 = ChannelAttention(in_planes=4 * n_filters)
                self.att_6 = ChannelAttention(in_planes=4 * n_filters)
            elif self.attention == 'CBAM_Spatial':
                self.att_1 = SpatialAttention()
                self.att_2 = SpatialAttention()
                self.att_3 = SpatialAttention()
                self.att_4 = SpatialAttention()
                self.att_5 = SpatialAttention()
                self.att_6 = SpatialAttention()
            elif self.attention == 'CBAM':
                self.att_1 = nn.Sequential(ChannelAttention(in_planes=4 * n_filters), SpatialAttention())
                self.att_2 = nn.Sequential(ChannelAttention(in_planes=4 * n_filters), SpatialAttention())
                self.att_3 = nn.Sequential(ChannelAttention(in_planes=4 * n_filters), SpatialAttention())
                self.att_4 = nn.Sequential(ChannelAttention(in_planes=4 * n_filters), SpatialAttention())
                self.att_5 = nn.Sequential(ChannelAttention(in_planes=4 * n_filters), SpatialAttention())
                self.att_6 = nn.Sequential(ChannelAttention(in_planes=4 * n_filters), SpatialAttention())
            else:
                print("Attention type is not correct!")
                exit(1)

    def forward(self, X):
        if self.return_indices:
            if self.attention:
                Z, i1 = self.inception_1(X)
                Z = self.att_1(Z)
                Z, i2 = self.inception_2(Z)
                Z = self.att_2(Z)
                Z, i3 = self.inception_3(Z)
                Z = self.att_3(Z)
                Z, i4 = self.inception_4(Z)
                Z = self.att_4(Z)
                Z, i5 = self.inception_5(Z)
                Z = self.att_5(Z)
                Z, i6 = self.inception_6(Z)
                Z = self.att_6(Z)
            else:
                Z, i1 = self.inception_1(X)
                Z, i2 = self.inception_2(Z)
                Z, i3 = self.inception_3(Z)
                Z, i4 = self.inception_4(Z)
                Z, i5 = self.inception_5(Z)
                Z, i6 = self.inception_6(Z)
        else:
            if self.attention:
                Z = self.inception_1(X)
                Z = self.att_1(Z)
                Z = self.inception_2(Z)
                Z = self.att_2(Z)
                Z = self.inception_3(Z)
                Z = self.att_3(Z)
                Z = self.inception_4(Z)
                Z = self.att_4(Z)
                Z = self.inception_5(Z)
                Z = self.att_5(Z)
                Z = self.inception_6(Z)
                Z = self.att_6(Z)
            else:
                Z = self.inception_1(X)
                Z = self.inception_2(Z)
                Z = self.inception_3(Z)
                Z = self.inception_4(Z)
                Z = self.inception_5(Z)
                Z = self.inception_6(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z

class InceptionTimeV1(nn.Module):
    def __init__(self, in_channels, num_classes, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param use_residual            If or not use residual connection
        : param activation				Activation function for output tensor (nn.ReLU()).
        : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(InceptionTimeV1, self).__init__()
        self.num_classes = num_classes
        self.return_indices = return_indices
        self.inception_block_1 = InceptionTimeBlockV1(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_2 = InceptionTimeBlockV1(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_3 = InceptionTimeBlockV1(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.fc = nn.Linear(in_features=4 * n_filters, out_features=num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_block_1(X)
            Z, i2 = self.inception_block_2(Z)
            Z, i3 = self.inception_block_3(Z)
        else:
            Z = self.inception_block_1(X)
            Z = self.inception_block_2(Z)
            Z = self.inception_block_3(Z)
        Z = torch.mean(Z, dim=2)
        Z = self.fc(Z)
        # Z = self.sigmoid(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z

class InceptionTimeV2(nn.Module):
    def __init__(self, in_channels, num_classes, n_blocks, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, input_bottleneck=True, use_residual=True,
                 attention='SENet', activation=nn.ReLU(), return_indices=False):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param use_residual            If or not use residual connection
        : param activation				Activation function for output tensor (nn.ReLU()).
        : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(InceptionTimeV2, self).__init__()
        self.num_classes = num_classes
        self.return_indices = return_indices
        self.blocks = nn.ModuleList()
        self.blocks.append(InceptionTimeBlockV2(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            input_bottleneck=input_bottleneck,
            use_residual=use_residual,
            attention=attention,
            activation=activation,
            return_indices=return_indices
        ))

        for i in range(n_blocks-1):
            self.blocks.append(InceptionTimeBlockV2(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            attention=attention,
            activation=activation,
            return_indices=return_indices
        ))
        # self.inception_block_1 = InceptionTimeBlockV2(
        #     in_channels=in_channels,
        #     n_filters=n_filters,
        #     kernel_sizes=kernel_sizes,
        #     bottleneck_channels=bottleneck_channels,
        #     input_bottleneck=input_bottleneck,
        #     use_residual=use_residual,
        #     activation=activation,
        #     return_indices=return_indices
        # )
        # self.inception_block_2 = InceptionTimeBlockV2(
        #     in_channels=4 * n_filters,
        #     n_filters=n_filters,
        #     kernel_sizes=kernel_sizes,
        #     bottleneck_channels=bottleneck_channels,
        #     use_residual=use_residual,
        #     activation=activation,
        #     return_indices=return_indices
        # )
        # self.inception_block_3 = InceptionTimeBlockV2(
        #     in_channels=4 * n_filters,
        #     n_filters=n_filters,
        #     kernel_sizes=kernel_sizes,
        #     bottleneck_channels=bottleneck_channels,
        #     use_residual=use_residual,
        #     activation=activation,
        #     return_indices=return_indices
        # )
        self.fc = nn.Linear(in_features=4 * n_filters, out_features=num_classes)
        # self.sigmoid = nn.Sigmoid()

    def weight_init(self, mode='kaiming'):
        if mode == 'kaiming':
            initializer = self.kaiming_init
        elif mode == 'normal':
            initializer = self.normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def normal_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, X):
       for i in range(len(self.blocks)):
            X = self.blocks[i](X)
       X = torch.mean(X, dim=2)
       X = self.fc(X)
       return X

    # def forward(self, X):
    #     if self.return_indices:
    #         Z, i1 = self.inception_block_1(X)
    #         Z, i2 = self.inception_block_2(Z)
    #         Z, i3 = self.inception_block_3(Z)
    #     else:
    #         Z = self.inception_block_1(X)
    #         Z = self.inception_block_2(Z)
    #         Z = self.inception_block_3(Z)
    #     Z = torch.mean(Z, dim=2)
    #     Z = self.fc(Z)
    #     # Z = self.sigmoid(Z)
    #     if self.return_indices:
    #         return Z, [i1, i2, i3]
    #     else:
    #         return Z

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 12, 18000)
    m = InceptionTimeV2(in_channels=12, num_classes=108, n_blocks=4, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, input_bottleneck=True, attention='CBAM_Channel')
    flops, params = get_model_complexity_info(m, (12, 18000), as_strings=True, print_per_layer_stat=True)
    print("%s |%s" % (flops, params))
    print(m)