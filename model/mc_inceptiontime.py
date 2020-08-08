import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import torch.nn.init as init
import numpy as np

def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(X):
    return X


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
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), stride=1,
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
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

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
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False):
        super(InceptionTimeBlockV2, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = InceptionTimeMoudule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            stride=2,
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

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_1(X)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
            Z, i4 = self.inception_4(Z)
            Z, i5 = self.inception_5(Z)
            Z, i6 = self.inception_6(Z)
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
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
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
        super(InceptionTimeV2, self).__init__()
        self.return_indices = return_indices
        self.inception_block_1 = InceptionTimeBlockV2(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_2 = InceptionTimeBlockV2(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_block_3 = InceptionTimeBlockV2(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
            activation=activation,
            return_indices=return_indices
        )
        # self.fc = nn.Linear(in_features=4 * n_filters, out_features=num_classes)
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
        if self.return_indices:
            Z, i1 = self.inception_block_1(X)
            Z, i2 = self.inception_block_2(Z)
            Z, i3 = self.inception_block_3(Z)
        else:
            Z = self.inception_block_1(X)
            Z = self.inception_block_2(Z)
            Z = self.inception_block_3(Z)
        Z = torch.mean(Z, dim=2)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z

class MCInceptionTimeV2(nn.Module):
    def __init__(self,  num_classes=108,
                 groups=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]],
                 n_filters=[32,32,32,32,32,32,32,32,32,32,32,32],
                 kernel_sizes=[[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39],[9, 19, 39]],
                 bottleneck_channels=[32,32,32,32,32,32,32,32,32,32,32,32],
                 use_residual=True,
                 n_hid = [100, 100],
                 activation=nn.ReLU()
                 ):
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
        super(MCInceptionTimeV2, self).__init__()
        self.num_classes = num_classes
        self.groups = groups
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = bottleneck_channels
        self.use_residual = use_residual
        self.n_hid = n_hid
        self.inceptions = nn.ModuleList()
        self.mlp = nn.ModuleList()

        for i in range(len(groups)):
            self.inceptions.append(InceptionTimeV2(
                in_channels=len(groups[i]),
                n_filters=n_filters[i],
                kernel_sizes=kernel_sizes[i],
                bottleneck_channels=bottleneck_channels[i],
                use_residual=use_residual,
                activation=activation,
                return_indices=False
            ))
        fc1 = nn.Linear(np.array(n_filters).sum()*4, n_hid[0])
        self.mlp.append(fc1)
        for i in range(0, len(n_hid)-1):
            self.mlp.append(nn.Linear(n_hid[i], n_hid[i+1]))
        self.mlp.append(nn.Linear(n_hid[-1], num_classes))

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
        features = list()
        for i in range(len(self.groups)):
            features.append(self.inceptions[i](X[:, self.groups[i], :]))
        X = torch.cat(features, dim=1)
        for i in range(len(self.n_hid)+1):
            X = self.mlp[i](X)
        return X
