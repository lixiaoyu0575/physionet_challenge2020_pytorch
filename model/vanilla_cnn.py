import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class VanillaCNN(nn.Module):
    def __init__(self, in_channels=12, num_classes=108, channels=[32, 64, 128, 256, 512], kernel_size=[5, 5, 5, 5, 5], pool='max', pool_kernel_size=2, n_hid=[100, 80], drop_rate=None):
        super(VanillaCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_hid = n_hid

        # Convolution
        self.conv = nn.ModuleList()
        for i in range(len(channels)):
            if i == 0:
                self.conv.append(nn.Conv1d(in_channels, channels[0], kernel_size=kernel_size[0]))
            else:
                self.conv.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=kernel_size[i]))

        # Pooling
        if pool == 'max':
            self.pooling = nn.MaxPool1d(kernel_size=pool_kernel_size)
        elif pool == 'average':
            self.pooling = nn.AvgPool1d(kernel_size=pool_kernel_size)
        else:
            print("error!")
            exit(1)

        # Fully-connected Layer
        self.fc = nn.ModuleList()
        for i in range(len(n_hid)):
            if i == 0:
                self.fc.append(nn.Linear(channels[-1], n_hid[0]))
            else:
                self.fc.append(nn.Linear(n_hid[i-1], n_hid[i]))
        self.fc.append(nn.Linear(n_hid[-1], num_classes))

        # Batch Normalization
        self.batch_norm = nn.ModuleList()
        for i in range(len(channels)):
            self.batch_norm.append(nn.BatchNorm1d(channels[i]))

        # Dropout
        self.drop_rate = drop_rate

        # Activation
        self.act = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.conv)):
            x = self.pooling(self.act(self.batch_norm[i]((self.conv[i](x)))))
            if self.drop_rate:
                x = F.dropout(x, self.drop_rate)
        x = torch.mean(x, dim=2)

        for i in range(len(self.fc)-1):
            x = self.act(self.fc[i](x))
            if self.drop_rate:
                x = F.dropout(x, self.drop_rate)
        x = self.fc[-1](x)

        return x



