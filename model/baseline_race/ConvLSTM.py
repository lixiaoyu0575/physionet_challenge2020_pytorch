import torch.nn as nn

class ConvLSTM(nn.Module):

    def __init__(self, num_classes):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=4, padding=3,bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64,128,kernel_size=7, stride=3,padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=7, stride=2, padding=0)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 1024, kernel_size=7, stride=2, padding=0)
        self.bn5 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.lstm = nn.LSTM(12, 128, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(131072, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

