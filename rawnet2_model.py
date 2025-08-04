import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottle2neck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class RawNet2(nn.Module):
    def __init__(self):
        super(RawNet2, self).__init__()
        self.first_conv = nn.Conv1d(1, 80, kernel_size=3, stride=1, padding=1)
        self.first_bn = nn.BatchNorm1d(80)

        # Define residual blocks using nn.Sequential so key names match
        self.block0 = nn.Sequential(Bottle2neck(80, 80, downsample=nn.Conv1d(80, 80, 1)))
        self.block1 = nn.Sequential(Bottle2neck(80, 80, downsample=nn.Conv1d(80, 80, 1)))
        self.block2 = nn.Sequential(Bottle2neck(80, 80, downsample=nn.Conv1d(80, 80, 1)))
        self.block3 = nn.Sequential(Bottle2neck(80, 80, downsample=nn.Conv1d(80, 80, 1)))
        self.block4 = nn.Sequential(Bottle2neck(80, 80, downsample=nn.Conv1d(80, 80, 1)))
        self.block5 = nn.Sequential(Bottle2neck(80, 80, downsample=nn.Conv1d(80, 80, 1)))

        self.fc_attention0 = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        self.fc_attention1 = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        self.fc_attention2 = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        self.fc_attention3 = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        self.fc_attention4 = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        self.fc_attention5 = nn.Sequential(nn.Linear(3, 3), nn.ReLU())

        self.bn_before_gru = nn.BatchNorm1d(3)

        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1_gru = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = F.relu(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = x.permute(0, 2, 1)

        x, _ = self.gru(x)
        x = x[:, -1, :]

        x = F.relu(self.fc1_gru(x))
        x = torch.sigmoid(self.fc2(x))
        return x
