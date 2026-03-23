import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, 128)
        self.bn_lin1 = nn.BatchNorm1d(128)

        self.lin2 = nn.Linear(128, 1024)
        self.bn_lin2 = nn.BatchNorm1d(1024)

        self.tconv1 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.tconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.tconv3 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.tconv4 = nn.ConvTranspose2d(32, 1, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = x.view((x.shape[0], -1))
        x = F.relu(self.bn_lin1(self.lin1(x)))
        x = F.relu(self.bn_lin2(self.lin2(x)))

        x = x.view((x.shape[0], 64, 4, 4))

        x = F.leaky_relu(self.bn1(self.tconv1(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.tconv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.tconv3(x)), 0.1, inplace=True)

        img = torch.sigmoid(self.tconv4(x))

        return img






class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv2d(1, 32, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(32, 32, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1, bias=False))
        self.lin1 = spectral_norm(nn.Linear(1024, 128))


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv4(x), 0.1, inplace=True)

        x = x.view((x.shape[0], -1))

        x = F.leaky_relu(self.lin1(x), 0.1, inplace=True)

        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Conv2d(1024, 1, 1)
        self.lin = spectral_norm(nn.Linear(128, 1))

    def forward(self, x):
        # x = x.view((x.shape[0], -1, 1, 1))
        # output = torch.sigmoid(self.conv(x))
        output = torch.sigmoid(self.lin(x))

        return output



class QHead(nn.Module):
    def __init__(self, dis_n, cont_n):
        super().__init__()

        self.lin1 = spectral_norm(nn.Linear(128, 128))
        # self.bn1 = nn.BatchNorm1d(128)

        self.lin_disc = nn.Linear(128, dis_n)
        self.lin_mu = nn.Linear(128, cont_n)
        self.lin_var = nn.Linear(128, cont_n)

    def forward(self, x):
        x = F.leaky_relu(self.lin1(x), 0.1, inplace=True)

        disc_logits = self.lin_disc(x).squeeze()
        mu = self.lin_mu(x).squeeze()
        var = torch.exp(self.lin_var(x).squeeze())

        return disc_logits, mu, var


class TC_Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(TC_Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()








def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)





