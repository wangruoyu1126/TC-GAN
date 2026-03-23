import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



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

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)

        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.lin1 = nn.Linear(1024, 128)
        self.bn_lin1 = nn.BatchNorm1d(128)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)

        x = x.view((x.shape[0], -1))

        x = F.leaky_relu(self.bn_lin1(self.lin1(x)), 0.1, inplace=True)

        return x



class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Conv2d(1024, 1, 1)
        self.lin = nn.Linear(128, 1)

    def forward(self, x):
        # x = x.view((x.shape[0], -1, 1, 1))
        # output = torch.sigmoid(self.conv(x))
        output = torch.sigmoid(self.lin(x))

        return output



class QHead(nn.Module):
    def __init__(self, dis_n, cont_n):
        super().__init__()

        self.lin1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.lin_disc = nn.Linear(128, dis_n)
        self.lin_mu = nn.Linear(128, cont_n)
        self.lin_var = nn.Linear(128, cont_n)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.lin1(x)), 0.1, inplace=True)

        disc_logits = self.lin_disc(x).squeeze()
        mu = self.lin_mu(x).squeeze()
        var = torch.exp(self.lin_var(x).squeeze())

        return disc_logits, mu, var




# class QHead(nn.Module):
#     def __init__(self, dis_n, cont_n):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(128)
#
#         self.conv_disc = nn.Conv2d(128, dis_n, 1)
#         self.conv_mu = nn.Conv2d(128, cont_n, 1)
#         self.conv_var = nn.Conv2d(128, cont_n, 1)
#
#     def forward(self, x):
#         x = x.view((x.shape[0], -1, 1, 1))
#
#         x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
#
#         disc_logits = self.conv_disc(x).squeeze()
#
#         mu = self.conv_mu(x).squeeze()
#         var = torch.exp(self.conv_var(x).squeeze())
#
#         return disc_logits, mu, var







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





