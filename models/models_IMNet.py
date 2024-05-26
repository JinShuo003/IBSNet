from torch import nn
from datetime import datetime
import torch.nn.functional as F

from models.pn2_utils import *
from models.models_utils import *


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.relu1(x))
        dx = self.fc_1(self.relu2(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetPointnet(nn.Module):
    """ PointNet-based encoder model with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the model
    """

    def __init__(self, c_dim=256, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        # batch_size, T, D = p.size()
        # output size: B x T X F

        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class IM_Decoder(nn.Module):
    """l2+leakyRelu+noNorm+noDrop+all+changeDim-6"""

    def __init__(self, input_dim):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building IM-decoder.')
        super().__init__()

        # self.linear_0 = nn.Linear(input_dim, 2048, bias=True)
        # self.linear_1 = nn.Linear(input_dim + 2048, 1024, bias=True)
        # self.linear_2 = nn.Linear(input_dim + 1024, 512, bias=True)
        # self.linear_3 = nn.Linear(input_dim + 512, 256, bias=True)
        # self.linear_4 = nn.Linear(input_dim + 256, 128, bias=True)
        # self.linear_5 = nn.Linear(128, 2, bias=True)

        self.linear_0 = nn.Linear(input_dim, 512, bias=True)
        self.linear_1 = nn.Linear(input_dim + 512, 512, bias=True)
        self.linear_2 = nn.Linear(input_dim + 512, 512, bias=True)
        self.linear_3 = nn.Linear(input_dim + 512, 512, bias=True)
        self.linear_4 = nn.Linear(input_dim + 512, 512, bias=True)
        self.linear_5 = nn.Linear(512, 1, bias=True)

        num_params = sum(p.data.nelement() for p in self.parameters())

    def forward(self, batch_input):
        l0 = self.linear_0(batch_input)
        l0 = F.leaky_relu(l0, negative_slope=0.02, inplace=True)
        l0 = torch.cat([l0, batch_input], dim=-1)

        l1 = self.linear_1(l0)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)
        l1 = torch.cat([l1, batch_input], dim=-1)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)
        l2 = torch.cat([l2, batch_input], dim=-1)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)
        l3 = torch.cat([l3, batch_input], dim=-1)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        return l5[:, 0]


class IBSNet(nn.Module):
    def __init__(self, latent_size=256):
        super().__init__()

        self.encoder = ResnetPointnet()
        self.decoder = IM_Decoder(latent_size + 3)

        num_params = sum(p.data.nelement() for p in self.parameters())

    def forward(self, pcd, query_points, sample_points_num):
        """
        Args:
            pcd: tensor, (batch_size, pcd_points_num, 3)
            query_points: tensor, (batch_size, query_points_num, 3)
        Returns:
            ufd1_pred: tensor, (batch_size, query_points_num)
            ufd2_pred: tensor, (batch_size, query_points_num)
        """
        latentcode = self.encoder(pcd)
        latentcode = latentcode.repeat_interleave(sample_points_num, dim=0)

        latentcode = torch.cat([latentcode, query_points], 1)

        ufd_pred = self.decoder(latentcode)

        return ufd_pred
