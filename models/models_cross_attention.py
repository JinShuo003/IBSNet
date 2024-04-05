import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel, channel)
        self.sa1_1 = cross_transformer(channel * 2, channel * 2)
        self.sa2 = cross_transformer((channel) * 2, channel * 2)
        self.sa2_1 = cross_transformer((channel) * 4, channel * 4)
        self.sa3 = cross_transformer((channel) * 4, channel * 4)
        self.sa3_1 = cross_transformer((channel) * 8, channel * 8)

        self.relu = nn.GELU()

        self.sa0_d = cross_transformer(channel * 8, channel * 8)
        self.sa1_d = cross_transformer(channel * 8, channel * 8)
        self.sa2_d = cross_transformer(channel * 8, channel * 8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

    def forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_operation(x0, idx_0)
        points = gather_operation(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1, x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_operation(x1, idx_1)
        points = gather_operation(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_operation(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3, x3).contiguous()
        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size, self.channel * 4, N // 8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine


class IM_Decoder(nn.Module):
    """l2+leakyRelu+noNorm+noDrop+all+changeDim-6"""

    def __init__(self, input_dim):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building IM-decoder.')
        super().__init__()

        self.linear_0 = nn.Linear(input_dim, 2048, bias=True)
        self.linear_1 = nn.Linear(input_dim + 2048, 1024, bias=True)
        self.linear_2 = nn.Linear(input_dim + 1024, 512, bias=True)
        self.linear_3 = nn.Linear(input_dim + 512, 256, bias=True)
        self.linear_4 = nn.Linear(input_dim + 256, 128, bias=True)
        self.linear_5 = nn.Linear(128, 2, bias=True)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'IM decoder done(#parameters=%d).' % num_params)

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
        return l5[:, :, 0], l5[:, :, 1]


class IBSNet(nn.Module):
    def __init__(self, latent_size=512):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building network.')
        super().__init__()

        self.encoder1 = PCT_encoder(channel=latent_size)
        self.encoder2 = PCT_encoder(channel=latent_size)

        self.decoder = IM_Decoder(2 * latent_size + 3)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Network done(#parameters=%d).' % num_params)

    def forward(self, pcd1, pcd2, query_points):
        """
        Args:
            pcd1: tensor, (batch_size, pcd_points_num, 3)
            pcd2: tensor, (batch_size, pcd_points_num, 3)
            query_points: tensor, (batch_size, query_points_num, 3)
        Returns:
            ufd1_pred: tensor, (batch_size, query_points_num)
            ufd2_pred: tensor, (batch_size, query_points_num)
        """
        B, N, d = query_points.shape

        latentcode1 = self.encoder1(pcd1)
        latentcode2 = self.encoder2(pcd2)

        latentcode1 = latentcode1.unsqueeze(1).repeat(1, N, 1)
        latentcode2 = latentcode2.unsqueeze(1).repeat(1, N, 1)

        batch_input = torch.cat([latentcode1, latentcode2, query_points], dim=-1)

        ufd1_pred, udf2_pred = self.decoder(batch_input)

        return ufd1_pred, udf2_pred
