from torch import nn
from models.pn2_utils import *
from datetime import datetime
import torch.nn.functional as F


class Feature_Extractor_new(nn.Module):
    def __init__(self, points_num=2048, latent_size=512):
        """Encoder that encodes information of partial point cloud"""
        super().__init__()

        self.mlp_in = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(points_num),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32))

        self.transformer_in = Transformer(32)

        self.sa1 = PointNet_SA_Module_KNN(int(points_num / 4), 16, 32, [32, 64])
        self.transformer1 = Transformer(64)

        self.sa2 = PointNet_SA_Module_KNN(int(points_num / 16), 16, 64, [64, 128])
        self.transformer2 = Transformer(128)

        self.sa3 = PointNet_SA_Module_KNN(int(points_num / 64), 16, 128, [128, 256])
        self.transformer3 = Transformer(256)

        self.mlp_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_size))

    def forward(self, point_cloud):
        """
        Args:
        point_cloud: b, n, 3

        Returns:
        l3_points: (B, out_dim, 1)
        """
        feature = self.mlp_in(point_cloud).permute(0, 2, 1).contiguous()
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        feature = self.transformer_in(feature, point_cloud)

        point_cloud, feature = self.sa1(point_cloud, feature)
        feature = self.transformer1(feature, point_cloud)

        point_cloud, feature = self.sa2(point_cloud, feature)
        feature = self.transformer2(feature, point_cloud)

        point_cloud, feature = self.sa3(point_cloud, feature)
        feature = self.transformer3(feature, point_cloud)

        feature = torch.mean(feature, dim=-1)

        return self.mlp_out(feature)


class Feature_Extractor(nn.Module):
    def __init__(self, points_num=2048, latent_size=256):
        """Encoder that encodes information of partial point cloud"""
        super().__init__()
        layer_1_in_size = int(latent_size / 8)
        layer_1_out_size = int(latent_size / 4)
        layer_2_in_size = int(latent_size / 4)
        layer_2_out_size = int(latent_size / 2)
        layer_3_in_size = int(latent_size / 2)
        layer_3_out_size = latent_size
        self.sa1 = PointNet_SA_Module_KNN(int(points_num / 2), 16, 3, [layer_1_in_size, layer_1_out_size],
                                                  group_all=False, if_bn=False, if_idx=True)
        self.transformer1 = Transformer(layer_1_out_size, dim=32)
        self.sa2 = PointNet_SA_Module_KNN(int(points_num / 4), 16, layer_2_in_size,
                                                  [layer_2_in_size, layer_2_out_size], group_all=False, if_bn=False,
                                                  if_idx=True)
        self.transformer2 = Transformer(layer_2_out_size, dim=32)
        self.sa3 = PointNet_SA_Module_KNN(None, None, layer_3_in_size, [layer_3_in_size, layer_3_out_size],
                                                  group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
        point_cloud: b, n, 3

        Returns:
        l3_points: (B, out_dim)
        """
        point_cloud = point_cloud.permute(0, 2, 1)
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa1(l0_xyz, l0_points)  # (B, 3, 256), (B, 64, 256)
        l1_points = self.transformer1(l1_points, l1_xyz)  # (B, 64, 256)
        l2_xyz, l2_points, idx2 = self.sa2(l1_xyz, l1_points)  # (B, 3, 128), (B, 128, 128)
        l2_points = self.transformer2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points.squeeze(dim=2)


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
        self.linear_5 = nn.Linear(512, 2, bias=True)

        num_params = sum(p.data.nelement() for p in self.parameters())

    def forward(self, batch_input):
        l0 = self.linear_0(batch_input)
        l0 = F.prelu(l0, negative_slope=0.02, inplace=True)
        l0 = torch.cat([l0, batch_input], dim=-1)

        l1 = self.linear_1(l0)
        l1 = F.prelu(l1, negative_slope=0.02, inplace=True)
        l1 = torch.cat([l1, batch_input], dim=-1)

        l2 = self.linear_2(l1)
        l2 = F.prelu(l2, negative_slope=0.02, inplace=True)
        l2 = torch.cat([l2, batch_input], dim=-1)

        l3 = self.linear_3(l2)
        l3 = F.prelu(l3, negative_slope=0.02, inplace=True)
        l3 = torch.cat([l3, batch_input], dim=-1)

        l4 = self.linear_4(l3)
        l4 = F.prelu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        return l5[:, 0], l5[:, 1]


class IBSNet(nn.Module):
    def __init__(self, points_num=2048, latent_size=256):
        super().__init__()

        self.encoder1 = Feature_Extractor(points_num=points_num, latent_size=latent_size)
        self.encoder2 = Feature_Extractor(points_num=points_num, latent_size=latent_size)

        self.decoder = IM_Decoder(2 * latent_size + 3)

        num_params = sum(p.data.nelement() for p in self.parameters())

    def forward(self, pcd1, pcd2, query_points, sample_points_num):
        """
        Args:
            pcd1: tensor, (batch_size, pcd_points_num, 3)
            pcd2: tensor, (batch_size, pcd_points_num, 3)
            query_points: tensor, (batch_size, query_points_num, 3)
        Returns:
            ufd1_pred: tensor, (batch_size, query_points_num)
            ufd2_pred: tensor, (batch_size, query_points_num)
        """
        latentcode1 = self.encoder1(pcd1)
        latentcode2 = self.encoder2(pcd2)
        latentcode2 = latentcode2.repeat_interleave(sample_points_num, dim=0)
        latentcode1 = latentcode1.repeat_interleave(sample_points_num, dim=0)

        latentcode = torch.cat([latentcode1, latentcode2, query_points], 1)

        ufd1_pred, udf2_pred = self.decoder(latentcode)

        return ufd1_pred, udf2_pred
