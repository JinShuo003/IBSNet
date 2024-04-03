from models.models_utils import *
from torch import nn, einsum
import torch
import torch.nn.functional as F
from models.pn2_utils import *


# -------------------------------------Encoder-----------------------------------
class PointNet2TransformerEncoder(nn.Module):
    def __init__(self, out_dim=512):
        """pointnet++提取骨架点逐点特征，然后利用N层transformer进行注意力计算，得到(B, feature_demension, npoints)的骨架点特征"""
        super().__init__()

        self.sa = PointNet_SA_Module_KNN(256, 16, 3, [32, 128, 512], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(512, dim=128)
        self.transformer_2 = Transformer(512, dim=128)
        self.transformer_3 = Transformer(512, dim=128)
        self.transformer_4 = Transformer(512, dim=128)
        self.transformer_5 = Transformer(512, dim=128)

        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 128, [256, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
        point_cloud: b, 3, n
        Returns:
        feature: (B, feature_demension, npoints)
        """
        xyz = point_cloud
        feature = point_cloud

        xyz, feature, idx = self.sa(xyz, feature)  # (B, 3, npoints), (B, feature_demension, npoints)
        feature = self.transformer_1(feature, xyz)  # (B, feature_demension, npoints)
        feature = self.transformer_2(feature, xyz)  # (B, feature_demension, npoints)
        feature = self.transformer_3(feature, xyz)  # (B, feature_demension, npoints)
        feature = self.transformer_4(feature, xyz)  # (B, feature_demension, npoints)
        feature = self.transformer_5(feature, xyz)  # (B, feature_demension, npoints)

        return feature


class PN2_Transformer_Encoder(nn.Module):
    def __init__(self, out_dim=512):
        """Encoder that encodes information of partial point cloud"""
        super().__init__()
        # self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        # self.transformer_1 = Transformer(128, dim=64)
        # self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        # self.transformer_2 = Transformer(256, dim=64)
        # self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

        self.sa_module_1 = PointNet_SA_Module_KNN(256, 16, 3, [32, 64], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(64, dim=32)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 64, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(128, dim=32)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 128, [256, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
        point_cloud: b, 3, n

        Returns:
        l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 256), (B, 64, 256)
        l1_points = self.transformer_1(l1_points, l1_xyz)  # (B, 64, 256)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 128, 128)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points.squeeze(dim=2)


class CrossAttentionEncoder(nn.Module):
    def __init__(self, in_channel_obj, in_channel_query, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start_query = nn.Conv1d(in_channel_query, dim, 1)
        self.linear_start_obj = nn.Conv1d(in_channel_obj, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel_obj, 1)

    def forward(self, obj_feature, pos, query_points):
        """feed forward of transformer
        Args:
            obj_feature: 物体的特征向量, (B, in_channel_obj, n)
            pos: 物体的点信息, (B, 3, n)
            query_points: 查询点的信息, (B, in_channel_query, samplePoints)
        Returns:
            y: Tensor of features with attention, (B, in_channel_obj, samplePoints)
        """

        # 将查询特征和被查询特征都映射为dim维
        query_feature = self.linear_start_query(query_points)
        obj_feature = self.linear_start(obj_feature)
        b, dim, n = obj_feature.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(obj_feature)
        value = self.conv_value(obj_feature)
        query = self.conv_query(query_feature)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y


class CrossAttention(nn.Module):
    def __init__(self, feature_dimension, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(CrossAttention, self).__init__()
        self.n_knn = n_knn

        self.conv_key = nn.Conv1d(feature_dimension, feature_dimension, 1)
        self.conv_query = nn.Conv2d(3, feature_dimension, 1)
        self.conv_value = nn.Conv1d(feature_dimension, feature_dimension, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, feature_dimension, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(feature_dimension, feature_dimension * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(feature_dimension * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(feature_dimension * attn_hidden_multiplier, feature_dimension, 1)
        )

        self.linear_start = nn.Conv1d(feature_dimension, feature_dimension, 1)
        self.linear_end = nn.Conv1d(feature_dimension, feature_dimension, 1)

    def forward(self, x, y):
        """Feed forward of cross-attention
        Args:
            x: Tensor of queries, (B, n1, 3)
            y: Tensor of features, (B, feature_dimension, n2)

        Returns:
            Tensor of features with cross-attention, (B, feature_dimension, n2)
        """

        identity = y

        y = self.linear_start(y)
        b, dim, n = y.shape

        idx_knn = query_knn(self.n_knn, x, x)
        key = self.conv_key(y)
        value = y
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.unsqueeze(3) - key

        pos_rel = x.unsqueeze(3) - grouping_operation(x, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.unsqueeze(3) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity


# -------------------------------------Dncoder-----------------------------------
class CombinedDecoder(nn.Module):
    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            xyz_in_all=None,
            use_tanh=False,
            latent_dropout=False,
            use_classifier=False,
    ):
        super(CombinedDecoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [2]  # <<<< 2 outputs instead of 1.

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3
            # print("out dim  out_dim)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

            # print(dims[layer], out_dim)
            # classifier
            if self.use_classifier and layer == self.num_layers - 2:
                # print("dim last_layer", dims[layer])
                self.classifier_head = nn.Linear(dims[layer], self.num_class)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        # hand, object, class label
        if self.use_classifier:
            return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)
        else:
            return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)


class IBSNet_transformer(nn.Module):
    def __init__(self, num_samp_per_scene):
        super().__init__()
        # 逐点特征提取层
        self.feature_encoder1 = PointNet2TransformerEncoder()
        self.feature_encoder2 = PointNet2TransformerEncoder()
        # 点云交叉注意力层
        self.pcd_cross_attetntion_encoder1 = CrossAttentionEncoder(in_channel_obj=512, in_channel_query=512, dim=128)
        self.pcd_cross_attetntion_encoder2 = CrossAttentionEncoder(in_channel_obj=512, in_channel_query=512, dim=128)
        # 点云和查询点交叉注意力层
        self.pcd_query_cross_attetntion_encoder1 = CrossAttentionEncoder(in_channel_obj=512, in_channel_query=3, dim=128)
        self.pcd_query_cross_attetntion_encoder2 = CrossAttentionEncoder(in_channel_obj=512, in_channel_query=3, dim=128)

        self.num_samp_per_scene = num_samp_per_scene

    def forward(self, x_obj1, x_obj2, xyz):
        # 将输入点云从(B, N, 3)变换为(B, 3, N)
        x_obj1 = x_obj1.permute(0, 2, 1)
        x_obj2 = x_obj2.permute(0, 2, 1)

        # 利用pointnet++和transformer提取骨架点特征
        feature_obj1 = self.feature_encoder1(x_obj1)  # (B, feature_demension, npoints)
        feature_obj2 = self.feature_encoder2(x_obj2)  # (B, feature_demension, npoints)

        # 利用cross-attention层提取点云的交叉注意力逐点特征
        feature_obj1 = self.pcd_cross_attetntion_encoder1(feature_obj1, feature_obj2)  # (B, feature_demension, npoints)
        feature_obj2 = self.pcd_cross_attetntion_encoder2(feature_obj2, feature_obj1)  # (B, feature_demension, npoints)

        # 利用点云和查询点之间的cross-attention层为每个查询点提取交叉注意力特征
        feature_obj1 = self.pcd_query_cross_attetntion_encoder1(xyz, feature_obj1)  # (B, samplePoints, feature_demension)
        feature_obj2 = self.pcd_query_cross_attetntion_encoder2(xyz, feature_obj2)  # (B, samplePoints, feature_demension)

        latent = torch.cat([feature_obj1, feature_obj2], 1)  # (B, samplePoints, 2*feature_demension)

        decoder_inputs = torch.cat([latent, xyz], 1)
        udf_obj1, udf_obj2 = self.decoder(decoder_inputs)
        return udf_obj1, udf_obj2


