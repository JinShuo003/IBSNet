from models.models_utils import *
from torch import nn, einsum
import torch
import torch.nn.functional as F


# -------------------------------------Encoder-----------------------------------
# Resnet Blocks
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


# 带Resnet的Pointnet（完整的点云特征提取）
class ResnetPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

        # 每个ResnetBlock的维度为(2 * hidden_dim, hidden_dim)，这是因为需要与池化后结果进行拼接
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


class Decoder(nn.Module):
    def __init__(
            self,
            latent_size=512,
            dims=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
            dropout=[0, 1, 2, 3, 4, 5, 6, 7],
            dropout_prob=0.2,
            norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            latent_in=[4],
            weight_norm=True,
            xyz_in_all=False,
            use_tanh=False,
            latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        # 定义每个全连接层的输入输出维度
        dims = [latent_size + 3] + dims + [2]
        # 全连接层数
        self.num_layers = len(dims)
        # 是否进行weight normalization
        self.weight_norm = weight_norm
        # 进行normalization的层（可能是batch norm或weight norm）
        self.norm_layers = norm_layers
        # Decoder的输入拼接到第几个线性层的输入
        self.latent_in = latent_in
        # 是否对latent_code进行dropout
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        # 是否所有层都拼接xyz
        self.xyz_in_all = xyz_in_all

        # 根据读取到的配置参数生成全部线性层
        for layer in range(0, self.num_layers - 1):
            # 处理跳层，跳层处的out_dim单独处理，获取out_dim
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                # 是否每层都拼接xyz
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3
            # 若需要进行weight_normalization
            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            # 不进行weight_normalization
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            # 不进行weight_norm，并且设定了某些层需要norm，则生成batch norm层
            if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        # 是否使用tanh
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # dropout的概率
        self.dropout_prob = dropout_prob
        # 进行dropout的层
        self.dropout = dropout
        # tanh层
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        # print('input.shape[2]', input.shape[2])

        # 获取xyz
        xyz = input[:, -3:]

        # 如果需要对latent code进行dropout，则先将latent code 切分出来，drop完毕后再与坐标拼接
        # if input_size_2 > 3 and self.latent_dropout:
        if self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        # 遍历所有层
        for layer in range(0, self.num_layers - 1):
            # 根据层号获取层
            lin = getattr(self, "lin" + str(layer))
            # 当前层需要拼接latent_code
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            # 若设置了所有层都拼接xyz
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            # 当前线性层进行运算
            x = lin(x)
            # 如果最后一层需要进行tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            # 在最后一层之前，进行normalization和dropout
            if layer < self.num_layers - 2:
                # 进行batch norm
                if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                # 进行dropout
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # 最后进行tanh
        if hasattr(self, "th"):
            x = self.th(x)

        return x


class IBSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_obj1 = ResnetPointnet()
        self.encoder_obj2 = ResnetPointnet()
        self.decoder = Decoder()
        self.num_samp_per_scene = 50000

    def forward(self, x_obj1, x_obj2, xyz):
        x_obj1 = self.encoder_obj1(x_obj1)
        latent_obj1 = x_obj1.repeat_interleave(self.num_samp_per_scene, dim=0)
        x_obj2 = self.encoder_obj2(x_obj2)
        latent_obj2 = x_obj2.repeat_interleave(self.num_samp_per_scene, dim=0)

        latent = torch.cat([latent_obj1, latent_obj2], 1)

        decoder_inputs = torch.cat([latent, xyz], 1)
        udf_obj1, udf_obj2 = self.decoder(decoder_inputs)
        return udf_obj1, udf_obj2
