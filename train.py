import logging
import os.path

import torch
import torch.utils.data as data_utils
import deep_sdf
import deep_sdf.workspace as ws
import json
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
from utils import *
from datetime import datetime, timedelta

from models.models import *


def visualize_data1(pcd1, pcd2, xyz, udf_gt1, udf_gt2):
    # 将udf数据拆分开，并且转移到cpu
    xyz = xyz.cpu().detach().numpy()
    pcd1_np = pcd1.cpu().detach().numpy()
    pcd2_np = pcd2.cpu().detach().numpy()

    xyz = np.split(xyz, pcd1_np.shape[0])
    udf_gt1 = np.split(udf_gt1, pcd1_np.shape[0])
    udf_gt2 = np.split(udf_gt2, pcd1_np.shape[0])

    for i in range(pcd1_np.shape[0]):
        ibs_points = [xyz[i][j] for j in range(xyz[i].shape[0]) if abs(udf_gt1[i][j] - udf_gt2[i][j]) < 0.003]

        pcd1_o3d = o3d.geometry.PointCloud()
        pcd2_o3d = o3d.geometry.PointCloud()
        ibs_o3d = o3d.geometry.PointCloud()

        pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1_np[i])
        pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2_np[i])
        ibs_o3d.points = o3d.utility.Vector3dVector(ibs_points)

        pcd1_o3d.paint_uniform_color([1, 0, 0])
        pcd2_o3d.paint_uniform_color([0, 1, 0])
        ibs_o3d.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([ibs_o3d, pcd1_o3d, pcd2_o3d])


def visualize_data(pcd1, pcd2, sdf_data):
    pcd1_np = pcd1.numpy()[0, :, :].reshape(-1, 3)
    pcd2_np = pcd2.numpy()[0, :, :].reshape(-1, 3)
    sdf_np = sdf_data.numpy()[0, :, :].reshape(-1, 5)

    surface_points1 = [points[0:3] for points in sdf_np if abs(points[3]) < 0.1]
    surface_points2 = [points[0:3] for points in sdf_np if abs(points[4]) < 0.1]
    surface_points_ibs = [points[0:3] for points in sdf_np if abs(points[3] - points[4]) < 0.1]

    pcd1_o3d = o3d.geometry.PointCloud()
    pcd2_o3d = o3d.geometry.PointCloud()
    sdf1_o3d = o3d.geometry.PointCloud()
    sdf2_o3d = o3d.geometry.PointCloud()
    ibs_o3d = o3d.geometry.PointCloud()

    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1_np)
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2_np)
    sdf1_o3d.points = o3d.utility.Vector3dVector(surface_points1)
    sdf2_o3d.points = o3d.utility.Vector3dVector(surface_points2)
    ibs_o3d.points = o3d.utility.Vector3dVector(surface_points_ibs)

    pcd1_o3d.paint_uniform_color([1, 0, 0])
    pcd2_o3d.paint_uniform_color([1, 0, 0])
    sdf1_o3d.paint_uniform_color([0, 1, 0])
    sdf2_o3d.paint_uniform_color([0, 1, 0])
    ibs_o3d.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd1_o3d, pcd2_o3d, sdf1_o3d, sdf2_o3d, ibs_o3d])


def visualize(xyz, udf1, udf2, scene_per_batch):
    xyz = xyz.cpu()
    sdf_data = torch.cat([xyz, udf1, udf2], 1)
    sdf_data = torch.reshape(sdf_data, (scene_per_batch, -1, 5))
    for i in range(scene_per_batch):
        scene_data = sdf_data[i, :, :].squeeze(0)

        surface_points1 = [points[0:3] for points in scene_data if abs(points[3]) < 0.02]
        surface_points2 = [points[0:3] for points in scene_data if abs(points[4]) < 0.02]
        surface_points_ibs = [points[0:3] for points in scene_data if abs(points[3] - points[4]) < 0.02]

        sdf1_o3d = o3d.geometry.PointCloud()
        sdf2_o3d = o3d.geometry.PointCloud()
        ibs_o3d = o3d.geometry.PointCloud()

        sdf1_o3d.points = o3d.utility.Vector3dVector(surface_points1)
        sdf2_o3d.points = o3d.utility.Vector3dVector(surface_points2)
        ibs_o3d.points = o3d.utility.Vector3dVector(surface_points_ibs)

        sdf1_o3d.paint_uniform_color([1, 0, 0])
        sdf2_o3d.paint_uniform_color([0, 1, 0])
        ibs_o3d.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([sdf1_o3d, sdf2_o3d, ibs_o3d])


def get_dataloader(specs):
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    logging.info("batch_size: {}".format(scene_per_batch))
    logging.info("dataLoader threads: {}".format(num_data_loader_threads))

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    sdf_train_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )
    sdf_test_dataset = deep_sdf.data.SDFSamples(
        data_source, test_split, num_samp_per_scene, load_ram=False
    )
    logging.info("length of sdf_train_dataset: {}".format(sdf_train_dataset.__len__()))
    logging.info("length of sdf_test_dataset: {}".format(sdf_test_dataset.__len__()))

    sdf_train_loader = data_utils.DataLoader(
        sdf_train_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    sdf_test_loader = data_utils.DataLoader(
        sdf_test_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logging.info("length of sdf_train_loader: {}".format(sdf_train_loader.__len__()))
    logging.info("length of sdf_test_loader: {}".format(sdf_test_loader.__len__()))

    return sdf_train_loader, sdf_test_loader


def get_network(specs):
    num_samp_per_scene = specs["SamplesPerScene"]
    latent_size = specs["CodeLength"]
    device = specs["Device"]

    encoder_obj1 = ResnetPointnet(c_dim=256, dim=3, hidden_dim=256)
    encoder_obj2 = ResnetPointnet(c_dim=256, dim=3, hidden_dim=256)
    # encoder_obj1 = PointTransformerLayer()
    # encoder_obj2 = PointTransformerLayer()
    # decoder = Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = CombinedDecoder(latent_size, **specs["NetworkSpecs"])
    IBS_Net = IBSNet(encoder_obj1, encoder_obj2, decoder, num_samp_per_scene)

    if torch.cuda.is_available():
        IBS_Net = IBS_Net.to(device)
    return IBS_Net


def get_optimizer(specs, IBS_Net):
    lr_schedules = get_learning_rate_schedules(specs)
    optimizer = torch.optim.Adam(IBS_Net.parameters(), lr_schedules.get_learning_rate(0))

    return lr_schedules, optimizer


def get_tensorboard_writer(specs, log_path, IBS_Net, TIMESTAMP):
    device = specs["Device"]
    train_split_file = specs["TrainSplit"]

    writer_path = os.path.join(log_path, "{}_{}".format(os.path.basename(train_split_file).split('.')[-2], TIMESTAMP))
    if os.path.isdir(writer_path):
        os.mkdir(writer_path)

    tensorboard_writer = SummaryWriter(writer_path)

    input_pcd1_shape = torch.randn(1, 512, 3)
    input_pcd2_shape = torch.randn(1, 512, 3)
    input_xyz_shape = torch.randn(30000, 3)

    if torch.cuda.is_available():
        input_pcd1_shape = input_pcd1_shape.to(device)
        input_pcd2_shape = input_pcd2_shape.to(device)
        input_xyz_shape = input_xyz_shape.to(device)

    tensorboard_writer.add_graph(IBS_Net, (input_pcd1_shape, input_pcd2_shape, input_xyz_shape))

    return tensorboard_writer


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def train(IBS_Net, sdf_train_loader, lr_schedules, optimizer, epoch, specs, tensorboard_writer, TIMESTAMP):
    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        optimizer.param_groups[0]["lr"] = lr_schedules.get_learning_rate(epoch)

    scene_per_batch = specs["ScenesPerBatch"]
    num_samp_per_scene = specs["SamplesPerScene"]
    para_save_dir = specs["ParaSaveDir"]
    train_split_file = specs["TrainSplit"]
    device = specs["Device"]

    loss_udf1 = torch.nn.L1Loss(reduction="sum")
    loss_udf2 = torch.nn.L1Loss(reduction="sum")

    IBS_Net.train()
    adjust_learning_rate(lr_schedules, optimizer, epoch)
    logging.info('epoch: {}, learning rate: {}'.format(epoch, lr_schedules.get_learning_rate(epoch)))

    train_total_loss = 0
    for pcd1, pcd2, sdf_data, indices in sdf_train_loader:
        sdf_data = sdf_data.reshape(-1, 5)

        pcd1.requires_grad = False
        pcd2.requires_grad = False
        sdf_data.requires_grad = False

        xyz = sdf_data[:, 0:3]

        pcd1 = pcd1.to(device)
        pcd2 = pcd2.to(device)
        xyz = xyz.to(device)

        udf_gt1 = sdf_data[:, 3].unsqueeze(1)
        udf_gt2 = sdf_data[:, 4].unsqueeze(1)

        visualize_data1(pcd1, pcd2, xyz, udf_gt1, udf_gt2)
        udf_pred1, udf_pred2 = IBS_Net(pcd1, pcd2, xyz)

        # 计算每个点的平均l1-loss
        batch_loss1 = loss_udf1(udf_pred1, udf_gt1.to(device)) / (scene_per_batch * num_samp_per_scene)
        batch_loss2 = loss_udf2(udf_pred2, udf_gt2.to(device)) / (scene_per_batch * num_samp_per_scene)
        batch_loss = (batch_loss1 + batch_loss2) / 2

        # 统计一个epoch的平均loss
        train_total_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    train_avrg_loss = train_total_loss / sdf_train_loader.__len__()
    tensorboard_writer.add_scalar("train_loss", train_avrg_loss, epoch)
    logging.info('train_avrg_loss: {}'.format(train_avrg_loss))

    # 保存模型
    if epoch % 5 == 0:
        para_save_path = os.path.join(para_save_dir, "{}_{}".format(os.path.basename(train_split_file).split('.')[-2], TIMESTAMP))
        if not os.path.isdir(para_save_path):
            os.mkdir(para_save_path)
        model_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))
        torch.save(IBS_Net, model_filename)


def test(IBS_Net, sdf_test_loader, epoch, specs, tensorboard_writer):
    scene_per_batch = specs["ScenesPerBatch"]
    num_samp_per_scene = specs["SamplesPerScene"]
    device = specs["Device"]

    loss_udf1 = torch.nn.L1Loss(reduction="sum")
    loss_udf2 = torch.nn.L1Loss(reduction="sum")
    with torch.no_grad():
        test_total_loss = 0
        for pcd1, pcd2, sdf_data, indices in sdf_test_loader:
            sdf_data = sdf_data.reshape(-1, 5)

            pcd1.requires_grad = False
            pcd2.requires_grad = False
            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]

            pcd1 = pcd1.to(device)
            pcd2 = pcd2.to(device)
            xyz = xyz.to(device)

            udf_gt1 = sdf_data[:, 3].unsqueeze(1)
            udf_gt2 = sdf_data[:, 4].unsqueeze(1)

            visualize_data(pcd1, pcd2, xyz)
            udf_pred1, udf_pred2 = IBS_Net(pcd1, pcd2, xyz)

            batch_loss1 = loss_udf1(udf_pred1, udf_gt1.to(device)) / (scene_per_batch * num_samp_per_scene)
            batch_loss2 = loss_udf2(udf_pred2, udf_gt2.to(device)) / (scene_per_batch * num_samp_per_scene)
            batch_loss = (batch_loss1 + batch_loss2) / 2

            test_total_loss += batch_loss.item()

        test_avrg_loss = test_total_loss / sdf_test_loader.__len__()
        tensorboard_writer.add_scalar("test_loss", test_avrg_loss, epoch)
        logging.info(' test_avrg_loss: {}\n'.format(test_avrg_loss))


def main_function(experiment_config_file):
    specs = ws.load_experiment_specifications(experiment_config_file)
    epoch_num = specs["NumEpochs"]
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))

    logging.info("current experiment config file: {}".format(experiment_config_file))
    logging.info("current time: {}".format(TIMESTAMP))
    logging.info("There are {} epochs in total".format(epoch_num))

    sdf_train_loader, sdf_test_loader = get_dataloader(specs)
    IBS_Net = get_network(specs)
    lr_schedules, optimizer = get_optimizer(specs, IBS_Net)
    tensorboard_writer = get_tensorboard_writer(specs, './tensorboard_logs', IBS_Net, TIMESTAMP)

    for epoch in range(epoch_num):
        train(IBS_Net, sdf_train_loader, lr_schedules, optimizer, epoch, specs, tensorboard_writer, TIMESTAMP)
        test(IBS_Net, sdf_test_loader, epoch, specs, tensorboard_writer)

    tensorboard_writer.close()


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a IBS Net")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/scan/specs.json",
        required=False,
        help="The experiment config file."
    )

    # 添加日志参数
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_config_file)
