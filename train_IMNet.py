"""
IMNet的训练代码
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import argparse
import time
from datetime import datetime, timedelta

from utils import path_utils
from utils.train_utils import *
from dataset import dataset_udfSamples_single
from models.models_IMNet import IBSNet


def get_dataloader(dataset_class, specs: dict):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    data_source = specs.get("DataSource")
    train_split_file = specs.get("TrainSplit")
    test_split_file = specs.get("TestSplit")
    trian_options = specs.get("TrainOptions")
    batch_size = trian_options.get("BatchSize")
    num_data_loader_threads = trian_options.get("DataLoaderThreads")
    obj_idx = trian_options.get("ObjIdx")

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = dataset_class(data_source, train_split, obj_idx)
    test_dataset = dataset_class(data_source, test_split, obj_idx)
    logger.info("length of train_dataset: {}".format(train_dataset.__len__()))
    logger.info("length of test_dataset: {}".format(test_dataset.__len__()))

    # get dataloader
    train_dataloader = data_utils.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logger.info("length of train_dataloader: {}".format(train_dataloader.__len__()))
    logger.info("length of test_dataloader: {}".format(test_dataloader.__len__()))

    return train_dataloader, test_dataloader


def train(network, train_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")
    obj_idx = int(specs.get("TrainOptions").get("ObjIdx"))

    network.train()
    logger.info("")
    logger.info('epoch: {}, learning rate: {}'.format(epoch, optimizer.param_groups[0]["lr"]))

    loss_l1 = torch.nn.L1Loss(reduction="mean")
    loss_l2 = torch.nn.MSELoss(reduction="mean")

    train_total_loss_l1 = 0
    train_total_loss_l2 = 0
    for data in train_dataloader:
        pcd, udf_data, indices = data
        udf_data = udf_data.reshape(-1, 5)

        optimizer.zero_grad()

        xyz = udf_data[:, 0:3]
        udf_gt = udf_data[:, 3 + obj_idx].to(device)
        pcd = pcd.to(device)
        xyz = xyz.to(device)

        udf_pred = network(pcd, xyz)
        
        l1_loss = loss_l1(udf_pred, udf_gt)
        l2_loss = loss_l2(udf_pred, udf_gt)

        train_total_loss_l1 += l1_loss.item()
        train_total_loss_l2 += l2_loss.item()

        l2_loss.backward()
        optimizer.step()

    lr_schedule.step()

    record_loss_info(specs, "train_loss_l1", train_total_loss_l1 / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_l2", train_total_loss_l2 / train_dataloader.__len__(), epoch, tensorboard_writer)


def test(network, test_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer, best_loss, best_epoch):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")
    obj_idx = int(specs.get("TrainOptions").get("ObjIdx"))

    network.eval()

    loss_l1 = torch.nn.L1Loss(reduction="mean")
    loss_l2 = torch.nn.MSELoss(reduction="mean")

    test_total_loss_l1 = 0
    test_total_loss_l2 = 0
    with torch.no_grad():
        for data in test_dataloader:
            pcd, udf_data, indices = data
            udf_data = udf_data.reshape(-1, 5)

            xyz = udf_data[:, 0:3]
            udf_gt = udf_data[:, 3 + obj_idx].to(device)
            pcd = pcd.to(device)
            xyz = xyz.to(device)

            udf_pred = network(pcd, xyz)

            l1_loss = loss_l1(udf_pred, udf_gt)
            l2_loss = loss_l2(udf_pred, udf_gt)

            test_total_loss_l1 += l1_loss.item()
            test_total_loss_l2 += l2_loss.item()

        record_loss_info(specs, "test_loss_l1", test_total_loss_l1 / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_l2", test_total_loss_l2 / test_dataloader.__len__(), epoch, tensorboard_writer)

        if test_total_loss_l1 < best_loss:
            best_epoch = epoch
            best_loss = test_total_loss_l1
            logger.info('current best epoch: {}, cd: {}'.format(best_epoch, best_loss))
        save_model(specs, network, lr_schedule, optimizer, epoch)

        return best_loss, best_epoch


def main_function(specs):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    epoch_num = specs.get("TrainOptions").get("NumEpochs")
    continue_train = specs.get("TrainOptions").get("ContinueTrain")

    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))

    logger.info("current model TAG: {}".format(specs.get("TAG")))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("There are {} epochs in total".format(epoch_num))

    train_loader, test_loader = get_dataloader(dataset_udfSamples_single.UDFSamples, specs)
    checkpoint = get_checkpoint(specs)
    network = get_network(specs, IBSNet, checkpoint)
    optimizer = get_optimizer(specs, network, checkpoint)
    lr_scheduler_class, kwargs = get_lr_scheduler_info(specs)
    lr_scheduler = get_lr_scheduler(specs, optimizer, checkpoint, lr_scheduler_class, **kwargs)
    tensorboard_writer = get_tensorboard_writer(specs)

    best_cd = 1e8
    best_epoch = -1
    epoch_begin = 0
    if continue_train:
        last_epoch = specs.get("TrainOptions").get("ContinueFromEpoch")
        epoch_begin = last_epoch + 1
        logger.info("continue train from epoch {}".format(epoch_begin))
    for epoch in range(epoch_begin, epoch_num + 1):
        time_begin_train = time.time()
        train(network, train_loader, lr_scheduler, optimizer, epoch, specs, tensorboard_writer)
        time_end_train = time.time()
        logger.info("use {} to train".format(time_end_train - time_begin_train))

        time_begin_test = time.time()
        best_cd, best_epoch = test(network, test_loader, lr_scheduler, optimizer, epoch, specs, tensorboard_writer, best_cd, best_epoch)
        time_end_test = time.time()
        logger.info("use {} to test".format(time_end_test - time_begin_test))

    tensorboard_writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train IMNet")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_train_IMNet.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)

