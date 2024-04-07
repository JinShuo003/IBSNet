import json
import re
import os
import open3d as o3d
import numpy as np
import shutil
import torch.utils.data as data_utils

from utils.log_utils import LogFactory
from utils.geometry_utils import get_pcd_from_np


def get_network(specs, model_class, checkpoint, **kwargs):
    assert checkpoint is not None

    device = specs.get("Device")
    logger = LogFactory.get_logger(specs.get("LogOptions"))

    network = model_class(**kwargs).to(device)

    logger.info("load model parameter from epoch {}".format(checkpoint["epoch"]))
    network.load_state_dict(checkpoint["model"])

    return network


def get_dataloader(dataset_class, specs: dict):
    data_source = specs.get("DataSource")
    test_split_file = specs.get("TestSplit")
    batch_size = specs.get("BatchSize")
    num_data_loader_threads = specs.get("DataLoaderThreads")
    logger = LogFactory.get_logger(specs.get("LogOptions"))

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    test_dataset = dataset_class(data_source, test_split)
    logger.info("length of test_dataset: {}".format(test_dataset.__len__()))

    # get dataloader
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logger.info("length of test_dataloader: {}".format(test_dataloader.__len__()))

    return test_dataloader


def save_result(specs: dict, filename_list: list, ibs_pcd_list: list):
    save_dir = specs.get("ResultSaveDir")
    tag = specs.get("TAG")

    filename_patten = specs.get("FileNamePatten")

    for index, filename_abs in enumerate(filename_list):
        # [dataset, category, filename], example:[MVP, scene1, scene1.1000_view0_0.ply]
        _, category, filename = filename_abs.split('/')
        filename = re.match(filename_patten, filename).group()  # scene1.1000_view0

        # the real directory is save_dir/tag/category
        save_path = os.path.join(save_dir, tag, category)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename_final = "{}.ply".format(filename)
        absolute_path = os.path.join(save_path, filename_final)
        ibs_pcd = ibs_pcd_list[index]

        o3d.io.write_point_cloud(absolute_path, ibs_pcd)


def create_zip(specs: dict):
    test_result_save_dir = specs.get("ResultSaveDir")
    tag = specs.get("TAG")
    dataset = specs.get("Dataset")
    zip_file_base_name = os.path.join(test_result_save_dir, tag, tag)
    zip_dir = os.path.join(test_result_save_dir, tag, dataset)
    shutil.make_archive(zip_file_base_name, 'zip', zip_dir)


def update_loss_dict(dist_dict_total: dict, filename_list, dist, tag: str):
    assert dist.shape[0] == len(filename_list)
    assert tag in dist_dict_total.keys()

    dist_dict = dist_dict_total.get(tag)
    for idx, filename in enumerate(filename_list):
        category = "scene{}".format(str(int(re.findall(r'\d+', filename)[0])))
        if category not in dist_dict:
            dist_dict[category] = {
                "dist_total": 0,
                "num": 0
            }
        dist_dict[category]["dist_total"] += dist[idx]
        dist_dict[category]["num"] += 1


def cal_avrg_dist(dist_dict_total: dict):
    for tag in dist_dict_total.keys():
        cal_avrg_dist_single(dist_dict_total, tag)


def cal_avrg_dist_single(dist_dict_total: dict, tag: str):
    dist_dict = dist_dict_total.get(tag)
    dist_total = 0
    num = 0
    for category in dist_dict.keys():
        dist_dict[category]["avrg_dist"] = dist_dict[category]["dist_total"] / dist_dict[category]["num"]
        dist_total += dist_dict[category]["dist_total"]
        num += dist_dict[category]["num"]
    dist_dict["avrg_dist"] = dist_total / num
