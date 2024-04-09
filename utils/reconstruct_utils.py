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


def save_result(specs: dict, filename: str, ibs_pcd: o3d.geometry.PointCloud):
    save_dir = specs.get("ResultSaveDir")
    tag = specs.get("TAG")

    filename_patten = specs.get("FileNamePatten")

    # [dataset, category, filename], example:[MVP, scene1, scene1.1000_view0_0.ply]
    _, category, filename = filename.split('/')
    filename = re.match(filename_patten, filename).group()  # scene1.1000_view0

    save_path = os.path.join(save_dir, tag, category)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    filename_final = "{}.ply".format(filename)
    absolute_path = os.path.join(save_path, filename_final)

    o3d.io.write_point_cloud(absolute_path, ibs_pcd)


def create_zip(specs: dict):
    reconstruct_result_save_dir = specs.get("reconstruct_result_save_dir")
    tag = specs.get("TAG")
    zip_file_base_name = os.path.join(reconstruct_result_save_dir, tag)
    zip_dir = os.path.join(reconstruct_result_save_dir, tag)
    shutil.make_archive(zip_file_base_name, 'zip', zip_dir)
