import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/data/jinshuo/IBS_Net")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import torch
import logging

from models.models_transformer import IBSNet
from utils.reconstruct_utils import *
from utils import log_utils, path_utils, geometry_utils, random_utils
from dataset import dataset_udfSamples


def get_aabb(specs: dict, filename: str, aabb_dir: str=r"data/boundingBox"):
    category_patten = specs.get("CategoryPatten")
    scene_patten = specs.get("ScenePatten")
    aabb_scale = specs.get("ReconstructOptions").get("AABBScale")
    category = re.match(category_patten, filename).group()
    scene = re.match(scene_patten, filename).group()
    filename = "{}.obj".format(scene)
    aabb = geometry_utils.read_mesh(os.path.join(aabb_dir, category, filename)).get_axis_aligned_bounding_box()
    aabb.scale(aabb_scale, aabb.get_center())

    return aabb

def is_point_on_ibs(point: np.ndarray, min_bound, max_bound):
    for i in range(3):
        if point[i] < min_bound[i] or point[i] > max_bound[i]:
            return False
    return True


def get_points_on_ibs(model: torch.nn.Module, pcd1: torch.Tensor, pcd2: torch.Tensor, query_points: torch.Tensor, threshold: float):
    """
    :param model: pretrained model
    :param pcd1: point cloud 1, (2048, 3)
    :param pcd2: point cloud 2, (2048, 3)
    :param query_points: query points, (n, 3)
    :param threshold: when |udf1-udf2| < threhold, it is on ibs
    :return: points: np.ndarray, points in query_points which is on ibs
    """
    assert pcd1.device == pcd2.device == query_points.device
    sample_points_num = query_points.shape[0]
    udf1, udf2 = model(pcd1, pcd2, query_points, sample_points_num)
    udf1_np = udf1.detach().cpu().numpy()
    udf2_np = udf2.detach().cpu().numpy()
    points = [query_point.detach().cpu().numpy() for i, query_point in enumerate(query_points) if abs(udf1_np[i] - udf2_np[i]) < threshold]

    return np.array(points, dtype=np.float32).reshape(-1, 3)


def get_seed_points(specs: dict, filename: str, model: torch.nn.Module, pcd1: torch.Tensor, pcd2: torch.Tensor, threshold: float):
    device = specs.get("Device")
    seed_num = specs.get("ReconstructOptions").get("SeedPointNum")

    _, _, filename = filename.split('/')
    aabb = get_aabb(specs, filename)
    seed_points = np.zeros((0, 3))

    while seed_points.shape[0] < seed_num:
        query_points = random_utils.get_random_points_in_aabb(aabb, seed_num)
        query_points = torch.from_numpy(np.array(query_points, dtype=np.float32)).to(device)
        points_on_ibs = get_points_on_ibs(model, pcd1, pcd2, query_points, threshold)
        seed_points = np.concatenate((seed_points, points_on_ibs), axis=0)

    return seed_points


def get_pcd_torch(specs: dict, filename: str):
    device = specs.get("Device")
    pcd_dir = specs.get("path_options").get("geometries_dir").get("pcd_dir")
    category_re = specs.get("path_options").get("format_info").get("category_re")
    category = re.match(category_re, filename)

    pcd_path = os.path.join(pcd_dir, category)

    pcd1_filename = "{}_0.ply".format(filename)
    pcd2_filename = "{}_1.ply".format(filename)

    pcd1_path = os.path.join(pcd_path, pcd1_filename)
    pcd2_path = os.path.join(pcd_path, pcd2_filename)

    pcd1 = geometry_utils.read_point_cloud(pcd1_path)
    pcd2 = geometry_utils.read_point_cloud(pcd2_path)

    pcd1_torch = torch.from_numpy(np.array(pcd1.points, dtype=np.float32)).to(device)
    pcd2_torch = torch.from_numpy(np.array(pcd2.points, dtype=np.float32)).to(device)

    return pcd1_torch, pcd2_torch


def reconstruct_ibs(specs: dict, filename: str, model: torch.nn.Module):
    """
    :param specs: specification
    :param filename: filename
    :param model: pretrained model
    :return: ibs_pcd: o3d.geometry.PointCloud
    """
    device = specs.get("Device")
    threshold = specs.get("ReconstructOptions").get("IBSThreshold")
    pcd1, pcd2 = get_pcd_torch(specs, filename)
    pcd1 = pcd1.unsqueeze(0)
    pcd2 = pcd2.unsqueeze(0)

    point_num = specs.get("ReconstructOptions").get("ReconstructPointNum")
    diffuse_num = specs.get("ReconstructOptions").get("DiffuseNum")
    diffuse_radius = specs.get("ReconstructOptions").get("DiffuseRadius")

    # generate seed points
    seed_points = get_seed_points(specs, filename, model, pcd1, pcd2, threshold)
    points = seed_points

    while points.shape[0] < point_num:
        logger.info("current points number: {}, target: {}".format(points.shape[0], point_num))
        query_points = random_utils.get_random_points_from_seeds(points, diffuse_num, diffuse_radius)
        query_points = np.array(query_points, dtype=np.float32)
        query_points = torch.from_numpy(query_points).to(device)
        points_ = get_points_on_ibs(model, pcd1, pcd2, query_points, threshold)
        points = np.concatenate((points, points_), axis=0)

    ibs_pcd = o3d.geometry.PointCloud()
    ibs_pcd.points = o3d.utility.Vector3dVector(points)
    ibs_pcd.farthest_point_down_sample(point_num)

    save_result(specs, filename, ibs_pcd)


def get_filename_list(specs):
    test_split_file_path = specs.get("path_options").get("test_split_file_path")
    handle_category = specs.get("path_options").get("format_info").get("handle_category")
    handle_scene = specs.get("path_options").get("format_info").get("handle_scene")
    handle_filename = specs.get("path_options").get("format_info").get("handle_filename")
    filename_list = []

    with open(test_split_file_path, "r") as f:
        split_file = json.load(f)
        for dataset in split_file:
            for category in split_file[dataset]:
                if re.match(handle_category, category) is None:
                    continue
                for filename in split_file[dataset][category]:
                    if re.match(handle_scene, filename) is None:
                        continue
                    if re.match(handle_filename, filename) is None:
                        continue
                    filename_list.append(filename)
    return filename_list


if __name__ == '__main__':
    config_filepath = 'configs/reconstruct_ibs.json'
    specs = path_utils.read_config(config_filepath)
    path_utils.generate_path(specs.get("path_options").get("ibs_mesh_save_dir"))

    logger = logging.getLogger("reconstruct ibs")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # get pretrained model
    device = specs.get("Device")
    model_path = specs.get("ModelPath")
    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    model = get_network(specs, IBSNet, checkpoint)

    # get instance name
    filename_list = get_filename_list(specs)

    # reconstruct
    time_begin_test = time.time()
    for filename in filename_list:
        logger.info("current scene: {}".format(filename))
        _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), filename)
        reconstruct_ibs(specs, filename, model)
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))

    # zip
    time_begin_zip = time.time()
    create_zip(specs)
    time_end_zip = time.time()
    logger.info("use {} to zip".format(time_end_zip - time_begin_zip))