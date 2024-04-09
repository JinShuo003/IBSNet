import sys
import os

sys.path.insert(0, "/home/data/jinshuo/IBS_Net")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import time
import torch
from datetime import datetime, timedelta
import open3d as o3d
import numpy as np

from models.models_transformer import IBSNet
from utils.test_utils import *
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


def reconstruct_ibs(specs: dict, filename: str, model: torch.nn.Module, pcd1: torch.Tensor, pcd2: torch.Tensor, threshold: float):
    """
    :param specs: specification
    :param filename: filename
    :param model: pretrained model
    :param pcd1: point cloud 1
    :param pcd2: point cloud 2
    :param threshold: when |udf1-udf2| < threhold, it is on ibs
    :return: ibs_pcd: o3d.geometry.PointCloud
    """
    device = specs.get("Device")
    pcd1 = pcd1.unsqueeze(0).to(device)
    pcd2 = pcd2.unsqueeze(0).to(device)

    point_num = specs.get("ReconstructOptions").get("ReconstructPointNum")
    diffuse_num = specs.get("ReconstructOptions").get("DiffuseNum")
    diffuse_radius = specs.get("ReconstructOptions").get("DiffuseRadius")

    # 先采集一定数量的种子点，需要保证足够密集
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


def test(model, test_dataloader, specs):
    ibs_threshold = specs.get("ReconstructOptions").get("IBSThreshold")
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            pcd1, pcd2, udf_data, indices = data
            filename_list = [test_dataloader.dataset.pcd1files[i] for i in indices]

            for i, filename in enumerate(filename_list):
                logger.info("filename: {}".format(filename))
                reconstruct_ibs(specs, filename, model, pcd1[i], pcd2[i], ibs_threshold)
                logger.info("filename: {}, success\n".format(filename))


def main_function(specs):
    device = specs.get("Device")
    model_path = specs.get("ModelPath")
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(dataset_udfSamples.UDFSamples, specs, shuffle=False)
    logger.info("init dataloader succeed")

    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    model = get_network(specs, IBSNet, checkpoint)
    logger.info("load trained model succeed, epoch: {}".format(checkpoint["epoch"]))

    time_begin_test = time.time()
    test(model, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))

    time_begin_zip = time.time()
    create_zip(specs)
    time_end_zip = time.time()
    logger.info("use {} to zip".format(time_end_zip - time_begin_zip))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_test.json",
        required=False,
        help="The experiment config file."
    )
    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))

    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("test split: {}".format(specs.get("TestSplit")))
    logger.info("specs file: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))
    logger.info("model: {}".format(specs.get("ModelPath")))

    main_function(specs)
