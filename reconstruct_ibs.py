import sys
import os

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")

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


def get_aabb(specs: dict, filename: str, aabb_dir: str=r"D:\dataset\IBSNet\trainData\boundingBox"):
    category_patten = specs.get("CategoryPatten")
    scene_patten = specs.get("ScenePatten")
    category = re.match(category_patten, filename)
    scene = re.match(scene_patten, filename)
    aabb = geometry_utils.read_mesh(os.path.join(aabb_dir, category, scene)).get_axis_aligned_bounding_box()

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
    udf1, udf2 = model(pcd1, pcd2, query_points)
    udf1_np = udf1.detach().numpy()
    udf2_np = udf2.detach().numpy()
    points = [query_point for i, query_point in enumerate(query_points) if abs(udf1_np[i] - udf2_np[i]) < threshold]

    return np.array(points, dtype=np.float32)


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
    aabb = get_aabb(specs, filename)
    point_num = specs.get("ReconstructOptions").get("ReconstructPointNum")
    seed_num = specs.get("ReconstructOptions").get("SeedPointNum")

    # 先在aabb内采点，然后保留符合条件的点作为初始值
    query_points = random_utils.get_random_points_in_aabb(aabb, 50000)
    query_points = torch.from_numpy(np.array(query_points, dtype=np.float32))
    points = get_points_on_ibs(model, pcd1, pcd2, query_points, threshold)
    while len(points) < point_num:
        rate = 50000 / len(points) + 1
        query_points = random_utils.get_random_points_from_seeds(points, rate, 0.01)
        query_points = query_points[0:50000, :]
        query_points = torch.from_numpy(np.array(query_points, dtype=np.float32))
        points_ = get_points_on_ibs(model, pcd1, pcd2, query_points, threshold)
        points = np.concatenate((points, points_), axis=0)

    ibs_pcd = o3d.geometry.PointCloud()
    ibs_pcd.points = points
    ibs_pcd.farthest_point_down_sample(point_num)

    return ibs_pcd


def test(model, test_dataloader, specs):
    ibs_threshold = specs.get("IBSThreshold")
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            pcd1, pcd2, udf_data, indices = data
            filename_list = [test_dataloader.dataset.pcd_partial_filenames[i] for i in indices]

            ibs_pcd_list = []
            for i, filename in filename_list:
                ibs_pcd = reconstruct_ibs(specs, filename, model, pcd1[i], pcd2[i], ibs_threshold)
                ibs_pcd_list.append(ibs_pcd)

            save_result(specs, filename_list, ibs_pcd_list)
            logger.info("saved {} ibs".format(filename_list.shape[0]))


def main_function(specs):
    device = specs.get("Device")
    model_path = specs.get("ModelPath")
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(dataset_udfSamples.UDFSamples, specs)
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
