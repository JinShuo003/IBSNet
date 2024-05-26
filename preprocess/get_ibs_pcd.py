"""
获取点云形式ibs，从ibs mesh进行采样，以aabb截断
"""
import os
import re
import logging

import multiprocessing
import open3d as o3d
import numpy as np

from utils import log_utils, path_utils, geometry_utils


def getGeometriesPath(specs, instance_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    category = re.match(category_re, instance_name).group()
    scene = re.match(scene_re, instance_name).group()

    geometries_path = dict()

    ibs_mesh_dir = specs.get("path_options").get("geometries_dir").get("ibs_mesh_dir")
    IOU_dir = specs.get("path_options").get("geometries_dir").get("IOU_dir")

    ibs_mesh_filename = '{}.obj'.format(instance_name)
    IOU_filename = '{}.obj'.format(scene)

    geometries_path['ibs_mesh'] = os.path.join(ibs_mesh_dir, category, ibs_mesh_filename)
    geometries_path['aabb'] = os.path.join(IOU_dir, category, IOU_filename)

    return geometries_path


def save_pcd(specs, instance_name, pcd):
    ibs_pcd_save_dir = specs.get("path_options").get("ibs_pcd_save_dir")
    category_re = specs.get("path_options").get("format_info").get("category_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")

    category = re.match(category_re, instance_name).group()
    filename = re.match(filename_re, instance_name).group()
    ibs_pcd_save_path = os.path.join(ibs_pcd_save_dir, category)
    path_utils.generate_path(ibs_pcd_save_path)

    pcd_filename = '{}.ply'.format(filename)
    pcd_path = os.path.join(ibs_pcd_save_path, pcd_filename)

    o3d.io.write_point_cloud(pcd_path, pcd)


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.geometries_path = None
        self.logger = logger

    def get_ibs_pcd(self, mesh, aabb):
        sample_points_num = self.specs.get("sample_optinos").get("sample_points_num")
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()

        ibs_pcd_np = np.array([]).reshape(-1, 3)
        while len(ibs_pcd_np) < sample_points_num:
            points_sample = np.asarray(mesh.sample_points_uniformly(sample_points_num).points)
            points_sample = [point for point in points_sample if self.is_point_in_aabb(point, min_bound, max_bound)]
            points_sample = np.array(points_sample)
            ibs_pcd_np = np.concatenate((ibs_pcd_np, points_sample), axis=0)
            ibs_pcd_np = np.unique(ibs_pcd_np, axis=0)

        ibs_pcd = o3d.geometry.PointCloud()
        ibs_pcd.points = o3d.utility.Vector3dVector(ibs_pcd_np)
        ibs_pcd.farthest_point_down_sample(sample_points_num)

        return ibs_pcd

    def is_point_in_aabb(self, point, min_bound, max_bound):
        for i in range(3):
            if point[i] < min_bound[i] or point[i] > max_bound[i]:
                return False
        return True

    def handle_scene(self, scene):
        aabb_scale = self.specs.get("sample_optinos").get("aabb_scale")
        self.geometries_path = getGeometriesPath(self.specs, scene)

        ibs_mesh = geometry_utils.read_mesh(self.geometries_path.get("ibs_mesh"))
        aabb = geometry_utils.read_mesh(self.geometries_path.get("aabb")).get_axis_aligned_bounding_box()
        aabb.scale(aabb_scale, aabb.get_center())

        ibs_pcd = self.get_ibs_pcd(ibs_mesh, aabb)
        save_pcd(self.specs, scene, ibs_pcd)


def my_process(scene, specs):
    _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)
    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs, _logger)

    try:
        trainDataGenerator.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e.message))
    finally:
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)


if __name__ == '__main__':
    config_filepath = 'configs/get_ibs_pcd.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("ibs_mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("ibs_pcd_save_dir"))

    logger = logging.getLogger("get_IBS_pcd")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # 参数
    view_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                view_list.append(filename)

    if specs.get("use_process_pool"):
        pool = multiprocessing.Pool(processes=specs.get("process_num"))

        for filename in view_list:
            logger.info("current scene: {}".format(filename))
            pool.apply_async(my_process, (filename, specs))

        pool.close()
        pool.join()
    else:
        for filename in view_list:
            logger.info("current scene: {}".format(filename))
            _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), filename)

            trainDataGenerator = TrainDataGenerator(specs, _logger)
            trainDataGenerator.handle_scene(filename)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)
