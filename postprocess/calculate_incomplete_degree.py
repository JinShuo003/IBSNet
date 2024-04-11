"""
计算点云的不完整程度，以完整点云到残缺点云的单向cd值表示，该值越大说明越残缺
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

    pcd_gt_dir = specs.get("path_options").get("geometries_dir").get("pcd_gt_dir")
    pcd_partial_dir = specs.get("path_options").get("geometries_dir").get("pcd_partial_dir")

    pcd1_gt_filename = '{}_0.ply'.format(scene)
    pcd2_gt_filename = '{}_1.ply'.format(scene)
    pcd1_partial_filename = '{}_0.ply'.format(instance_name)
    pcd2_partial_filename = '{}_1.ply'.format(instance_name)

    geometries_path['pcd1_gt'] = os.path.join(pcd_gt_dir, category, pcd1_gt_filename)
    geometries_path['pcd2_gt'] = os.path.join(pcd_gt_dir, category, pcd2_gt_filename)
    geometries_path['pcd1_partial'] = os.path.join(pcd_partial_dir, category, pcd1_partial_filename)
    geometries_path['pcd2_partial'] = os.path.join(pcd_partial_dir, category, pcd2_partial_filename)

    return geometries_path


def save_cd(specs, instance_name, cd1, cd2):
    cd_save_dir = specs.get("path_options").get("incomplete_cd_save_dir")
    category_patten = specs.get("path_options").get("format_info").get("category_re")

    path_utils.generate_path(cd_save_dir)
    category = re.match(category_patten, instance_name).group()

    cd_save_path = os.path.join(cd_save_dir, category)
    path_utils.generate_path(cd_save_path)

    cd1_filename = '{}_0.txt'.format(instance_name)
    cd2_filename = '{}_1.txt'.format(instance_name)
    cd1_filepath = os.path.join(cd_save_path, cd1_filename)
    cd2_filepath = os.path.join(cd_save_path, cd2_filename)

    with open(cd1_filepath, 'w+') as cd_file:
        cd_file.write(str(float(cd1)))
    with open(cd2_filepath, 'w+') as cd_file:
        cd_file.write(str(float(cd2)))


class CDCalculater:
    def __init__(self, specs, logger):
        self.specs = specs
        self.geometries_path = None
        self.logger = logger

    def caculate_cd(self, pcd_gt, pcd_partial):
        cd = np.array(pcd_gt.compute_point_cloud_distance(pcd_partial), dtype=np.float32)
        cd = sum(cd) / len(cd)
        return cd

    def handle_scene(self, scene):
        self.geometries_path = getGeometriesPath(self.specs, scene)

        pcd1_gt = geometry_utils.read_point_cloud(self.geometries_path.get("pcd1_gt"))
        pcd2_gt = geometry_utils.read_point_cloud(self.geometries_path.get("pcd2_gt"))
        pcd1_partial = geometry_utils.read_point_cloud(self.geometries_path.get("pcd1_partial"))
        pcd2_partial = geometry_utils.read_point_cloud(self.geometries_path.get("pcd2_partial"))

        cd1 = self.caculate_cd(pcd1_gt, pcd1_partial)
        cd2 = self.caculate_cd(pcd2_gt, pcd2_partial)

        save_cd(self.specs, scene, cd1, cd2)


def my_process(scene, specs):
    _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)
    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    cdCalculater = CDCalculater(specs, _logger)

    try:
        cdCalculater.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e.message))
    finally:
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)


if __name__ == '__main__':
    config_filepath = 'configs/calculate_incomplete_degree.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("pcd_partial_dir"))
    path_utils.generate_path(specs.get("path_options").get("incomplete_cd_save_dir"))

    logger = logging.getLogger("calculate_cd")
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

            cdCalculater = CDCalculater(specs, _logger)
            cdCalculater.handle_scene(filename)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)
