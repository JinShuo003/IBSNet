"""
输入两点云，计算IBS，不进行穿插检测和迭代
"""
import logging
import multiprocessing
import os
import re

import open3d as o3d

from utils import geometry_utils, path_utils, ibs_utils, log_utils


def get_geometries_path(specs, filename):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    category = re.match(category_re, filename).group()

    geometries_path = dict()

    pcd_dir = specs.get("path_options").get("geometries_dir").get("pcd_dir")

    pcd1_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_filename = '{}_{}.ply'.format(filename, 1)

    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)

    return geometries_path


def save_ibs_mesh(specs, scene, ibs_mesh_o3d):
    mesh_dir = specs.get("path_options").get("ibs_mesh_save_dir")
    category = re.match(specs.get("path_options").get("format_info").get("category_re"), scene).group()
    # mesh_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    ibs_mesh_filename = '{}.obj'.format(scene)
    mesh_path = os.path.join(mesh_dir, category, ibs_mesh_filename)
    o3d.io.write_triangle_mesh(mesh_path, ibs_mesh_o3d)


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger

    def get_ibs_mesh_o3d(self, geometries_path: dict):
        clip_border_type = self.specs.get("caculate_options").get("clip_border_options").get("clip_border_type")
        clip_sphere_radius = self.specs.get("caculate_options").get("clip_border_options").get("clip_sphere_radius")
        clip_border_magnification = self.specs.get("caculate_options").get("clip_border_options").get("clip_border_magnification")

        pcd1 = geometry_utils.read_point_cloud(geometries_path["pcd1"])
        pcd2 = geometry_utils.read_point_cloud(geometries_path["pcd2"])

        ibs = ibs_utils.IBS(pcd1=pcd1,
                            pcd2=pcd2,
                            mode="pcd",
                            clip_border_type=clip_border_type,
                            clip_sphere_radius=clip_sphere_radius,
                            clip_border_magnification=clip_border_magnification,
                            logger=self.logger)
        ibs.launch()
        ibs_o3d = ibs.get_ibs_o3d()

        return ibs_o3d

    def handle_scene(self, scene):
        geometries_path = get_geometries_path(self.specs, scene)
        ibs_mesh_o3d = self.get_ibs_mesh_o3d(geometries_path)
        save_ibs_mesh(self.specs, scene, ibs_mesh_o3d)


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
    config_filepath = 'configs/get_IBS_from_pcd.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("pcd_dir"))
    path_utils.generate_path(specs.get("path_options").get("ibs_mesh_save_dir"))

    logger = logging.getLogger("get_IBS")
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
