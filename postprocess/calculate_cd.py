"""
计算重建出的ibs点云到ibs mesh gt的单向cd
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

    ibs_gt_dir = specs.get("path_options").get("geometries_dir").get("ibs_gt_dir")
    ibs_pcd_dir = specs.get("path_options").get("geometries_dir").get("ibs_pcd_dir")
    IOU_dir = specs.get("path_options").get("geometries_dir").get("IOU_dir")

    ibs_gt_filename = '{}.obj'.format(scene)
    ibs_pcd_filename = '{}.ply'.format(instance_name)
    IOU_filename = '{}.obj'.format(scene)

    geometries_path['ibs_gt'] = os.path.join(ibs_gt_dir, category, ibs_gt_filename)
    geometries_path['ibs_pcd'] = os.path.join(ibs_pcd_dir, category, ibs_pcd_filename)
    geometries_path['aabb'] = os.path.join(IOU_dir, category, IOU_filename)

    return geometries_path


def save_cd(specs, instance_name, cd):
    cd_save_dir = specs.get("path_options").get("cd_save_dir")
    category_patten = specs.get("path_options").get("format_info").get("category_re")

    path_utils.generate_path(cd_save_dir)
    category = re.match(category_patten, instance_name).group()

    cd_save_path = os.path.join(cd_save_dir, category)
    path_utils.generate_path(cd_save_path)

    cd_filename = '{}.txt'.format(instance_name)
    cd_filepath = os.path.join(cd_save_path, cd_filename)

    with open(cd_filepath, 'w+') as cd_file:
        cd_file.write(str(float(cd)))


class CDCalculater:
    def __init__(self, specs, logger):
        self.specs = specs
        self.geometries_path = None
        self.logger = logger

    def is_point_in_aabb(self, point, min_bound, max_bound):
        for i in range(3):
            if point[i] < min_bound[i] or point[i] > max_bound[i]:
                return False
        return True

    def query_dist(self, mesh, points):
        points = o3d.core.Tensor(np.array(points), dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        return scene.compute_distance(points)

    def caculate_cd(self, mesh, pcd, aabb: o3d.geometry.AxisAlignedBoundingBox):
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()

        pcd_np = np.array(pcd.points)
        points_in_aabb = np.array([point for point in pcd_np if self.is_point_in_aabb(point, min_bound, max_bound)])

        if points_in_aabb.shape[0] == 0:
            self.logger.info("no point in aabb")
            return 0

        cd = self.query_dist(mesh, points_in_aabb).numpy()
        cd = sum(cd) / len(cd)
        return cd

    def handle_scene(self, scene):
        self.geometries_path = getGeometriesPath(self.specs, scene)

        ibs_gt = geometry_utils.read_mesh(self.geometries_path.get("ibs_gt"))
        ibs_pcd = geometry_utils.read_point_cloud(self.geometries_path.get("ibs_pcd"))
        aabb = geometry_utils.read_mesh(self.geometries_path.get("aabb")).get_axis_aligned_bounding_box()

        cd = self.caculate_cd(ibs_gt, ibs_pcd, aabb)

        save_cd(self.specs, scene, cd)


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
    config_filepath = 'configs/calculate_cd.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("ibs_pcd_dir"))
    path_utils.generate_path(specs.get("path_options").get("cd_save_dir"))

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
