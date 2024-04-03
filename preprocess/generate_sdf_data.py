"""
输入两个mesh，以一定规则在空间内生成随机点，并计算这些点的sdf值
"""
import copy
import multiprocessing
import os
import random
import re
import logging

import numpy as np
import open3d as o3d

from utils import randomNum, log_utils, path_utils


class SampleMethodException(Exception):
    def __init__(self, message="Illegal sample method, surface or IOU are supported"):
        self.message = message
        super.__init__(message)


def getGeometriesPath(specs, scene):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    category = re.match(category_re, scene).group()
    scene = re.match(scene_re, scene).group()

    geometries_path = dict()

    mesh_dir = specs.get("path_options").get("geometries_dir").get("mesh_dir")
    IOUgt_dir = specs.get("path_options").get("geometries_dir").get("IOUgt_dir")

    mesh1_filename = '{}_{}.obj'.format(scene, 0)
    mesh2_filename = '{}_{}.obj'.format(scene, 1)
    IOUgt_filename = '{}.npy'.format(scene)

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometries_path['IOUgt'] = os.path.join(IOUgt_dir, category, IOUgt_filename)

    return geometries_path


def save_sdf(specs, sdf_dir, SDF_data, scene):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    category = re.match(category_re, scene).group()

    path_utils.generate_path(os.path.join(sdf_dir, category))

    sdf_filename = '{}.npz'.format(scene)
    sdf_path = os.path.join(sdf_dir, category, sdf_filename)
    np.savez(sdf_path, data=SDF_data)


class IndirectSdfSampleGenerator:
    """生成间接法的sdf数据"""

    def __init__(self, specs):
        self.specs = specs

    def query_dist(self, mesh, points):
        points = o3d.core.Tensor(np.array(points), dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        return scene.compute_distance(points)

    def get_sdf_samples_surface(self, mesh1, mesh2):
        """在mesh1和mesh2表面附近、半径为0.5的球形区域内进行采样"""
        points_num = self.specs["sdf_sample_options_indirect"]["points_num"]
        sample_option = self.specs["sdf_sample_options_indirect"]["surface_sample_option"]
        dist = sample_option["dist"]

        mesh1_num = int(points_num * sample_option["proportion1"])
        mesh2_num = int(points_num * sample_option["proportion2"])
        sphere_num = int(points_num * sample_option["proportion_sphere"])

        mesh1_points = []
        mesh2_points = []
        sphere_points = []

        while len(mesh1_points) < mesh1_num:
            if len(mesh1_points) == 0:
                random_points = randomNum.get_random_points_in_sphere(200000)
            else:
                random_points = randomNum.get_random_points_from_seeds(mesh1_points, 3, dist * 3)
            dists = self.query_dist(mesh1, random_points)
            mesh1_points += [random_points[i] for i in range(len(random_points)) if
                           dists[i] <= dist and np.linalg.norm(random_points[i]) <= 0.5]
        mesh1_points = np.array(random.sample(mesh1_points, mesh1_num))

        while len(mesh2_points) < mesh2_num:
            if len(mesh2_points) == 0:
                random_points = randomNum.get_random_points_in_sphere(200000)
            else:
                random_points = randomNum.get_random_points_from_seeds(mesh2_points, 3, dist * 3)
            dists = self.query_dist(mesh2, random_points)
            mesh2_points += [random_points[i] for i in range(len(random_points)) if
                           dists[i] <= dist and np.linalg.norm(random_points[i]) <= 0.5]
        mesh2_points = np.array(random.sample(mesh2_points, mesh2_num))

        sphere_points = np.array(randomNum.get_random_points_in_sphere(sphere_num))

        return np.concatenate([mesh1_points, mesh2_points, sphere_points], axis=0)

    def get_sdf_samples_IOU(self, aabb1, aabb2, aabb_IOU):
        """在aabb_IOU和全空间内按比例进行采样"""
        points_num = self.specs["sdf_sample_options_indirect"]["points_num"]
        sample_options = self.specs["sdf_sample_options_indirect"]["IOU_sample_option"]
        proportion_aabb1 = sample_options["proportion_aabb1"]
        proportion_aabb2 = sample_options["proportion_aabb2"]
        proportion_IOU = sample_options["proportion_IOU"]
        proportion_other = sample_options["proportion_other"]

        random_points_aabb1 = randomNum.get_random_points_with_limit(aabb1, int(points_num * proportion_aabb1))
        random_points_aabb2 = randomNum.get_random_points_with_limit(aabb2, int(points_num * proportion_aabb2))
        random_points_IOU = randomNum.get_random_points_with_limit(aabb_IOU, int(points_num * proportion_IOU))
        random_points_other = randomNum.get_random_points_in_sphere(int(points_num * proportion_other))
        return np.concatenate([random_points_aabb1, random_points_aabb2, random_points_IOU, random_points_other], axis=0)

    def get_sdf_values(self, mesh1, mesh2, sdf_samples):
        """获取sdf_samples在mesh1，mesh2场内的sdf值，拼接成(x, y, z, sdf1, sdf2)的形式"""
        dists_1 = self.query_dist(mesh1, sdf_samples).numpy().reshape(-1, 1)
        dists_2 = self.query_dist(mesh2, sdf_samples).numpy().reshape(-1, 1)
        SDF_data = np.concatenate([sdf_samples, dists_1, dists_2], axis=1)
        return SDF_data

    def get_indirect_partial_sdf(self, pcd1_partial_list, pcd2_partial_list, sdf_samples):
        """依次计算sdf_samples在pcd1_partial_list，pcd2_partial_list中各个点云对场内的sdf值，返回列表"""
        SDF_partial_data = []
        for i in range(len(pcd1_partial_list)):
            SDF_partial_data.append(self.get_sdf_values(pcd1_partial_list[i], pcd2_partial_list[i], sdf_samples))
        return SDF_partial_data


class TrainDataGenerator:
    def __init__(self, specs, logger=None):
        self.specs = specs
        self.geometries_path = None
        self.mesh1 = None
        self.mesh2 = None
        # aabb框
        self.aabb1 = None
        self.aabb2 = None
        self.aabb_total = None
        self.aabb_IOU = None
        self.aabb1_zoom = None
        self.aabb2_zoom = None
        self.aabb_IOU_zoom = None
        self.aabb_IOU_ibs = None
        self.logger = logger

    def get_mesh(self):
        self.mesh1 = o3d.io.read_triangle_mesh(self.geometries_path["mesh1"])
        self.mesh1.paint_uniform_color((1, 0, 0))
        self.mesh2 = o3d.io.read_triangle_mesh(self.geometries_path["mesh2"])
        self.mesh2.paint_uniform_color((0, 1, 0))

    def merge_aabb(self, aabb1, aabb2):
        min_point = np.minimum(aabb1.min_bound, aabb2.min_bound)
        max_point = np.maximum(aabb1.max_bound, aabb2.max_bound)

        # 创建合并后的AABB框
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=min_point, max_bound=max_point)

    def get_aabb(self):
        """获取mesh的aabb框"""
        self.aabb1 = self.mesh1.get_axis_aligned_bounding_box()
        self.aabb1.color = (1, 0, 0)
        self.aabb2 = self.mesh2.get_axis_aligned_bounding_box()
        self.aabb2.color = (0, 1, 0)
        self.aabb_total = self.merge_aabb(self.aabb1, self.aabb2)
        self.aabb_total.color = (0, 0, 1)

        scale_1 = self.specs["sdf_sample_options_indirect"]["IOU_sample_option"]["scale_aabb1"]
        scale_2 = self.specs["sdf_sample_options_indirect"]["IOU_sample_option"]["scale_aabb2"]
        self.aabb1_zoom = copy.deepcopy(self.aabb1)
        self.aabb1_zoom.scale(scale_1, np.array([0, 0, 0]))
        self.aabb2_zoom = copy.deepcopy(self.aabb2)
        self.aabb2_zoom.scale(scale_2, np.array([0, 0, 0]))

    def get_IOU(self):
        import copy
        aabb_data = np.load(self.geometries_path["IOUgt"])
        min_bound = aabb_data[0]
        max_bound = aabb_data[1]
        self.aabb_IOU = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        self.aabb_IOU.color = (1, 0, 0)
        self.aabb_IOU_zoom = copy.deepcopy(self.aabb_IOU)
        self.aabb_IOU_zoom.scale(self.specs["sdf_sample_options_indirect"]["IOU_sample_option"]["scale_IOU"],
                                 self.aabb_IOU_zoom.get_center())
        self.aabb_IOU_zoom.color = (0, 1, 0)

    def aabb2pcd(self, aabb):
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()

        # 构建八个顶点的坐标
        vertices = [
            [min_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]]
        ]

        # 将顶点坐标放入NumPy数组
        vertices_np = np.array(vertices, dtype=np.float32)
        return vertices_np

    def get_init_geometries(self):
        """获取初始几何体，包括mesh、点云、aabb框、IOU标注框等"""
        self.get_mesh()
        self.get_aabb()
        self.get_IOU()

    def get_sdf_data(self):
        """获取间接法的数据，即完整点云下的sdf场(x, y, z, sdf1, sdf2)"""
        method = self.specs["sdf_sample_options_indirect"]["method"]
        idSdfGenerator = IndirectSdfSampleGenerator(self.specs)
        if method == "surface":
            sdf_samples_indirect = idSdfGenerator.get_sdf_samples_surface(self.mesh1, self.mesh2)
        elif method == "IOU":
            sdf_samples_indirect = idSdfGenerator.get_sdf_samples_IOU(self.aabb1_zoom, self.aabb2_zoom, self.aabb_IOU_zoom)
        else:
            raise SampleMethodException
        sdf_data_indirect = idSdfGenerator.get_sdf_values(self.mesh1, self.mesh2, sdf_samples_indirect)
        return sdf_data_indirect

    def visualize_result(self, sdf_data_indirect):
        coor = self.geometryHandler.get_unit_coordinate()
        sphere = self.geometryHandler.get_unit_sphere_pcd()

        surface_points_ibs = [points[0:3] for points in sdf_data_indirect if abs(points[3] - points[4]) < 0.005]
        sdf_samples_indirect_pcd = o3d.geometry.PointCloud()
        sdf_samples_indirect_pcd.points = o3d.utility.Vector3dVector(surface_points_ibs)
        sdf_samples_indirect_pcd.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries(
            [sdf_samples_indirect_pcd, self.mesh1, self.mesh2, self.aabb_IOU_zoom, coor, sphere],
            window_name="indirect")

    def handle_scene(self, scene):
        self.geometries_path = getGeometriesPath(self.specs, scene)
        self.get_init_geometries()

        sdf_data = self.get_sdf_data()

        save_sdf(self.specs, self.specs.get("path_options").get("sdf_data_save_dir"), sdf_data, scene)


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
    config_filepath = 'configs/generate_sdf_data.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("sdf_data_save_dir"))

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
