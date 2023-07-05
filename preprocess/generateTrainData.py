"""
1. 从mesh进行采样，得到完整点云和归一化参数
2. 利用归一化参数在交互区域内和两物体点云的aabb框内随机散点
3. 计算这些点在两物体完整点云sdf场内的sdf值，该值作为间接法网络的gt
4. 保留这些点中sdf之差小于阈值的那部分点（认为位于ibs面上），然后计算所有点在ibs面的sdf场内的sdf值，该值作为直接法网络的gt
5. 从n个视角对完整点云进行扫描，得到n个角度的残缺点云数据，作为网络的输入
6. 计算这些点在两物体残缺点云sdf场内的sdf值，作为对比数据
"""
import copy
import math
import os
import re
import multiprocessing
import open3d as o3d
import numpy as np
from ordered_set import OrderedSet
import random
import json
from utils import randomNum


class SampleMethodException(Exception):
    def __init__(self, message="Illegal sample method, surface or IOU are supported"):
        self.message = message
        super.__init__(message)


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def getFilenameTree(specs: dict, base_path: str):
    """以base_path为基准构建文件树，文件树的格式为
    {
    'scene1': ['scene1.1000',
               'scene1.1001'],
    'scene2': ['scene1.1011',
               'scene1.1012']}
    """
    # 构建文件树
    base_path = specs[base_path]
    scene_re = specs["scene_re"]

    filename_tree = dict()
    folder_info = os.walk(base_path)
    for dir_path, dir_names, filenames in folder_info:
        # 顶级目录不做处理
        if dir_path == base_path:
            continue
        category = dir_path.split('\\')[-1]
        if not regular_match(specs["category_re"], category):
            continue
        if not category in filename_tree:
            filename_tree[category] = OrderedSet()
        for filename in filenames:
            if not regular_match(specs["scene_re"], filename):
                continue
            filename = re.match(scene_re, filename).group()
            filename_tree[category].add(filename)
    # 将有序集合转为列表
    for key in filename_tree.keys():
        filename_tree[key] = list(filename_tree[key])
    return filename_tree


def generatePath(specs: dict, path_list: list):
    """检查specs中的path是否存在，不存在则创建"""
    for path in path_list:
        if not os.path.isdir(specs[path]):
            os.makedirs(specs[path])


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)


def getGeometriesPath(specs, scene):
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    category = re.match(category_re, scene).group()
    scene = re.match(scene_re, scene).group()

    geometries_path = dict()

    mesh_dir = specs["mesh_dir"]
    ibs_dir = specs["ibs_dir"]
    IOUgt_dir = specs["IOUgt_dir"]

    mesh1_filename = '{}_{}.obj'.format(scene, 0)
    mesh2_filename = '{}_{}.obj'.format(scene, 1)
    ibs_filename = '{}.obj'.format(scene)
    IOUgt_filename = '{}.npy'.format(scene)

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometries_path['ibs'] = os.path.join(ibs_dir, category, ibs_filename)
    geometries_path['IOUgt'] = os.path.join(IOUgt_dir, category, IOUgt_filename)

    return geometries_path


def save_pcd(specs, pcd1_list, pcd2_list, view_index_list, scene):
    pcd_dir = specs['pcd_partial_save_dir']
    category = re.match(specs['category_re'], scene).group()

    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_dir, category)):
        os.makedirs(os.path.join(pcd_dir, category))

    for i in range(len(pcd1_list)):
        # 获取点云名
        pcd1_filename = '{}_view{}_0.ply'.format(scene, view_index_list[i])
        pcd2_filename = '{}_view{}_1.ply'.format(scene, view_index_list[i])

        # 保存点云
        pcd1_path = os.path.join(pcd_dir, category, pcd1_filename)
        if os.path.isfile(pcd1_path):
            os.remove(pcd1_path)
        pcd2_path = os.path.join(pcd_dir, category, pcd2_filename)
        if os.path.isfile(pcd2_path):
            os.remove(pcd2_path)
        o3d.io.write_point_cloud(pcd1_path, pcd1_list[i])
        o3d.io.write_point_cloud(pcd2_path, pcd2_list[i])


def save_sdf(specs, sdf_dir, SDF_data, scene, view_index):
    category = re.match(specs['category_re'], scene).group()

    # 目录不存在则创建
    if not os.path.isdir(os.path.join(sdf_dir, category)):
        os.mkdir(os.path.join(sdf_dir, category))

    # 将data写入文件
    for i, scan_index in enumerate(view_index):
        sdf_filename = '{}_view{}.npz'.format(scene, scan_index)
        sdf_path = os.path.join(sdf_dir, category, sdf_filename)
        if os.path.isfile(sdf_path):
            print('sdf file exsit')
            os.remove(sdf_path)
        if isinstance(SDF_data, list):
            np.savez(sdf_path, data=SDF_data[i])
        else:
            np.savez(sdf_path, data=SDF_data)


def save_ibs_mesh(specs, scene, ibs_mesh_o3d):
    mesh_dir = specs['ibs_save_dir']
    category = re.match(specs['category_re'], scene).group()
    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    ibs_filename = '{}.obj'.format(scene)
    pcd1_path = os.path.join(mesh_dir, category, ibs_filename)
    o3d.io.write_triangle_mesh(pcd1_path, ibs_mesh_o3d)


# ----------------------------------------其他-------------------------------------------

class GeometryHandler:
    def __init__(self):
        pass

    def get_pcd_normalize_para(self, pcd):
        pcd_np = np.asarray(pcd.points)
        # 求点云的中心
        centroid = np.mean(pcd_np, axis=0)
        # 求长轴长度
        scale = np.max(np.sqrt(np.sum(pcd_np ** 2, axis=1)))
        return centroid, scale

    def geometry_transform(self, geometry, centroid, scale):
        coor = self.get_unit_coordinate()
        geometry.translate(-centroid)
        geometry.scale(1 / scale, np.array([0, 0, 0]))

    def get_unit_sphere(self):
        # 创建单位球点云
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.compute_vertex_normals()
        return mesh_sphere

    def get_unit_coordinate(self):
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord_frame.compute_vertex_normals()
        return coord_frame

    def get_unit_sphere_pcd(self):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        pcd = mesh_sphere.sample_points_uniformly(256)
        return pcd


class ScanPcdGenerator:
    """按照specs中的配置对mesh1和mesh2进行单角度扫描，得到若干个视角的单角度残缺点云"""

    def __init__(self, specs, visualize, mesh1, mesh2):
        self.specs = specs
        self.visualize = visualize
        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.mesh1_triangels = np.asarray(mesh1.triangles)
        self.mesh2_triangels = np.asarray(mesh2.triangles)
        self.mesh1_vertices = np.asarray(mesh1.vertices)
        self.mesh2_vertices = np.asarray(mesh2.vertices)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.get_ray_casting_scene()

    def get_ray_casting_scene(self):
        """初始化光线追踪场景"""
        mesh1_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh1)
        mesh2_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh2)
        self.scene.add_triangles(mesh1_t)
        self.scene.add_triangles(mesh2_t)

    def get_rays(self, theta, camera_height, r, fov_deg):
        """获取当前视角的光线"""
        eye = [r * math.cos(theta), camera_height, r * math.sin(theta)]
        rays = self.scene.create_rays_pinhole(fov_deg=fov_deg,
                                              center=[0, 0, 0],
                                              eye=eye,
                                              up=[0, 1, 0],
                                              width_px=self.specs["scan_options"]["width_px"],
                                              height_px=self.specs["scan_options"]["height_px"])
        return rays

    def visualize_rays(self, rays):
        # 可视化光线和mesh
        if self.visualize:
            cast_rays = self.get_rays_visualization(rays)
            cast_rays_single = self.get_rays_visualization_single_view(rays)
            mesh_sphere = GeometryHandler().get_unit_sphere_pcd()
            o3d.visualization.draw_geometries([cast_rays, cast_rays_single, mesh_sphere, self.mesh1, self.mesh2])

    def get_ray_cast_result(self, rays):
        return self.scene.cast_rays(rays)

    def generate_scan_pcd(self):
        scan_options = self.specs["scan_options"]
        scan_num = scan_options["scan_num"]
        camera_height = scan_options["camera_height"]
        camera_ridius = scan_options["camera_ridius"]
        fov_deg = scan_options["fov_deg"]

        pcd1_partial_list = []
        pcd2_partial_list = []
        scan_view_list = []
        for i in range(scan_options["scan_num"]):
            rays = self.get_rays(theta=2 * math.pi * i / scan_num, camera_height=camera_height, r=camera_ridius,
                                 fov_deg=fov_deg)
            self.visualize_rays(rays)
            cast_result = self.get_ray_cast_result(rays)
            # 根据光线投射结果获取当前角度的残缺点云
            pcd1_scan, pcd2_scan = self.get_cur_view_pcd(rays, cast_result)
            # 两者都采样成功则保存，并记录视角下标，否则丢弃
            if pcd1_scan and pcd2_scan:
                pcd1_partial_list.append(pcd1_scan)
                pcd2_partial_list.append(pcd2_scan)
                scan_view_list.append(i)
        return pcd1_partial_list, pcd2_partial_list, scan_view_list

    def get_cur_view_pcd(self, rays, cast_result):
        hit = cast_result['t_hit'].numpy()
        geometry_ids = cast_result["geometry_ids"].numpy()
        primitive_ids = cast_result["primitive_ids"].numpy()
        primitive_uvs = cast_result["primitive_uvs"].numpy()

        points_pcd1 = []
        points_pcd2 = []

        # 获取光线击中的点
        for i in range(rays.shape[0]):
            for j in range(rays.shape[1]):
                if not math.isinf(hit[i][j]):
                    if geometry_ids[i][j] == 0:
                        points_pcd1.append(
                            self.get_real_coordinate(self.mesh1_vertices, self.mesh1_triangels[primitive_ids[i][j]],
                                                     primitive_uvs[i][j]))
                    if geometry_ids[i][j] == 1:
                        points_pcd2.append(
                            self.get_real_coordinate(self.mesh2_vertices, self.mesh2_triangels[primitive_ids[i][j]],
                                                     primitive_uvs[i][j]))

        pcd_sample_options = self.specs["PCD_sample_options"]
        pcd1_scan = o3d.geometry.PointCloud()
        pcd2_scan = o3d.geometry.PointCloud()
        pcd1_scan.points = o3d.utility.Vector3dVector(points_pcd1)
        pcd2_scan.points = o3d.utility.Vector3dVector(points_pcd2)
        try:
            pcd1_scan = pcd1_scan.farthest_point_down_sample(pcd_sample_options["number_of_points"])
            pcd2_scan = pcd2_scan.farthest_point_down_sample(pcd_sample_options["number_of_points"])
            pcd1_scan.paint_uniform_color((1, 0, 0))
            pcd2_scan.paint_uniform_color((0, 1, 0))
        except:
            pcd1_scan = None
            pcd2_scan = None
            print('vertices not enough, sample failed')

        return pcd1_scan, pcd2_scan

    def get_real_coordinate(self, vertices, triangles, uv_coordinate):
        # 将三角形的重心坐标变换为真实坐标
        point1 = vertices[triangles[0]]
        point2 = vertices[triangles[1]]
        point3 = vertices[triangles[2]]
        return uv_coordinate[0] * point1 + uv_coordinate[1] * point2 + (
                1 - uv_coordinate[0] - uv_coordinate[1]) * point3

    def get_rays_visualization(self, rays):
        """获取所有的光线"""
        rays_ = rays.numpy()
        eye = rays_[0][0][0:3].reshape(1, 3)
        rays_ = rays_[:, :, 3:6]
        rays_points = eye
        for i in range(rays_.shape[0]):
            rays_points = np.concatenate((rays_points, rays_[i]))
        lines = [[0, i] for i in range(1, rays_points.shape[0] - 1)]
        colors = [[1, 0, 0] for i in range(lines.__sizeof__())]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(rays_points)

        return lines_pcd

    def get_rays_visualization_single_view(self, rays):
        """获取eye到坐标原点的连线，表示当前视角的方向向量"""
        eye = rays.numpy()[0][0][0:3].reshape(1, 3)
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector([[0, 1]])
        lines_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.array([[0., 0., 0.]]), eye)))
        return lines_pcd


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
            # 获取随机点，如果是初次采样则先生成比较多的点，留下seed点，否则从seed点出发进行扩充
            if len(mesh1_points) == 0:
                random_points = randomNum.get_random_points_in_sphere(200000)
            else:
                random_points = randomNum.get_random_points_from_seeds(mesh1_points, 3, dist * 3)
            # 计算随机点到ibs的距离
            dists = self.query_dist(mesh1, random_points)
            # 保留满足筛选条件的点
            mesh1_points += [random_points[i] for i in range(len(random_points)) if
                           dists[i] <= dist and np.linalg.norm(random_points[i]) <= 0.5]
        mesh1_points = np.array(random.sample(mesh1_points, mesh1_num))

        while len(mesh2_points) < mesh2_num:
            # 获取随机点，如果是初次采样则先生成比较多的点，留下seed点，否则从seed点出发进行扩充
            if len(mesh2_points) == 0:
                random_points = randomNum.get_random_points_in_sphere(200000)
            else:
                random_points = randomNum.get_random_points_from_seeds(mesh2_points, 3, dist * 3)
            # 计算随机点到ibs的距离
            dists = self.query_dist(mesh2, random_points)
            # 保留满足筛选条件的点
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


class DirectSdfSampleGenerator:
    """生成直接法的sdf数据"""

    def __init__(self, specs):
        self.specs = specs
        self.sdf_sample_option = self.specs["sdf_sample_options_direct"]

    def query_dist(self, mesh, points):
        points = o3d.core.Tensor(np.array(points), dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        return scene.compute_distance(points)

    def get_sdf_samples(self, ibs_mesh):
        """在ibs附近和全空间范围内按比例进行采样"""
        sample_options = self.specs["sdf_sample_options_direct"]
        points_num = sample_options["points_num"]
        points_num_ibs = int(points_num * sample_options["proportion_ibs"])
        points_num_other = int(points_num * sample_options["proportion_other"])
        clamp_dist = sample_options["clamp_dist"]
        points_ibs = []
        points_other = []

        # 生成随机点，保留落在ibs面一定距离内的点，直到采集够点数为止
        while len(points_ibs) < points_num_ibs:
            # 获取随机点，如果是初次采样则先生成比较多的点，留下seed点，否则从seed点出发进行扩充
            if len(points_ibs) == 0:
                random_points = randomNum.get_random_points_in_sphere(200000)
            else:
                random_points = randomNum.get_random_points_from_seeds(points_ibs, 3, clamp_dist * 3)
            # 计算随机点到ibs的距离
            dists = self.query_dist(ibs_mesh, random_points)
            # 保留满足筛选条件的点
            points_ibs += [random_points[i] for i in range(len(random_points)) if
                           dists[i] <= clamp_dist and np.linalg.norm(random_points[i]) <= 0.5]
        points_ibs = np.array(random.sample(points_ibs, points_num_ibs))
        points_other = randomNum.get_random_points_in_sphere(points_num_other)

        return np.concatenate([points_ibs, points_other], axis=0)

    def get_sdf_values(self, ibs, sdf_samples):
        """获取sdf_samples在pcd1，pcd2场内的sdf值，拼接成(x, y, z, sdf1, sdf2)的形式"""
        dists = self.query_dist(ibs, sdf_samples).numpy().reshape(-1, 1)
        SDF_data = np.concatenate([sdf_samples, dists], axis=1)
        return SDF_data


class TrainDataGenerator:
    def __init__(self, specs):
        self.geometryHandler = GeometryHandler()
        self.specs = specs
        self.geometries_path = None
        # mesh
        self.mesh1 = None
        self.mesh2 = None
        self.ibs_mesh = None
        # 原始点云，从mesh采样得到
        self.pcd1 = None
        self.pcd2 = None
        # aabb框
        self.aabb1 = None
        self.aabb2 = None
        self.aabb_total = None
        self.aabb_IOU = None
        self.aabb1_zoom = None
        self.aabb2_zoom = None
        self.aabb_IOU_zoom = None
        self.aabb_IOU_ibs = None
        # 归一化参数
        self.centroid = None
        self.scale = None
        self.normalize_geometries = []
        # 单视角残缺点云及视角号列表
        self.pcd1_partial_list = []
        self.pcd2_partial_list = []
        self.scan_view_list = []

    def get_mesh(self):
        """读取mesh文件"""
        self.mesh1 = o3d.io.read_triangle_mesh(self.geometries_path["mesh1"])
        self.mesh1.compute_triangle_normals()
        self.mesh1.compute_vertex_normals()
        self.mesh1.paint_uniform_color((1, 0, 0))
        self.mesh2 = o3d.io.read_triangle_mesh(self.geometries_path["mesh2"])
        self.mesh2.compute_triangle_normals()
        self.mesh2.compute_vertex_normals()
        self.mesh2.paint_uniform_color((0, 1, 0))
        self.ibs_mesh = o3d.io.read_triangle_mesh(self.geometries_path["ibs"])
        self.ibs_mesh.compute_triangle_normals()
        self.ibs_mesh.compute_vertex_normals()
        self.ibs_mesh.paint_uniform_color((0, 0, 1))

    def get_pcd(self):
        """从mesh采样得到点云"""
        pcd_sample_options = self.specs["PCD_sample_options"]
        self.pcd1 = self.mesh1.sample_points_uniformly(number_of_points=pcd_sample_options["number_of_points"])
        self.pcd1.paint_uniform_color((0, 0, 1))
        self.pcd2 = self.mesh2.sample_points_uniformly(number_of_points=pcd_sample_options["number_of_points"])
        self.pcd2.paint_uniform_color((0, 1, 0))

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
        self.get_pcd()
        self.get_aabb()
        self.get_IOU()

    def paint_geometries(self):
        geometryHandler = GeometryHandler()
        unit_sphere = geometryHandler.get_unit_sphere()
        unit_coordinate = geometryHandler.get_unit_coordinate()
        o3d.visualization.draw_geometries(
            [unit_coordinate, self.ibs_mesh, self.mesh1, self.mesh2, self.pcd1, self.pcd2, self.aabb_IOU,
             self.aabb_IOU_zoom])

    def get_direct_sdf_data(self, ibs_mesh_o3d):
        """获取直接法的数据，即完整点云下的sdf场(x, y, z, sdf)"""
        dSdfGenerator = DirectSdfSampleGenerator(self.specs)
        sdf_samples_direct = dSdfGenerator.get_sdf_samples(ibs_mesh_o3d)
        sdf_data_direct = dSdfGenerator.get_sdf_values(ibs_mesh_o3d, sdf_samples_direct)
        return sdf_data_direct

    def get_indirect_sdf_data(self):
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

    def get_scan_pcd(self):
        """获取残缺点云数据，即完整点云下的sdf场(x, y, z, sdf1, sdf2)"""
        scanPcdGenerator = ScanPcdGenerator(self.specs, self.specs["visualize"], self.mesh1, self.mesh2)
        return scanPcdGenerator.generate_scan_pcd()

    def visualize_result(self, sdf_data_direct, sdf_data_indirect, pcd1_partial_list, pcd2_partial_list,
                         scan_view_list):
        coor = self.geometryHandler.get_unit_coordinate()
        sphere = self.geometryHandler.get_unit_sphere_pcd()
        # 可视化-直接法
        surface_points_ibs = [points[0:3] for points in sdf_data_direct if points[3] < 0.005]
        sdf_samples_direct_pcd = o3d.geometry.PointCloud()
        sdf_samples_direct_pcd.points = o3d.utility.Vector3dVector(surface_points_ibs)
        sdf_samples_direct_pcd.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([sdf_samples_direct_pcd, self.mesh1, self.mesh2, coor, sphere],
                                          window_name="direct")

        # 可视化-间接法
        surface_points_ibs = [points[0:3] for points in sdf_data_indirect if abs(points[3] - points[4]) < 0.005]
        sdf_samples_indirect_pcd = o3d.geometry.PointCloud()
        sdf_samples_indirect_pcd.points = o3d.utility.Vector3dVector(surface_points_ibs)
        sdf_samples_indirect_pcd.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries(
            [sdf_samples_indirect_pcd, self.mesh1, self.mesh2, self.aabb_IOU_zoom, coor, sphere],
            window_name="indirect")

        for i in range(len(pcd1_partial_list)):
            o3d.visualization.draw_geometries(
                [sdf_samples_direct_pcd, pcd1_partial_list[i], pcd2_partial_list[i], self.aabb_IOU_zoom, coor, sphere],
                window_name="{}".format(scan_view_list[i]))

    def handle_scene(self, scene):
        """处理当前场景，包括采集多角度的残缺点云、计算直接法和间接法网络的sdf gt、计算残缺点云下的ibs"""
        # ------------------------------获取点云数据，包括完整点云和各个视角的残缺点云--------------------------
        self.geometries_path = getGeometriesPath(self.specs, scene)
        self.get_init_geometries()

        # 直接法(x, y, z, sdf)
        print("begin generate direct data")
        sdf_data_direct = self.get_direct_sdf_data(self.ibs_mesh)

        # 间接法(x, y, z, sdf1, sdf2)
        print("begin generate indirect data")
        sdf_data_indirect = self.get_indirect_sdf_data()

        # 单角度扫描点云
        print("begin generate scan pointcloud")
        pcd1_partial_list, pcd2_partial_list, scan_view_list = self.get_scan_pcd()

        # # 计算采样点在各个角度残缺点云下的sdf
        # SDF_indirect_partial = get_indirect_partial_sdf(pcd1_list, pcd2_list, sdf_samples)

        if self.specs["visualize"]:
            self.visualize_result(sdf_data_direct, sdf_data_indirect, pcd1_partial_list, pcd2_partial_list,
                                  scan_view_list)
        # 保存直接法SDF
        save_sdf(self.specs, self.specs["sdf_direct_save_dir"], sdf_data_direct, scene, scan_view_list)
        # 保存间接法SDF
        save_sdf(self.specs, self.specs["sdf_indirect_complete_save_dir"], sdf_data_indirect, scene, scan_view_list)
        # 保存残缺点云
        save_pcd(self.specs, pcd1_partial_list, pcd2_partial_list, scan_view_list, scene)
        # # 保存残缺点云ibs_pcd
        # save_sdf(specs, specs["sdf_indirect_partial_save_dir"], SDF_indirect_partial, scene, view_index_list)


def my_process(scene, specs):
    process_name = multiprocessing.current_process().name
    print(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs)

    try:
        trainDataGenerator.handle_scene(scene)
        print("scene: {} succeed".format(scene))
    except Exception as e:
        print("scene: {} failed, exception message: {}".format(scene, e.message))


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/generateTrainData.json'
    specs = parseConfig(config_filepath)
    processNum = specs["process_num"]
    # 构建文件树
    filename_tree = getFilenameTree(specs, "mesh_dir")
    # 处理文件夹，不存在则创建
    generatePath(specs, ["pcd_partial_save_dir", "sdf_indirect_complete_save_dir", "sdf_indirect_partial_save_dir",
                         "sdf_direct_save_dir"])

    # 创建进程池，指定进程数量
    pool = multiprocessing.Pool(processes=processNum)
    # 参数
    scene_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            scene_list.append(scene)
    # 使用进程池执行任务，返回结果列表
    for scene in scene_list:
        pool.apply_async(my_process, (scene, specs,))

    # 关闭进程池
    pool.close()
    pool.join()
