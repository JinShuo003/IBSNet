"""
从mesh形式的IBS获取点云形式的IBS，用标注框进行截断
"""
import os
import re
import numpy as np
from ordered_set import OrderedSet
import multiprocessing
import json
import open3d as o3d


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def getFilenameTree(specs: dict, base_path: str):
    """以base_path为基准构建文件树，文件树的格式为
    {
    'scene1': {
              'scene1.1000' :[scene1.1000_view0_0,
                              scene1.1000_view0_1],
              'scene1.1001' :[scene1.1001_view0_0,
                              scene1.1001_view0_1]
              }
    }
    """
    # 构建文件树
    base_path = specs[base_path]

    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    filename_re = specs["filename_re"]

    handle_category = specs["handle_category"]
    handle_scene = specs["handle_scene"]
    handle_filename = specs["handle_filename"]

    filename_tree = dict()
    folder_info = os.walk(base_path)
    for dir_path, dir_names, filenames in folder_info:
        # 顶级目录不做处理
        if dir_path == base_path:
            continue
        category = dir_path.split('\\')[-1]
        if not regular_match(handle_category, category):
            continue
        if category not in filename_tree:
            filename_tree[category] = dict()
        for filename in filenames:
            scene = re.match(scene_re, filename).group()
            if not regular_match(handle_scene, scene):
                continue
            if scene not in filename_tree[category]:
                filename_tree[category][scene] = OrderedSet()
            filename = re.match(filename_re, filename).group()
            if not regular_match(handle_filename, filename):
                continue
            filename_tree[category][scene].add(filename)
    # 将有序集合转为列表
    filename_tree_copy = dict()
    for category in filename_tree.keys():
        for scene in filename_tree[category]:
            filename_tree[category][scene] = list(filename_tree[category][scene])
            if len(filename_tree[category][scene]) != 0:
                if category not in filename_tree_copy.keys():
                    filename_tree_copy[category] = dict()
                filename_tree_copy[category][scene] = filename_tree[category][scene]
    return filename_tree_copy


def generatePath(specs: dict, path_list: list):
    """检查specs中的path是否存在，不存在则创建"""
    for path in path_list:
        if not os.path.isdir(specs[path]):
            os.makedirs(specs[path])


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)


def getGeometryPath(specs, filename):
    scene_re = specs["scene_re"]
    category_re = specs["category_re"]
    filename_re = specs["filename_re"]
    category = re.match(category_re, filename).group()
    scene = re.match(scene_re, filename).group()
    filename = re.match(filename_re, filename).group()

    geometry_path = dict()

    ibs_mesh_dir = specs['ibs_mesh_dir']
    IOU_dir = specs['IOU_dir']

    ibs_mesh_filename = '{}.obj'.format(filename)
    IOU_filename = '{}.obj'.format(scene)

    geometry_path['ibs_mesh'] = os.path.join(ibs_mesh_dir, category, ibs_mesh_filename)
    geometry_path['IOU'] = os.path.join(IOU_dir, category, IOU_filename)

    return geometry_path


def save_pcd(specs, scene, pcd):
    pcd_save_dir = specs['ibs_pcd_save_dir']
    category = re.match(specs['category_re'], scene).group()
    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_save_dir, category)):
        os.makedirs(os.path.join(pcd_save_dir, category))

    pcd_filename = '{}.ply'.format(scene)
    pcd_path = os.path.join(pcd_save_dir, category, pcd_filename)

    o3d.io.write_point_cloud(pcd_path, pcd)


class TrainDataGenerator:
    def __init__(self, specs):
        self.specs = specs
        self.geometries_path = None

    def read_mesh(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        return mesh

    def sample_pcd(self, mesh, bound):
        sample_num = self.specs["sample_num"]
        min_bound = bound.get_min_bound()
        max_bound = bound.get_max_bound()
        points = np.array([])
        while len(points) < sample_num:
            points_cur = np.asarray(mesh.sample_points_poisson_disk(sample_num).points)
            points_cur = [point for point in points_cur if self.is_point_in_volume(point, min_bound, max_bound)]
            points_cur = np.array(points_cur)
            if len(points) == 0:
                points = points_cur
            else:
                points = np.concatenate((points, points_cur), axis=0)
            points = np.unique(points, axis=0)
        points = points[np.random.choice(points.shape[0], size=sample_num, replace=False)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def is_point_in_volume(self, point, min_bound, max_bound):
        for i in range(3):
            if point[i] < min_bound[i] or point[i] > max_bound[i]:
                return False
        return True

    def handle_scene(self, scene):
        """读取mesh，组合后求取归一化参数，然后分别归一化到单位球内，保存结果"""
        self.geometries_path = getGeometryPath(self.specs, scene)

        ibs_mesh = self.read_mesh(self.geometries_path["ibs_mesh"])
        IOU = self.read_mesh(self.geometries_path["IOU"])
        IOU.scale(1.5, IOU.get_center())
        ibs_pcd = self.sample_pcd(ibs_mesh, IOU)
        # o3d.visualization.draw_geometries([ibs_mesh, ibs_pcd], mesh_show_wireframe=True, mesh_show_back_face=True)

        save_pcd(self.specs, scene, ibs_pcd)


def my_process(scene, specs):
    # 获取当前进程信息
    process_name = multiprocessing.current_process().name
    # 执行任务函数的逻辑
    print(f"Running task in process: {process_name}, scene: {scene}")
    # 其他任务操作
    trainDataGenerator = TrainDataGenerator(specs)
    try:
        trainDataGenerator.handle_scene(scene)
        print(f"scene: {scene} succeed")
    except:
        print(f"scene: {scene} failed")


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/getIBSPCDFromMesh.json'
    specs = parseConfig(config_filepath)
    # 构建文件树
    filename_tree = getFilenameTree(specs, "ibs_mesh_dir")
    # 处理文件夹，不存在则创建
    generatePath(specs, ["ibs_pcd_save_dir"])

    # 创建进程池，指定进程数量
    pool = multiprocessing.Pool(processes=10)
    # 参数
    file_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                file_list.append(filename)
    # 使用进程池执行任务，返回结果列表
    for file in file_list:
        pool.apply_async(my_process, (file, specs,))

    # 关闭进程池
    pool.close()
    pool.join()
