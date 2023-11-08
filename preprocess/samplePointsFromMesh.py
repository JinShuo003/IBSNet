"""
从mesh上随机散点，采用possion disk算法
"""
import os
import re

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


def getGeometriesPath(specs, scene):
    category_re = specs["category_re"]
    category = re.match(category_re, scene).group()

    geometries_path = dict()

    mesh_dir = specs["mesh_dir"]

    mesh_filename = '{}.obj'.format(scene)

    geometries_path['mesh'] = os.path.join(mesh_dir, category, mesh_filename)

    return geometries_path


def save_pcd(specs, scene, pcd):
    pcd_save_dir = specs['pcd_save_dir']
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

    def handle_scene(self, scene):
        """读取mesh，组合后求取归一化参数，然后分别归一化到单位球内，保存结果"""
        self.geometries_path = getGeometriesPath(self.specs, scene)

        sample_num = self.specs["sample_num"]
        mesh = o3d.io.read_triangle_mesh(self.geometries_path["mesh"])
        pcd = mesh.sample_points_poisson_disk(sample_num)

        save_pcd(self.specs, scene, pcd)


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
    config_filepath = 'config/samplePointsFromMesh.json'
    specs = parseConfig(config_filepath)
    # 构建文件树
    filename_tree = getFilenameTree(specs, "mesh_dir")
    # 处理文件夹，不存在则创建
    generatePath(specs, ["pcd_save_dir"])

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
