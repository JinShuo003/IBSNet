"""
从sdf获取ibs，具体做法是按照sdf值之差的绝对值排序，取绝对值最小且在标注框内的前k个点
"""
import numpy as np
import multiprocessing
import open3d as o3d

from utils.path_utils import *


def getGeometriesPath(specs, filename):
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]

    category = re.match(category_re, filename).group()
    scene = re.match(scene_re, filename).group()

    geometries_path = dict()

    sdf_dir = specs["sdf_dir"]
    IOU_dir = specs['IOU_dir']

    sdf_filename = '{}.npz'.format(filename)
    IOU_filename = '{}.obj'.format(scene)

    geometries_path['sdf'] = os.path.join(sdf_dir, category, sdf_filename)
    geometries_path['IOU'] = os.path.join(IOU_dir, category, IOU_filename)

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

    def read_sdf_data(self):
        sdf_path = self.geometries_path["sdf"]
        npz = np.load(sdf_path)
        data = npz["data"]
        return data

    def read_mesh(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        return mesh

    def sample_points_from_sdf(self, sdf_data, bound):
        sample_num = self.specs["sample_num"]
        min_bound = bound.get_min_bound()
        max_bound = bound.get_max_bound()

        sdf_subtract = np.abs(sdf_data[:, 3] - sdf_data[:, 4])
        sdf_data = np.concatenate((sdf_data[:, 0:3], sdf_subtract.reshape(-1, 1)), axis=-1)
        sdf_data = sdf_data[np.argsort(sdf_data[:, -1])]

        points = []
        for point_info in sdf_data:
            point = point_info[0:3]
            if self.is_point_in_volume(point, min_bound, max_bound):
                points.append(point)
            if len(points) == sample_num:
                break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color((1, 0, 0))
        return pcd

    def is_point_in_volume(self, point, min_bound, max_bound):
        for i in range(3):
            if point[i] < min_bound[i] or point[i] > max_bound[i]:
                return False
        return True

    def handle_scene(self, scene):
        """读取mesh，组合后求取归一化参数，然后分别归一化到单位球内，保存结果"""
        self.geometries_path = getGeometriesPath(self.specs, scene)

        sdf_data = self.read_sdf_data()
        IOU = self.read_mesh(self.geometries_path["IOU"])
        IOU.scale(1.5, IOU.get_center())
        pcd = self.sample_points_from_sdf(sdf_data, IOU)

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
    config_filepath = 'config/getIBSPCDFromSDF.json'
    specs = parseConfig(config_filepath)
    # 构建文件树
    filename_tree = getFilenameTree(specs, "sdf_dir")
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
