"""
从mesh生成点云的工具，配置好./config/generatePointCloud.json后可以按场景生成点云
"""
import os
import re
import open3d as o3d
import json
import numpy as np


def parseConfig(config_filepath: str = './config/generatePointCloud.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def removeFiles(dir: str, reserve_re: str):
    """
    :param dir: mesh数据的路径
    :param reserve_re: mesh数据的命名规则
    删除所有不符合命名规则的数据
    """
    filename_list = os.listdir(dir)
    for filename in filename_list:
        if re.match(reserve_re, filename) is None:
            os.remove(os.path.join(dir, filename))


def transformMeshToPCD(mesh_dir: str, pcd_dir: str, category, cur_filename, specs):
    """
    # 读取mesh_dir+category+mesh_filename的mesh文件，pcd_dir+category+mesh_filename.ply
    :param mesh_dir: mesh数据的目录
    :param cur_filename: 当前处理的文件名
    :param pcd_dir: 生成点云的目录
    """
    print('current mesh filename: ', cur_filename)
    # 获取mesh和点云数据
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, '{}_0.off'.format(cur_filename)))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, '{}_1.off'.format(cur_filename)))
    pcd1 = mesh1.sample_points_poisson_disk(number_of_points=specs["sample_options"]["number_of_points"], init_factor=10)
    pcd2 = mesh2.sample_points_poisson_disk(number_of_points=specs["sample_options"]["number_of_points"], init_factor=10)

    pcd1, pcd2, centroid, scale = normalize_point_cloud(pcd1, pcd2)
    # 可视化
    if specs["visualization"]:
        o3d.visualization.draw_geometries([mesh1, mesh2, pcd1, pcd2])

    # 获取点云名
    pcd1_filename = '{}_0.ply'.format(cur_filename)
    pcd2_filename = '{}_1.ply'.format(cur_filename)

    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_dir, category)):
        os.mkdir(os.path.join(pcd_dir, category))

    # 获取缩放比例文件
    scale_filename = '{}_scale.txt'.format(category)
    scale_path = os.path.join(pcd_dir, category, scale_filename)

    # 保存点云
    pcd1_path = os.path.join(pcd_dir, category, pcd1_filename)
    if os.path.isfile(pcd1_path):
        os.remove(pcd1_path)
    pcd2_path = os.path.join(pcd_dir, category, pcd2_filename)
    if os.path.isfile(pcd2_path):
        os.remove(pcd2_path)
    o3d.io.write_point_cloud(pcd1_path, pcd1)
    o3d.io.write_point_cloud(pcd2_path, pcd2)

    # 保存缩放比例
    with open(scale_path, 'a') as scale_file:
        scale_file.write('{},{},{}\n'.format(cur_filename, centroid, scale))


def normalize_point_cloud(pcd1, pcd2):
    # 获取物体12的点云
    pcd1_np = np.asarray(pcd1.points)
    pcd2_np = np.asarray(pcd2.points)
    # 将两点云进行拼接
    pcd_total_np = np.concatenate((pcd1_np, pcd2_np), axis=0)
    # 求取整体点云的中心
    centroid = np.mean(pcd_total_np, axis=0)
    # 将总体点云中心置于原点 (0, 0, 0)
    pcd1_np = pcd1_np - centroid
    pcd2_np = pcd2_np - centroid
    # 求取长轴的的长度
    m = np.max(np.sqrt(np.sum(pcd_total_np ** 2, axis=1)))
    # 依据长轴将点云归一化到 (-1, 1)
    pcd1_normalized_np = pcd1_np / m
    pcd2_normalized_np = pcd2_np / m

    pcd1_normalized = o3d.geometry.PointCloud()
    pcd2_normalized = o3d.geometry.PointCloud()
    pcd1_normalized.points = o3d.utility.Vector3dVector(pcd1_normalized_np)
    pcd2_normalized.points = o3d.utility.Vector3dVector(pcd2_normalized_np)
    return pcd1_normalized, pcd2_normalized, centroid, m


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generatePointCloud.json'
    specs = parseConfig(configFile_path)
    filename_re = specs['mesh_filename_re']

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["pcd_path"]):
        os.mkdir(specs["pcd_path"])

    categories = specs["categories"]
    handled_data = set()  # 成对处理，记录当前处理过的文件名
    for category in categories:
        filename_list = os.listdir(os.path.join(specs["mesh_path"], category))
        for filename in filename_list:
            cur_filename = re.match(filename_re, filename).group()
            if cur_filename in handled_data:
                continue
            else:
                handled_data.add(cur_filename)
            transformMeshToPCD(os.path.abspath(specs["mesh_path"]), os.path.abspath(specs["pcd_path"]), category, cur_filename, specs)
