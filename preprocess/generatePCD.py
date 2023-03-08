"""
从mesh生成点云的工具，配置好./config/generatePointCloud.json后可以按场景生成点云
"""
import os
import re
import open3d as o3d
import json


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


def transformMeshToPCD(mesh_dir: str, pcd_dir: str, category, mesh_filename, specs):
    """
    # 读取mesh_dir+category+mesh_filename的mesh文件，pcd_dir+category+mesh_filename.ply
    :param mesh_dir: mesh数据的目录
    :param mesh_filename: mesh数据文件名
    :param pcd_dir: 生成点云的目录
    """
    print('current mesh filename: ', mesh_filename)
    # 获取mesh和点云数据
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, mesh_filename))
    pcd = mesh.sample_points_poisson_disk(number_of_points=specs["sample_options"]["number_of_points"], init_factor=10)

    # 可视化
    if specs["visualization"]:
        o3d.visualization.draw_geometries([mesh, pcd])

    # 获取点云名
    filename_re = 'scene\d\.\d{4}_\d'
    pcd_filename = re.match(filename_re, mesh_filename).group()+'.ply'

    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_dir, category)):
        os.mkdir(os.path.join(pcd_dir, category))

    # 保存点云
    pcd_path = os.path.join(pcd_dir, category, pcd_filename)
    if os.path.isfile(pcd_path):
        os.remove(pcd_path)
    o3d.io.write_point_cloud(pcd_path, pcd)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generatePointCloud.json'
    specs = parseConfig(configFile_path)

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["pcd_path"]):
        os.mkdir(specs["pcd_path"])

    categories = specs["categories"]
    for category in categories:
        filename_list = os.listdir(os.path.join(specs["mesh_path"], category))
        for filename in filename_list:
            transformMeshToPCD(os.path.abspath(specs["mesh_path"]), os.path.abspath(specs["pcd_path"]), category, filename, specs)
