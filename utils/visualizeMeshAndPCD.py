import open3d as o3d
import os
import re
import json

filename_re = 'scene\d\.\d{4}_\d'
filepath_re = '.*scene\d\.\d{4}_\d.*'


def parseConfig(config_filepath: str = './config/visualizeMeshAndPCD.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)

    mesh_path = config['mesh_path']
    pcd_path = config['pcd_path']
    mesh_path = os.path.abspath(mesh_path)
    pcd_path = os.path.abspath(pcd_path)
    mesh_filename_re = config['mesh_filename_re']
    process_filename_re = config['process_filename_re']

    return mesh_path, pcd_path, mesh_filename_re, process_filename_re


def visualizePair(mesh_filename: str, pcd_filename: str):
    """
    :param mesh_filename: mesh数据目录
    :param pcd_filename: pcd数据目录
    读取mesh和点云数据，可视化在同一个窗口下，若不存在则报错
    """
    if not os.path.isfile(mesh_filename):
        return
    if not os.path.isfile(pcd_filename):
        return
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    pcd = o3d.io.read_point_cloud(pcd_filename)
    o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = './config/visualizeMeshAndPCD.json'
    mesh_path, pcd_path, mesh_filename_re, process_filename_re = parseConfig(config_filepath)
    if not os.path.isdir(pcd_path):
        os.mkdir(pcd_path)

    filename_list = os.listdir(mesh_path)
    for filename in filename_list:
        # 跳过不匹配正则式的文件
        if re.match(process_filename_re, filename) is None:
            continue

        mesh_filename = os.path.join(mesh_path, filename)
        pcd_filename = os.path.join(pcd_path, re.match(filename_re, filename).group() + '.ply')
        visualizePair(mesh_filename, pcd_filename)
