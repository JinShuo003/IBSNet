import open3d as o3d
import os
import re

filename_re = 'scene\d\.\d{4}_\d'
filepath_re = '.*scene\d\.\d{4}_\d.*'


def visualizeAll(mesh_dir: str, pcd_dir: str):
    """
    :param mesh_dir: mesh数据的目录
    :param pcd_dir: 生成点云的目录，不存在则自动创建
    依次读取mesh_dir下每一个mesh数据，将其对应的点云可视化在同一个窗口中
    """
    if not os.path.isdir(pcd_dir):
        os.mkdir(pcd_dir)
    filename_list = os.listdir(mesh_dir)
    for filename in filename_list:
        mesh_filename = os.path.join(mesh_dir, filename)
        pcd_filename = os.path.join(pcd_dir, re.match(filename_re, filename).group() + '.ply')
        visualizePair(mesh_filename, pcd_filename)


def visualizePair(mesh_filename: str, pcd_filename: str):
    """
    :param mesh_filename: mesh数据目录
    :param pcd_filename: pcd数据目录
    读取mesh和点云数据，可视化在同一个窗口下，若不存在则报错
    """
    if not os.path.isfile(mesh_filename):
        print('mesh not exist')
        return
    if not os.path.isfile(pcd_filename):
        print('pointcloud not exist')
        return
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    pcd = o3d.io.read_point_cloud(pcd_filename)
    print('{}\n{}\n'.format(mesh_filename, pcd_filename))
    o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == '__main__':
    mesh_relative_path = '../data/mesh'
    pcd_relative_path = '../data/pcd'
    visualizeAll(os.path.abspath(mesh_relative_path), os.path.abspath(pcd_relative_path))
