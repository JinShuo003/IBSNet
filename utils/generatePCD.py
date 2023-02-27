import os
import re
import open3d as o3d


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


def generatePCD(mesh_dir: str, pcd_dir: str):
    """
    :param mesh_dir: mesh数据的目录
    :param pcd_dir: 生成点云的目录，不存在则自动创建
    依次读取mesh_dir下每一个mesh数据，生成对应的点云后存到pcd_dir下
    """
    if not os.path.isdir(pcd_dir):
        os.mkdir(pcd_dir)
    filename_list = os.listdir(mesh_dir)
    for filename in filename_list:
        transFormMeshToPCD(mesh_dir, filename, pcd_dir)


def transFormMeshToPCD(mesh_dir: str, mesh_filename, pcd_dir: str):
    """
    :param mesh_dir: mesh数据的目录
    :param mesh_filename: mesh数据文件名
    :param pcd_dir: 生成点云的目录
    将mesh数据进行表面泊松采样，保留1024个点，生成同名的点云文件存到pcd_dir下
    """
    print('current mesh filename: ', mesh_filename)
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename))
    pcd = mesh.sample_points_poisson_disk(number_of_points=1024, init_factor=10)
    o3d.visualization.draw_geometries([mesh, pcd])
    filename_re = 'scene\d\.\d{4}_\d'
    result = re.match(filename_re, mesh_filename)
    pcd_filename = result.group()+'.ply'
    pcd_path = os.path.join(pcd_dir, pcd_filename)
    if os.path.isfile(pcd_path):
        print('exsit')
        os.remove(pcd_path)
    o3d.io.write_point_cloud(pcd_path, pcd)


if __name__ == '__main__':
    mesh_relative_path = '../data/mesh'
    pcd_relative_path = '../data/pcd'
    generatePCD(os.path.abspath(mesh_relative_path), os.path.abspath(pcd_relative_path))
