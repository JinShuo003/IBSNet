import os
import re
import open3d as o3d
import numpy as np
from utils import *
import csv
import json

mesh_filename_re = 'scene\d\.\d{4}'
visualization_mesh_filename_re = 'scene8'


def parseConfig(config_filepath: str = './config/generateSDF.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)

    mesh_path = config['mesh_path']
    sdf_path = config['sdf_path']
    save_data = config['save_data']
    visualize = config['visualize']
    points_num = config['points_num']
    scale1 = config['scale1']
    scale2 = config['scale2']
    proportion1 = config['proportion1']
    proportion2 = config['proportion2']
    sdf_threshold = config['sdf_threshold']
    method = config['method']
    return mesh_path, sdf_path, save_data, visualize, points_num, scale1, scale2, proportion1, proportion2, sdf_threshold, method


def generateAll(mesh_dir: str, sdf_dir: str, save_data: bool, visualize: bool, points_num: int, scale1: float,
                scale2: float, proportion1: float, proportion2: float, sdf_threshold: float, method: str = 'uniform'):
    """
    依次读取mesh_dir下每一个mesh数据，生成对应的sdf数据后存到sdf_dir下
    :param mesh_dir: mesh数据的目录
    :param sdf_dir: 生成sdf的目录，不存在则自动创建
    :param points_num: 采样点数
    :param sample_method: 采样方法，可选均匀采样和高斯采样
    """
    # 若目录不存在则创建目录
    if not os.path.isdir(sdf_dir):
        os.mkdir(sdf_dir)

    # 遍历mesh_dir下的每一对mesh，每对mesh生成一个sdf groundtruth文件
    filename_list = os.listdir(mesh_dir)
    handled_filename_list = set()
    for filename in filename_list:
        # 可视化生成过程使用，跳过不匹配正则式的部分
        if re.match(visualization_mesh_filename_re, filename) is None:
            continue

        # 数据成对出现，处理完一对后将前缀记录到map中，防止重复处理
        current_pair_name = re.match(mesh_filename_re, filename).group()
        if current_pair_name in handled_filename_list:
            continue
        else:
            handled_filename_list.add(current_pair_name)

        generateSDF(mesh_dir, "{}_0.off".format(current_pair_name), "{}_1.off".format(current_pair_name),
                    sdf_dir, "{}.csv".format(current_pair_name),
                    points_num, scale1, scale2, proportion1, proportion2, sample_method)


def generateSDF(mesh_dir: str, mesh_filename_1: str, mesh_filename_2: str,
                save_data: bool, visualize: bool,
                sdf_dir: str, sdf_filename: str, points_num: int, scale: float, sample_method: str = 'uniform'):
    """
    在mesh1和mesh2组成的空间下进行随机散点，计算每个点对于两个mesh的sdf值，生成(x, y, z, sdf1, sdf2)后存储到pcd_dir下
    :param mesh_dir: mesh数据的目录
    :param mesh_filename_1: 第一个mesh的文件名
    :param mesh_filename_2: 第二个mesh的文件名
    :param sdf_dir: 生成sdf groundTruth文件的存储路径
    :param sdf_filename: sdf groundTruth文件的文件名
    :param points_num: 采样点个数
    :param sample_method: 采样方法
    """
    # 获取mesh
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_1))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_2))

    # 获取总体的aabb框，在其范围内散点
    aabb1, aabb2, aabb = getTwoMeshBorder(mesh1, mesh2)
    # random_points = getRandomPointsTogether(aabb, points_num, sample_method)
    random_points = getRandomPointsSeparately(aabb1, aabb2, points_num, scale, sample_method)

    # 创建光线追踪场景，并将模型添加到场景中
    scene1 = o3d.t.geometry.RaycastingScene()
    scene2 = o3d.t.geometry.RaycastingScene()
    mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    scene1.add_triangles(mesh1)
    scene2.add_triangles(mesh2)

    # 将生成的随机点转为o3d.Tensor
    query_points = o3d.core.Tensor(random_points, dtype=o3d.core.Dtype.Float32)

    # 批量计算查询点的sdf值
    # unsigned_distance1 = scene1.compute_distance(query_points)
    # unsigned_distance2 = scene2.compute_distance(query_points)
    signed_distance1 = scene1.compute_signed_distance(query_points).numpy().reshape((-1, 1))
    signed_distance2 = scene2.compute_signed_distance(query_points).numpy().reshape((-1, 1))

    # 拼接查询点和两个SDF值
    data = np.concatenate([random_points, signed_distance1, signed_distance2], axis=1)

    # # 将data写入文件
    # sdf_path = os.path.join(sdf_dir, sdf_filename)
    # if os.path.isfile(sdf_path):
    #     print('exsit')
    #     os.remove(sdf_path)
    #
    # with open(sdf_path, 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(data)

    # 可视化
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_1))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_2))
    geometries = [mesh1, mesh2, aabb1, aabb2, aabb]
    visualizeMeshAndSDF(data, threshold=0.1, geometries=geometries)


def getTwoMeshBorder(mesh1, mesh2):
    """
    计算一组mesh的最小边界框
    :param mesh1: 第一个mesh
    :param mesh2: 第二个mesh
    :return: aabb1, aabb2, aabb
    """
    # 计算共同的最小和最大边界点，构造成open3d.geometry.AxisAlignedBoundingBox
    border_min = np.array([mesh1.get_min_bound(), mesh2.get_min_bound()]).min(0)
    border_max = np.array([mesh1.get_max_bound(), mesh2.get_max_bound()]).max(0)
    aabb = o3d.geometry.AxisAlignedBoundingBox(border_min, border_max)
    # 求mesh1和mesh2的边界
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()
    # 为边界框着色
    aabb1.color = (1, 0, 0)
    aabb2.color = (0, 1, 0)
    aabb.color = (0, 0, 1)
    return aabb1, aabb2, aabb


def getRandomPointsTogether(aabb, points_num: int, sample_method: str = 'uniform'):
    """
    在aabb范围内按照sample_method规定的采样方法采样points_num个点
    :param aabb: 公共aabb包围框
    :param points_num: 采样点的个数
    :param sample_method: 采样方法，uniform为一致采样，normal为正态采样
    """
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    if sample_method == 'uniform':
        random_points_d1 = randUniFormFloat(min_bound[0], max_bound[0], points_num).reshape((-1, 1))
        random_points_d2 = randUniFormFloat(min_bound[1], max_bound[1], points_num).reshape((-1, 1))
        random_points_d3 = randUniFormFloat(min_bound[2], max_bound[2], points_num).reshape((-1, 1))
    elif sample_method == 'normal':
        random_points_d1 = randNormalFloat(min_bound[0], max_bound[0], points_num).reshape((-1, 1))
        random_points_d2 = randNormalFloat(min_bound[1], max_bound[1], points_num).reshape((-1, 1))
        random_points_d3 = randNormalFloat(min_bound[2], max_bound[2], points_num).reshape((-1, 1))

    random_points = np.concatenate([random_points_d1, random_points_d2, random_points_d3], axis=1)

    return random_points


def getRandomPointsSeparately(aabb1, aabb2, points_num: int, k: float = 1, sample_method: str = 'uniform'):
    """
    在aabb范围内按照sample_method规定的采样方法采样points_num个点
    :param aabb1: mesh1的包围框
    :param aabb2: mesh2的包围框
    :param points_num: 采样点的个数
    :param k: 包围框到采样空间的放大倍数
    :param sample_method: 采样方法，uniform为一致采样，normal为正态采样
    """
    # 获取mesh1和mesh2的包围框边界点
    min_bound_mesh1 = aabb1.get_min_bound()
    max_bound_mesh1 = aabb1.get_max_bound()
    min_bound_mesh2 = aabb2.get_min_bound() * k
    max_bound_mesh2 = aabb2.get_max_bound() * k

    if sample_method == 'uniform':
        random_points_mesh1_d1 = randUniFormFloat(min_bound_mesh1[0], max_bound_mesh1[0], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh1_d2 = randUniFormFloat(min_bound_mesh1[1], max_bound_mesh1[1], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh1_d3 = randUniFormFloat(min_bound_mesh1[2], max_bound_mesh1[2], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh2_d1 = randUniFormFloat(min_bound_mesh2[0], max_bound_mesh2[0], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh2_d2 = randUniFormFloat(min_bound_mesh2[1], max_bound_mesh2[1], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh2_d3 = randUniFormFloat(min_bound_mesh2[2], max_bound_mesh2[2], points_num // 2).reshape(
            (-1, 1))
    elif sample_method == 'normal':
        random_points_mesh1_d1 = randNormalFloat(min_bound_mesh1[0], max_bound_mesh1[0], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh1_d2 = randNormalFloat(min_bound_mesh1[1], max_bound_mesh1[1], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh1_d3 = randNormalFloat(min_bound_mesh1[2], max_bound_mesh1[2], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh2_d1 = randNormalFloat(min_bound_mesh2[0], max_bound_mesh2[0], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh2_d2 = randNormalFloat(min_bound_mesh2[1], max_bound_mesh2[1], points_num // 2).reshape(
            (-1, 1))
        random_points_mesh2_d3 = randNormalFloat(min_bound_mesh2[2], max_bound_mesh2[2], points_num // 2).reshape(
            (-1, 1))

    random_points_mesh1 = np.concatenate([random_points_mesh1_d1, random_points_mesh1_d2, random_points_mesh1_d3],
                                         axis=1)
    random_points_mesh2 = np.concatenate([random_points_mesh2_d1, random_points_mesh2_d2, random_points_mesh2_d3],
                                         axis=1)

    return np.concatenate([random_points_mesh1, random_points_mesh2], axis=0)


def visualizeMeshAndSDF(data=None, sdf1=True, sdf2=True, ibs=True, threshold=1, geometries: list = []):
    """
    显示sdf值小于threshold的点，以及geometries中的内容
    :param data: sdf数据，格式为(x, y, z, sdf1, sdf2)
    :param sdf1: 是否显示sdf1
    :param sdf2: 是否显示sdf2
    :param ibs: 是否显示ibs
    :param threshold: 显示sdf点的阈值，小于阈值则显示
    :param geometries: 除sdf和ibs外，其他希望显示的几何体
    """
    if data is not None:
        if sdf1:
            points1 = [points[0:3] for points in data if abs(points[3]) < threshold]
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1)
            pcd1.paint_uniform_color([1, 0, 0])
            geometries.append(pcd1)

        if sdf2:
            points2 = [points[0:3] for points in data if abs(points[4]) < threshold]
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points2)
            pcd2.paint_uniform_color([0, 1, 0])
            geometries.append(pcd2)

        if ibs:
            points3 = [points[0:3] for points in data if abs(points[3] - points[4]) < threshold]
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(points3)
            pcd3.paint_uniform_color([0, 0, 1])
            geometries.append(pcd3)

    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    config_filepath = './config/generateSDF.json'

    mesh_path, sdf_path, points_num, scale1, scale2, proportion1, proportion2, sample_method = parseConfig(
        config_filepath)

    generateAll(os.path.abspath(mesh_path), os.path.abspath(sdf_path), points_num, scale1, scale2, proportion1,
                proportion2, sample_method)
