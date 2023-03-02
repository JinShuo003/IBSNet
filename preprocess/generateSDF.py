import os
import re
import open3d as o3d
import numpy as np
from utils import *
import csv
import json


def parseConfig(config_filepath: str = './config/generateSDF.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def generateSDF(mesh_dir: str, mesh_filename_1: str, mesh_filename_2: str, sample_option: dict):
    """
    在mesh1和mesh2组成的空间下进行随机散点，计算每个点对于两个mesh的sdf值，生成(x, y, z, sdf1, sdf2)后存储到pcd_dir下
    """

    # 获取mesh
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_1))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_2))

    # 获取总体的aabb框，在其范围内散点
    aabb1, aabb2, aabb = getTwoMeshBorder(mesh1, mesh2)
    # random_points = getRandomPointsTogether(aabb, points_num, sample_method)
    random_points = getRandomPointsSeparately(aabb1, aabb2, sample_option)

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
    SDF_data = np.concatenate([random_points, signed_distance1, signed_distance2], axis=1)
    return SDF_data


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


def getRandomPointsSeparately(aabb1, aabb2, sample_option: dict):
    """
    在aabb范围内按照sample_method规定的采样方法采样points_num个点
    """
    # 解析采样选项
    method = sample_option["method"]
    points_num = sample_option["points_num"]
    scale1 = sample_option["scale1"]
    scale2 = sample_option["scale2"]
    proportion1 = sample_option["proportion1"]
    proportion2 = sample_option["proportion2"]

    # 获取mesh1和mesh2的包围框边界点
    min_bound_mesh1 = aabb1.get_min_bound() * scale1
    max_bound_mesh1 = aabb1.get_max_bound() * scale1
    min_bound_mesh2 = aabb2.get_min_bound() * scale2
    max_bound_mesh2 = aabb2.get_max_bound() * scale2

    random_points_mesh1 = []
    random_points_mesh2 = []
    if method == 'uniform':
        for i in range(3):
            random_points_mesh1.append(randUniFormFloat(min_bound_mesh1[i], max_bound_mesh1[i],
                                                        int(points_num * proportion1)).reshape((-1, 1)))
            random_points_mesh2.append(randUniFormFloat(min_bound_mesh2[i], max_bound_mesh2[i],
                                                        int(points_num * proportion2)).reshape((-1, 1)))
    elif method == 'normal':
        for i in range(3):
            random_points_mesh1.append(randNormalFloat(min_bound_mesh1[i], max_bound_mesh1[i],
                                                       int(points_num * proportion1)).reshape((-1, 1)))
            random_points_mesh2.append(randNormalFloat(min_bound_mesh2[i], max_bound_mesh2[i],
                                                       int(points_num * proportion2)).reshape((-1, 1)))

    random_points_mesh1_ = np.concatenate([random_points_mesh1[0], random_points_mesh1[1], random_points_mesh1[2]],
                                          axis=1)
    random_points_mesh2_ = np.concatenate([random_points_mesh2[0], random_points_mesh2[1], random_points_mesh2[2]],
                                          axis=1)

    return np.concatenate([random_points_mesh1_, random_points_mesh2_], axis=0)


def visualizeMeshAndSDF(mesh_dir: str, mesh_filename_1: str, mesh_filename_2: str, SDF_data=None, visualization_option={}):
    """
    显示sdf值小于threshold的点，以及geometries中的内容
    """
    geometries = []
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_1))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_filename_2))
    aabb1, aabb2, aabb = getTwoMeshBorder(mesh1, mesh2)
    if visualization_option['mesh1']:
        # mesh1.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        geometries.append(mesh1)
    if visualization_option['mesh2']:
        # mesh2.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        geometries.append(mesh2)
    if visualization_option['aabb1']:
        geometries.append(aabb1)
    if visualization_option['aabb2']:
        geometries.append(aabb2)
    if visualization_option['aabb']:
        geometries.append(aabb)
    if SDF_data is not None:
        if visualization_option['sdf1']:
            points1 = [points[0:3] for points in SDF_data if abs(points[3]) < visualization_option['sdf_threshold']]
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1)
            pcd1.paint_uniform_color([1, 0, 0])
            geometries.append(pcd1)
        if visualization_option['sdf2']:
            points2 = [points[0:3] for points in SDF_data if abs(points[4]) < visualization_option['sdf_threshold']]
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points2)
            pcd2.paint_uniform_color([0, 1, 0])
            geometries.append(pcd2)
        if visualization_option['ibs']:
            points3 = [points[0:3] for points in SDF_data if abs(points[3] - points[4]) < visualization_option['sdf_threshold']]
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(points3)
            pcd3.paint_uniform_color([0, 0, 1])
            geometries.append(pcd3)

    o3d.visualization.draw_geometries(geometries)


def saveSDFData(sdf_dir: str, category: str, sdf_filename: str, SDF_data):
    # 目录不存在则创建
    if not os.path.isdir(os.path.join(sdf_dir, category)):
        os.mkdir(os.path.join(sdf_dir, category))
    # 将data写入文件
    sdf_path = os.path.join(sdf_dir, category, sdf_filename)
    if os.path.isfile(sdf_path):
        print('exsit')
        os.remove(sdf_path)

    print(sdf_path)
    np.savez(sdf_path, data=SDF_data)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generateSDF.json'
    specs = parseConfig(configFile_path)

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["sdf_path"]):
        os.mkdir(specs["sdf_path"])

    # 遍历mesh_dir下的每一对mesh，每对mesh生成一个sdf groundtruth文件，写入sdf_path
    filename_list = os.listdir(specs["mesh_path"])
    handled_filename_list = set()
    for filename in filename_list:
        # 跳过不匹配正则式的文件
        if re.match(specs["process_filename_re"], filename) is None:
            continue

        # 数据成对出现，处理完一对后将前缀记录到map中，防止重复处理
        current_pair_name = re.match(specs["mesh_filename_re"], filename).group()
        if current_pair_name in handled_filename_list:
            continue
        else:
            handled_filename_list.add(current_pair_name)

        mesh_filename_1 = "{}_0.off".format(current_pair_name)
        mesh_filename_2 = "{}_1.off".format(current_pair_name)
        sdf_filename = "{}.npz".format(current_pair_name)
        SDF_data = generateSDF(specs["mesh_path"], mesh_filename_1, mesh_filename_2, specs["sample_option"])

        if specs["save_data"]:
            category = re.match(specs["category_re"], current_pair_name).group()
            saveSDFData(specs["sdf_path"], category, sdf_filename, SDF_data)
        if specs["visualize"]:
            visualizeMeshAndSDF(specs["mesh_path"], mesh_filename_1, mesh_filename_2, SDF_data, specs["visualization_option"])
