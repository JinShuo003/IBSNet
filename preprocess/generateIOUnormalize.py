"""
生成原始训练数据，该数据只截取交互区域IOU gt区域的点云，并以此区域将点云进行归一化
"""
import math
import os
import re
import open3d as o3d
import json
import numpy as np
import random
from utils import *


def parseConfig(config_filepath: str = './config/generateIOUnormalize.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


# ----------------------------------------mesh------------------------------------------
def get_mesh(specs, category, cur_filename):
    # 获取mesh
    mesh_dir = specs['mesh_path']
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, '{}_0.off'.format(cur_filename)))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, '{}_1.off'.format(cur_filename)))
    return mesh1, mesh2


def getAABBfromTwoPoints(file_path: str):
    with open(file_path, 'r') as file:
        data = file.readlines()
        line1 = data[0].strip('\n').strip(' ').split(' ')
        line2 = data[1].strip('\n').strip(' ').split(' ')
        min_bound = np.array([float(item) for item in line1])
        max_bound = np.array([float(item) for item in line2])

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        aabb.color = (1, 1, 0)

        return aabb


def get_normalize_para(specs, category, cur_filename):
    IOUgt_dir = specs["IOUgt_path"]
    scale_IOU = specs["SDF_sample_options"]["scale_IOU"]

    IOUgt_filename = "{}.txt".format(cur_filename)
    aabb_IOUgt = getAABBfromTwoPoints(os.path.join(IOUgt_dir, category, IOUgt_filename))
    min_bound_IOUgt = aabb_IOUgt.get_min_bound()
    max_bound_IOUgt = aabb_IOUgt.get_max_bound()

    centroid = (min_bound_IOUgt + max_bound_IOUgt) / 2
    scale = np.sqrt(np.sum(min_bound_IOUgt - max_bound_IOUgt) ** 2) / 6 * scale_IOU

    aabb_IOUgt.translate(-centroid)
    aabb_IOUgt.scale(1 / scale, np.array([0, 0, 0]))
    return aabb_IOUgt, centroid, scale


def clip_mesh(mesh1, mesh2, centroid, scale):
    mesh1.translate(-centroid)
    mesh2.translate(-centroid)
    mesh1.scale(1 / scale, np.array([0, 0, 0]))
    mesh2.scale(1 / scale, np.array([0, 0, 0]))

    outlier_vertices_mesh1 = []
    outlier_vertices_mesh2 = []

    vertices_mesh1 = np.asarray(mesh1.vertices)
    for i in range(vertices_mesh1.shape[0]):
        if np.sum(vertices_mesh1[i] ** 2) > 1:
            outlier_vertices_mesh1.append(i)
    vertices_mesh2 = np.asarray(mesh2.vertices)
    for i in range(vertices_mesh2.shape[0]):
        if np.sum(vertices_mesh2[i] ** 2) > 1:
            outlier_vertices_mesh2.append(i)
    mesh1.remove_vertices_by_index(outlier_vertices_mesh1)
    mesh2.remove_vertices_by_index(outlier_vertices_mesh2)


def vertices_enough(specs, mesh1, mesh2):
    number_of_points = specs["PCD_sample_options"]["number_of_points"]

    # 如果mesh的顶点数不够，则进行曲面细分
    mesh1_vertices = np.asarray(mesh1.vertices)
    mesh2_vertices = np.asarray(mesh2.vertices)

    if mesh1_vertices.shape[0] <= 0.5 * number_of_points:
        return False
    if mesh2_vertices.shape[0] <= 0.5 * number_of_points:
        return False

    while mesh1_vertices.shape[0] <= number_of_points * 2:
        print("devide mesh1")
        mesh1.compute_vertex_normals()
        mesh1 = mesh1.subdivide_midpoint(number_of_iterations=1)
        mesh1_vertices = np.asarray(mesh1.vertices)
    while mesh2_vertices.shape[0] <= number_of_points * 2:
        print("devide mesh2")
        mesh2.compute_vertex_normals()
        mesh2 = mesh2.subdivide_midpoint(number_of_iterations=1)
        mesh2_vertices = np.asarray(mesh2.vertices)
    return True


# ----------------------------------------点云-------------------------------------------
def get_pcd(specs, mesh1, mesh2):
    number_of_points = specs["PCD_sample_options"]["number_of_points"]

    # 采样得到点云
    pcd1 = mesh1.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=10)
    pcd2 = mesh2.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=10)

    return pcd1, pcd2


def save_pcd(pcd1, pcd2, specs, category, cur_filename):
    pcd_dir = specs['pcd_path']

    # 获取点云名
    pcd1_filename = '{}_0.ply'.format(cur_filename)
    pcd2_filename = '{}_1.ply'.format(cur_filename)

    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_dir, category)):
        os.makedirs(os.path.join(pcd_dir, category))

    # 保存点云
    pcd1_path = os.path.join(pcd_dir, category, pcd1_filename)
    if os.path.isfile(pcd1_path):
        os.remove(pcd1_path)
    pcd2_path = os.path.join(pcd_dir, category, pcd2_filename)
    if os.path.isfile(pcd2_path):
        os.remove(pcd2_path)
    o3d.io.write_point_cloud(pcd1_path, pcd1)
    o3d.io.write_point_cloud(pcd2_path, pcd2)


# ----------------------------------------SDF-------------------------------------------
def get_sdf_samples(specs):
    sdf_sample_num = specs["SDF_sample_options"]["points_num"]

    points = randPointsUniform(sdf_sample_num, 1)

    return np.array(points, dtype='float32')


def get_sdf_value(mesh1, mesh2, sdf_samples):
    scene1 = o3d.t.geometry.RaycastingScene()
    scene2 = o3d.t.geometry.RaycastingScene()
    mesh1_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    mesh2_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    scene1.add_triangles(mesh1_t)
    scene2.add_triangles(mesh2_t)

    dists_1 = scene1.compute_distance(sdf_samples).numpy().reshape(-1, 1)
    dists_2 = scene2.compute_distance(sdf_samples).numpy().reshape(-1, 1)

    SDF_data = np.concatenate([sdf_samples, dists_1, dists_2], axis=1)
    return SDF_data


def save_sdf(SDF_data, specs, category, cur_filename):
    sdf_dir = specs['sdf_path']
    # 目录不存在则创建
    if not os.path.isdir(os.path.join(sdf_dir, category)):
        os.makedirs(os.path.join(sdf_dir, category))

    # 将data写入文件
    sdf_filename = '{}.npz'.format(cur_filename)
    sdf_path = os.path.join(sdf_dir, category, sdf_filename)
    if os.path.isfile(sdf_path):
        print('sdf file exsit')
        os.remove(sdf_path)

    np.savez(sdf_path, data=SDF_data)


def visualization(pcd1, pcd2, SDF_data):
    # 可视化结果
    surface_points1 = [points[0:3] for points in SDF_data if
                       abs(points[3]) < 0.01]
    surface_points2 = [points[0:3] for points in SDF_data if
                       abs(points[4]) < 0.01]
    surface_points_ibs = [points[0:3] for points in SDF_data if
                          abs(points[3] - points[4]) < 0.01]

    surface1_pcd = o3d.geometry.PointCloud()
    surface2_pcd = o3d.geometry.PointCloud()
    surface_ibs_pcd = o3d.geometry.PointCloud()
    surface1_pcd.points = o3d.utility.Vector3dVector(surface_points1)
    surface2_pcd.points = o3d.utility.Vector3dVector(surface_points2)
    surface_ibs_pcd.points = o3d.utility.Vector3dVector(surface_points_ibs)

    pcd1.paint_uniform_color((1, 0, 0))
    pcd2.paint_uniform_color((1, 0, 0))
    surface1_pcd.paint_uniform_color((0, 1, 0))
    surface2_pcd.paint_uniform_color((0, 1, 0))
    surface_ibs_pcd.paint_uniform_color((0, 0, 1))

    o3d.visualization.draw_geometries([pcd1, pcd2, surface1_pcd, surface2_pcd, surface_ibs_pcd])


# ----------------------------------------其他-------------------------------------------
def handle_scene(specs, category, cur_filename):
    # 获取mesh
    mesh1, mesh2 = get_mesh(specs, category, cur_filename)
    # 获取交互区域的归一化参数
    aabb_IOUgt, centroid, scale = get_normalize_para(specs, category, cur_filename)
    # 将mesh归一化，去除单位球外的面片
    clip_mesh(mesh1, mesh2, centroid, scale)
    # 检查mesh中的顶点是否足够，如果过小则直接丢弃，如果小一些则进行曲面细分
    if not vertices_enough(specs, mesh1, mesh2):
        print("not enough vertices, discard")
        return
    # o3d.visualization.draw_geometries([mesh1, mesh2])
    # 对mesh进行采样，得到点云
    pcd1, pcd2 = get_pcd(specs, mesh1, mesh2)
    # 在归一化球内进行采样，得到sdf sample点
    sdf_samples = get_sdf_samples(specs)
    # 用cast ray的方式计算出sample点的sdf值
    SDF_data = get_sdf_value(mesh1, mesh2, sdf_samples)

    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()

    pcd1.paint_uniform_color((1, 0, 0))
    pcd2.paint_uniform_color((0, 1, 0))

    # 可视化
    if specs['visualization']:
        visualization(pcd1, pcd2, SDF_data)
    # 保存点云
    save_pcd(pcd1, pcd2, specs, category, cur_filename)
    # 保存SDF_data
    save_sdf(SDF_data, specs, category, cur_filename)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generateIOUnormalize.json'
    specs = parseConfig(configFile_path)
    filename_re = specs['mesh_filename_re']

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["pcd_path"]):
        os.makedirs(specs["pcd_path"])
    if not os.path.isdir(specs["sdf_path"]):
        os.makedirs(specs["sdf_path"])

    categories = specs["categories"]
    handled_data = set()  # 成对处理，记录当前处理过的文件名
    for category in categories:
        category_dir = os.path.join(specs["mesh_path"], category)
        filename_list = os.listdir(os.path.join(specs["mesh_path"], category))
        for filename in filename_list:
            #  跳过非文件
            file_absPath = os.path.join(category_dir, filename)
            if not os.path.isfile(file_absPath):
                continue
            # 跳过不匹配正则式的文件
            if re.match(specs["process_filename_re"], filename) is None:
                continue
            # 数据成对出现，处理完一对后将前缀记录到map中，防止重复处理
            cur_filename = re.match(filename_re, filename).group()
            if cur_filename in handled_data:
                continue
            else:
                handled_data.add(cur_filename)

            print('\ncurrent scene: ', cur_filename)

            handle_scene(specs, category, cur_filename)
