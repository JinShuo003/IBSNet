"""
可视化工具，配置好./config/visualization.json后可以可视化mesh模型、点云、位于模型表面和IBS表面的sdf点、各自和总体的aabb框、交互区域gt
"""
import open3d as o3d
import os
import re
import json
import numpy as np


def parseConfig(config_filepath: str = './config/visualization.json'):
    with open(config_filepath, 'r') as configfile:
        specs = json.load(configfile)

    return specs


def getGeometriesPath(specs, category, filename_intersection, augment_index):
    geometries_path = dict()

    mesh_dir = specs['mesh_dir']
    pcd_dir = specs['pcd_dir']
    sdf_dir = specs['sdf_dir']
    IOUgt_dir = specs['IOUgt_dir']

    mesh1_filename = filename_intersection + '_{}.off'.format(0)
    mesh2_filename = filename_intersection + '_{}.off'.format(1)
    pcd1_filename = filename_intersection + '_rotate{}_{}.ply'.format(augment_index, 0)
    pcd2_filename = filename_intersection + '_rotate{}_{}.ply'.format(augment_index, 1)
    sdf_filename = filename_intersection + '_rotate{}.npz'.format(augment_index)
    IOUgt_filename = filename_intersection + '.txt'

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)
    geometries_path['sdf'] = os.path.join(sdf_dir, category, sdf_filename)
    geometries_path['IOUgt'] = os.path.join(IOUgt_dir, category, IOUgt_filename)

    return geometries_path


def get_mesh(specs, mesh_filepath, color_key):
    mesh = o3d.io.read_triangle_mesh(mesh_filepath)
    mesh.paint_uniform_color(specs['visualization_options']['colors'][color_key])

    return mesh


def get_pcd(specs, pcd_filepath, color_key):
    pcd = o3d.io.read_point_cloud(pcd_filepath)
    pcd.paint_uniform_color(specs['visualization_options']['colors'][color_key])

    return pcd


def get_surface_points(specs, sdf_filepath):
    npz = np.load(sdf_filepath)

    data = npz["data"]
    surface_points1 = [points[0:3] for points in data if abs(points[3]) < specs['visualization_options']['sdf_threshold']]
    surface_points2 = [points[0:3] for points in data if abs(points[4]) < specs['visualization_options']['sdf_threshold']]
    surface_points_ibs = [points[0:3] for points in data if abs(points[3] - points[4]) < specs['visualization_options']['sdf_threshold']]

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd_ibs = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(surface_points1)
    pcd2.points = o3d.utility.Vector3dVector(surface_points2)
    pcd_ibs.points = o3d.utility.Vector3dVector(surface_points_ibs)

    pcd1.paint_uniform_color(specs['visualization_options']['colors']['sdf1'])
    pcd2.paint_uniform_color(specs['visualization_options']['colors']['sdf2'])
    pcd_ibs.paint_uniform_color(specs['visualization_options']['colors']['ibs'])

    return pcd1, pcd2, pcd_ibs


def get_IOUgt(specs, IOUgt_filepath):
    with open(IOUgt_filepath, 'r') as file:
        data = file.readlines()
        line1 = data[0].strip('\n').strip(' ').split(' ')
        line2 = data[1].strip('\n').strip(' ').split(' ')
        min_bound = np.array([float(item) for item in line1])
        max_bound = np.array([float(item) for item in line2])

        IOUgt = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        IOUgt.color = tuple(specs['visualization_options']['colors']['aabb_IOUgt'])

        return IOUgt


def getTwoMeshBorder(mesh1, mesh2):
    # 计算共同的最小和最大边界点，构造成open3d.geometry.AxisAlignedBoundingBox
    border_min = np.array([mesh1.get_min_bound(), mesh2.get_min_bound()]).min(0)
    border_max = np.array([mesh1.get_max_bound(), mesh2.get_max_bound()]).max(0)
    aabb_total = o3d.geometry.AxisAlignedBoundingBox(border_min, border_max)

    # 求mesh1和mesh2的边界
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()

    # 为边界框着色
    aabb1.color = tuple(specs['visualization_options']['colors']['aabb1'])
    aabb2.color = tuple(specs['visualization_options']['colors']['aabb2'])
    aabb_total.color = tuple(specs['visualization_options']['colors']['aabb_total'])

    return aabb1, aabb2, aabb_total


def visualize(specs, category, filename_intersection):
    rotate_num = specs["augmentation_options"]["rotate_num"]

    for i in range(rotate_num):
        container = dict()
        geometries = []
        geometries_path = getGeometriesPath(specs, category, filename_intersection, i)
        mesh1 = get_mesh(specs, geometries_path['mesh1'], 'mesh1')
        mesh2 = get_mesh(specs, geometries_path['mesh2'], 'mesh2')
        aabb1, aabb2, aabb = getTwoMeshBorder(mesh1, mesh2)
        IOUgt = get_IOUgt(specs, geometries_path['IOUgt'])
        pcd1 = get_pcd(specs, geometries_path['pcd1'], 'pcd1')
        pcd2 = get_pcd(specs, geometries_path['pcd2'], 'pcd2')
        sdf1, sdf2, ibs = get_surface_points(specs, geometries_path['sdf'])

        container['mesh1'] = mesh1
        container['mesh2'] = mesh2
        container['pcd1'] = pcd1
        container['pcd2'] = pcd2
        container['aabb1'] = aabb1
        container['aabb2'] = aabb2
        container['aabb'] = aabb
        container['aabb_IOUgt'] = IOUgt
        container['sdf1'] = sdf1
        container['sdf2'] = sdf2
        container['ibs'] = ibs

        for key in specs['visualization_options']['geometries']:
            if specs['visualization_options']['geometries'][key]:
                geometries.append(container[key])

        o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/visualization_augment.json'
    specs = parseConfig(config_filepath)

    handled_scenes = set()
    for category in specs['categories']:
        category_dir = os.path.join(specs["IOUgt_dir"], category)
        # 列出当前类别目录下所有文件名
        filename_list = os.listdir(category_dir)
        for filename in filename_list:
            # 得到公共部分的名字
            filename_intersection = re.match(specs['filename_re'], filename).group()
            if filename_intersection in handled_scenes:
                continue
            else:
                handled_scenes.add(filename_intersection)
            # 跳过不匹配正则式的文件
            if re.match(specs["process_filename_re"], filename_intersection) is None:
                continue

            print('current file: ', filename_intersection)
            visualize(specs, category, filename_intersection)
