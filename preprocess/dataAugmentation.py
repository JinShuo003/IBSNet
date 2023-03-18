"""
将点云数据和SDF sample数据进行数据增强，在空间内绕z轴进行旋转
"""
import copy
import os
import re
import open3d as o3d
import json
import numpy as np
import datetime
from utils import *


def parseConfig(config_filepath: str = './config/generatePointCloud.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


# ----------------------------------------点云-------------------------------------------
def get_pcd(specs, category, cur_filename):
    pcd_dir = specs["pcd_src_path"]
    pcd1_filename = os.path.join(pcd_dir, category, "{}_0.ply".format(cur_filename))
    pcd2_filename = os.path.join(pcd_dir, category, "{}_1.ply".format(cur_filename))

    pcd1 = o3d.io.read_point_cloud(pcd1_filename)
    pcd2 = o3d.io.read_point_cloud(pcd2_filename)

    return pcd1, pcd2


def pcd_augment(specs, pcd1, pcd2):
    rotate_num = specs["augmentation_options"]["rotate_num"]
    pcd1_list = []
    pcd2_list = []
    for i in range(rotate_num):
        R = pcd1.get_rotation_matrix_from_xyz((0, (2 * np.pi / rotate_num) * i, 0))
        pcd1_temp = copy.deepcopy(pcd1)
        pcd2_temp = copy.deepcopy(pcd2)
        pcd1_temp.rotate(R, center=(0., 0., 0.))
        pcd2_temp.rotate(R, center=(0., 0., 0.))
        pcd1_list.append(pcd1_temp)
        pcd2_list.append(pcd2_temp)
    return pcd1_list, pcd2_list


def save_pcd(pcd1_list, pcd2_list, specs, category, cur_filename):
    pcd_save_dir = specs['pcd_save_path']
    rotate_num = specs["augmentation_options"]["rotate_num"]

    # pcd_save_path+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_save_dir, category)):
        os.makedirs(os.path.join(pcd_save_dir, category))

    pcd_save_path = os.path.join(pcd_save_dir, category)
    for i in range(rotate_num):
        # 获取点云名
        pcd1_filename = '{}_rotate{}_0.ply'.format(cur_filename, i)
        pcd2_filename = '{}_rotate{}_1.ply'.format(cur_filename, i)

        # 保存点云
        pcd1_path = os.path.join(pcd_save_path, pcd1_filename)
        if os.path.isfile(pcd1_path):
            os.remove(pcd1_path)
        pcd2_path = os.path.join(pcd_save_path, pcd2_filename)
        if os.path.isfile(pcd2_path):
            os.remove(pcd2_path)
        o3d.io.write_point_cloud(pcd1_path, pcd1_list[i])
        o3d.io.write_point_cloud(pcd2_path, pcd2_list[i])


# ----------------------------------------SDF-------------------------------------------
def get_sdf(specs, category, cur_filename):
    sdf_src_path = specs["sdf_src_path"]
    sdf_filepath = os.path.join(sdf_src_path, category, "{}.npz".format(cur_filename))
    npz = np.load(sdf_filepath)
    data = npz["data"]
    return data
    # surface_points1 = [points[0:3] for points in data if
    #                    abs(points[3]) < specs['visualization_options']['sdf_threshold']]
    # surface_points2 = [points[0:3] for points in data if
    #                    abs(points[4]) < specs['visualization_options']['sdf_threshold']]
    # surface_points_ibs = [points[0:3] for points in data if
    #                       abs(points[3] - points[4]) < specs['visualization_options']['sdf_threshold']]
    #
    # pcd1 = o3d.geometry.PointCloud()
    # pcd2 = o3d.geometry.PointCloud()
    # pcd_ibs = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(surface_points1)
    # pcd2.points = o3d.utility.Vector3dVector(surface_points2)
    # pcd_ibs.points = o3d.utility.Vector3dVector(surface_points_ibs)
    #
    # pcd1.paint_uniform_color(specs['visualization_options']['colors']['sdf1'])
    # pcd2.paint_uniform_color(specs['visualization_options']['colors']['sdf2'])
    # pcd_ibs.paint_uniform_color(specs['visualization_options']['colors']['ibs'])
    #
    # return pcd1, pcd2, pcd_ibs


def sdf_augment(specs, sdf_samples):
    rotate_num = specs["augmentation_options"]["rotate_num"]
    sdf_list = []
    # 先将sdf值切分出来
    xyz_np = sdf_samples[:, 0:3]
    # 将sdf采样点转换为o3d点云
    xyz_pcd = o3d.geometry.PointCloud()
    xyz_pcd.points = o3d.utility.Vector3dVector(xyz_np)
    for i in range(rotate_num):
        R = xyz_pcd.get_rotation_matrix_from_xyz((0, (2 * np.pi / rotate_num) * i, 0))
        xyz_temp = copy.deepcopy(xyz_pcd)
        xyz_temp.rotate(R, center=(0., 0., 0.))
        xyz_temp = np.asarray(xyz_temp.points)
        sdf_temp = np.concatenate([xyz_temp, sdf_samples[:, 3:5]], axis=1)
        sdf_list.append(sdf_temp)
    return sdf_list


def visualiza_sdf(sdf_list):
    for i in range(len(sdf_list)):
        surface_points1 = [points[0:3] for points in sdf_list[i] if abs(points[3]) < 0.01]
        surface_points2 = [points[0:3] for points in sdf_list[i] if abs(points[4]) < 0.01]
        surface_points_ibs = [points[0:3] for points in sdf_list[i] if abs(points[3] - points[4]) < 0.01]

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd_ibs = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(surface_points1)
        pcd2.points = o3d.utility.Vector3dVector(surface_points2)
        pcd_ibs.points = o3d.utility.Vector3dVector(surface_points_ibs)

        pcd1.paint_uniform_color((1, 0, 0))
        pcd2.paint_uniform_color((0, 1, 0))
        pcd_ibs.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd_ibs])
    # return pcd1, pcd2, pcd_ibs


def save_sdf(specs, category, cur_filename, sdf_list):
    sdf_save_dir = specs['sdf_save_path']
    rotate_num = specs["augmentation_options"]["rotate_num"]

    # sdf_save_path+category不存在则创建目录
    if not os.path.isdir(os.path.join(sdf_save_dir, category)):
        os.makedirs(os.path.join(sdf_save_dir, category))

    sdf_save_path = os.path.join(sdf_save_dir, category)
    for i in range(rotate_num):
        # 获取点云名
        sdf_filename = '{}_rotate{}.npz'.format(cur_filename, i)
        # 保存点云
        sdf_path = os.path.join(sdf_save_path, sdf_filename)
        if os.path.isfile(sdf_path):
            os.remove(sdf_path)
        np.savez(sdf_path, data=sdf_list[i])


# ----------------------------------------其他-------------------------------------------

def handle_scene(specs, category, cur_filename):
    # 读取点云数据
    pcd1, pcd2 = get_pcd(specs, category, cur_filename)
    # 将点云数据进行增强
    pcd1_list, pcd2_list = pcd_augment(specs, pcd1, pcd2)
    # 保存增强后的点云数据
    save_pcd(pcd1_list, pcd2_list, specs, category, cur_filename)

    # 读取SDF sample数据
    sdf_samples = get_sdf(specs, category, cur_filename)
    # 将点云数据进行增强
    sdf_list = sdf_augment(specs, sdf_samples)
    # 保存增强后的点云数据
    save_sdf(specs, category, cur_filename, sdf_list)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/dataAugmentation.json'
    specs = parseConfig(configFile_path)
    filename_re = specs['mesh_filename_re']

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

            print('current scene: ', cur_filename)

            handle_scene(specs, category, cur_filename)
