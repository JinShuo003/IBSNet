"""
从mesh生成单角度的残缺点云
"""
import math
import os
import re
import open3d as o3d
from open3d.visualization import rendering
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt
from utils import *


def parseConfig(config_filepath: str = './config/generatePointCloud.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


# ----------------------------------------点云-------------------------------------------
def get_pcd(specs, category, cur_filename):
    # 获取mesh
    mesh_dir = specs['mesh_path']
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, '{}_0.off'.format(cur_filename)))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, '{}_1.off'.format(cur_filename)))
    # 采样得到完整点云
    pcd_sample_options = specs["PCD_sample_options"]
    # pcd1 = mesh1.sample_points_poisson_disk(number_of_points=pcd_sample_options["number_of_points"],
    #                                         init_factor=10)
    # pcd2 = mesh2.sample_points_poisson_disk(number_of_points=pcd_sample_options["number_of_points"],
    #                                         init_factor=10)
    pcd1 = mesh1.sample_points_uniformly(number_of_points=pcd_sample_options["number_of_points"])
    pcd2 = mesh2.sample_points_uniformly(number_of_points=pcd_sample_options["number_of_points"])
    # 将点云进行归一化
    pcd1, pcd2, centroid, scale = normalize_point_cloud(pcd1, pcd2)

    # 从mesh获取单角度点云
    get_scan_pcd(specs, category, cur_filename, mesh1, mesh2, centroid, scale)
    return pcd1, pcd2, centroid, scale


def get_scan_pcd(specs, category, cur_filename, mesh1, mesh2, centroid, scale):
    scan_num = specs["scan_options"]["scan_num"]
    mesh1_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    mesh2_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    scene = o3d.t.geometry.RaycastingScene()
    mesh1_id = scene.add_triangles(mesh1_t)
    mesh2_id = scene.add_triangles(mesh2_t)

    # 将mesh归一化
    mesh1.translate(-centroid)
    mesh2.translate(-centroid)
    mesh1.scale(1 / scale, np.array([0, 0, 0]))
    mesh2.scale(1 / scale, np.array([0, 0, 0]))

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # 光线
    for i in range(scan_num):
        theta = 2 * math.pi * i / scan_num
        r = 2
        eye = [r * math.cos(theta), 1.0, r * math.sin(theta)]
        rays = scene.create_rays_pinhole(fov_deg=120,
                                         center=[0, 0, 0],
                                         eye=eye,
                                         up=[0, 1, 0],
                                         width_px=50,
                                         height_px=50)
        # paint_rays(rays, mesh1, mesh2, coordinate)
        cast_result = scene.cast_rays(rays)

        pcd1_scan, pcd2_scan = get_cur_pcd(rays, cast_result, eye)

        mesh1.compute_vertex_normals()
        mesh2.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh1, mesh2, pcd1_scan, pcd2_scan, coordinate])


def get_cur_pcd(rays, cast_result, eye):
    geometry_ids = cast_result["geometry_ids"]
    hit = cast_result['t_hit']
    rays_dis1 = []
    rays_dis2 = []
    rays_direction1 = []
    rays_direction2 = []
    for i in range(rays.shape[0]):
        for j in range(rays.shape[1]):
            if not math.isinf(hit[i][j]):
                if geometry_ids[i][j] == 0:
                    rays_direction1.append(rays[i][j])
                    rays_dis1.append(hit[i][j])
                if geometry_ids[i][j] == 1:
                    rays_direction2.append(rays[i][j])
                    rays_dis2.append(hit[i][j])

    points1 = [eye + rays_direction1[i]*rays_dis1[i] for i in range(rays_direction1.shape[0])]
    points2 = [eye + rays_direction2[i]*rays_dis2[i] for i in range(rays_direction2.shape[0])]
    # hit = cast_result["t_hit"].numpy().reshape(-1)
    # for i in range(rays.shape[0]):
    #     # if hit[i] != 'inf':
    #     if geometry_ids[i] == mesh1_id:
    #         points1.append(eye + rays[i][3:6] * hit[i])
    #     if geometry_ids[i] == mesh2_id:
    #         points2.append(get_triangle_centroid(vertices_mesh2, triangles_mesh2[primitive_ids[i]]))

    # # 取出投射结果中的集合体id和面片id
    # geometry_ids = cast_result["geometry_ids"].numpy().reshape(-1)
    # primitive_ids = cast_result["primitive_ids"].numpy().reshape(-1)
    # # 从原始mesh中取出顶点坐标和面片
    # triangles_mesh1 = np.asarray(mesh1.triangles)
    # triangles_mesh2 = np.asarray(mesh2.triangles)
    # vertices_mesh1 = np.asarray(mesh1.vertices)
    # vertices_mesh2 = np.asarray(mesh2.vertices)
    # points1 = []
    # points2 = []
    # for i in range(geometry_ids.shape[0]):
    #     if geometry_ids[i] == mesh1_id:
    #         points1.append(get_triangle_centroid(vertices_mesh1, triangles_mesh1[primitive_ids[i]]))
    #     if geometry_ids[i] == mesh2_id:
    #         points2.append(get_triangle_centroid(vertices_mesh2, triangles_mesh2[primitive_ids[i]]))
    #
    pcd1_scan = o3d.geometry.PointCloud()
    pcd2_scan = o3d.geometry.PointCloud()
    pcd1_scan.points = o3d.utility.Vector3dVector(np.array(points1))
    pcd2_scan.points = o3d.utility.Vector3dVector(np.array(points2))
    pcd1_scan.paint_uniform_color((1, 0, 0))
    pcd2_scan.paint_uniform_color((0, 1, 0))

    return pcd1_scan, pcd2_scan


def get_triangle_centroid(vertices, triangle):
    point1 = vertices[triangle[0]]
    point2 = vertices[triangle[1]]
    point3 = vertices[triangle[2]]

    triangle_centroid = np.mean(np.array([point1, point2, point3]), axis=0)
    return triangle_centroid


def get_visible_mesh(mesh1, mesh1_id, mesh2, mesh2_id, cast_result):
    triangles_mesh1 = np.asarray(mesh1.triangles)
    triangles_visible_mesh1 = []
    triangles_mesh2 = np.asarray(mesh2.triangles)
    triangles_visible_mesh2 = []
    geometry_ids = cast_result["geometry_ids"].numpy().reshape(-1)
    primitive_ids = cast_result["primitive_ids"].numpy().reshape(-1)
    for i in range(geometry_ids.shape[0]):
        if geometry_ids[i] == mesh1_id:
            triangles_visible_mesh1.append(triangles_mesh1[primitive_ids[i]])
        if geometry_ids[i] == mesh2_id:
            triangles_visible_mesh2.append(triangles_mesh2[primitive_ids[i]])
    mesh1.triangles = o3d.utility.Vector3iVector(np.array(triangles_visible_mesh1))
    mesh2.triangles = o3d.utility.Vector3iVector(np.array(triangles_visible_mesh2))
    mesh1.remove_duplicated_triangles()
    mesh2.remove_duplicated_triangles()
    mesh1.remove_duplicated_vertices()
    mesh2.remove_duplicated_vertices()
    mesh1.remove_degenerate_triangles()
    mesh2.remove_degenerate_triangles()
    pass


def paint_rays(rays, mesh1, mesh2, coordinate):
    rays = rays.numpy()
    eye = rays[0][0][0:3].reshape(1, 3)
    rays = rays[:, :, 3:6]
    rays_points = eye
    for i in range(rays.shape[0]):
        rays_points = np.concatenate((rays_points, rays[i]))
    lines = [[0, i] for i in range(1, rays_points.shape[0] - 1)]
    colors = [[1, 0, 0] for i in range(lines.__sizeof__())]
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(rays_points)

    o3d.visualization.draw_geometries([mesh1, mesh2, coordinate, lines_pcd])


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


def save_pcd(pcd1, pcd2, specs, category, cur_filename):
    pcd_dir = specs['pcd_path']

    # 获取点云名
    pcd1_filename = '{}_0.ply'.format(cur_filename)
    pcd2_filename = '{}_1.ply'.format(cur_filename)

    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_dir, category)):
        os.mkdir(os.path.join(pcd_dir, category))

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
def get_sdf_samples(specs, category, cur_filename, centroid, scale):
    mesh_dir = specs["mesh_path"]
    IOUgt_dir = specs["IOUgt_path"]
    sdf_sample_option = specs["SDF_sample_options"]
    mesh_filename_1 = "{}_0.off".format(cur_filename)
    mesh_filename_2 = "{}_1.off".format(cur_filename)
    IOUgt_filename = "{}.txt".format(cur_filename)

    # 获取mesh
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, mesh_filename_1))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, mesh_filename_2))

    # 获取两物体的aabb框和交互区域的aabb框，在各自范围内按比例散点
    aabb1, aabb2, aabb = getTwoMeshBorder(mesh1, mesh2)
    aabb_IOUgt = getAABBfromTwoPoints(os.path.join(IOUgt_dir, category, IOUgt_filename))

    random_points = getRandomPointsSeparately(aabb1, aabb2, aabb_IOUgt, sdf_sample_option)
    random_points -= centroid
    random_points /= scale
    return random_points


def get_sdf_value(pcd1, pcd2, sdf_samples):
    sdf_samples_pcd = o3d.geometry.PointCloud()
    sdf_samples_pcd.points = o3d.utility.Vector3dVector(sdf_samples)

    dists_1 = np.asarray(sdf_samples_pcd.compute_point_cloud_distance(pcd1)).reshape(-1, 1)
    dists_2 = np.asarray(sdf_samples_pcd.compute_point_cloud_distance(pcd2)).reshape(-1, 1)

    SDF_data = np.concatenate([sdf_samples, dists_1, dists_2], axis=1)
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


def getRandomPointsSeparately(aabb1, aabb2, aabb_IOU, sample_options: dict):
    """
    在aabb范围内按照sample_method规定的采样方法采样points_num个点
    """
    # 解析采样选项
    method = sample_options["method"]
    points_num = sample_options["points_num"]
    scale1 = sample_options["scale1"]
    scale2 = sample_options["scale2"]
    scale_IOU = sample_options["scale_IOU"]

    proportion1 = sample_options["proportion1"]
    proportion2 = sample_options["proportion2"]
    proportion_IOU = sample_options["proportion_IOU"]

    # 获取mesh1和mesh2的包围框边界点
    min_bound_mesh1 = aabb1.get_min_bound() * scale1
    max_bound_mesh1 = aabb1.get_max_bound() * scale1
    min_bound_mesh2 = aabb2.get_min_bound() * scale2
    max_bound_mesh2 = aabb2.get_max_bound() * scale2
    min_bound_mesh_IOUgt = aabb_IOU.get_min_bound() * scale_IOU
    max_bound_mesh_IOUgt = aabb_IOU.get_max_bound() * scale_IOU

    random_points_mesh1 = []
    random_points_mesh2 = []
    random_points_mesh_IOUgt = []

    if method == 'uniform':
        for i in range(3):
            random_points_mesh1.append(randUniFormFloat(min_bound_mesh1[i], max_bound_mesh1[i],
                                                        int(points_num * proportion1)).reshape((-1, 1)))
            random_points_mesh2.append(randUniFormFloat(min_bound_mesh2[i], max_bound_mesh2[i],
                                                        int(points_num * proportion2)).reshape((-1, 1)))
            random_points_mesh_IOUgt.append(randUniFormFloat(min_bound_mesh_IOUgt[i], max_bound_mesh_IOUgt[i],
                                                             int(points_num * proportion_IOU)).reshape((-1, 1)))
    elif method == 'normal':
        for i in range(3):
            random_points_mesh1.append(randNormalFloat(min_bound_mesh1[i], max_bound_mesh1[i],
                                                       int(points_num * proportion1)).reshape((-1, 1)))
            random_points_mesh2.append(randNormalFloat(min_bound_mesh2[i], max_bound_mesh2[i],
                                                       int(points_num * proportion2)).reshape((-1, 1)))
            random_points_mesh_IOUgt.append(randNormalFloat(min_bound_mesh_IOUgt[i], max_bound_mesh_IOUgt[i],
                                                            int(points_num * proportion_IOU)).reshape((-1, 1)))

    random_points_mesh1_ = np.concatenate([random_points_mesh1[0], random_points_mesh1[1], random_points_mesh1[2]],
                                          axis=1)
    random_points_mesh2_ = np.concatenate([random_points_mesh2[0], random_points_mesh2[1], random_points_mesh2[2]],
                                          axis=1)
    random_points_mesh_IOUgt_ = np.concatenate(
        [random_points_mesh_IOUgt[0], random_points_mesh_IOUgt[1], random_points_mesh_IOUgt[2]],
        axis=1)

    return np.concatenate([random_points_mesh1_, random_points_mesh2_, random_points_mesh_IOUgt_], axis=0)


def save_sdf(SDF_data, specs, category, cur_filename):
    sdf_dir = specs['sdf_path']
    # 目录不存在则创建
    if not os.path.isdir(os.path.join(sdf_dir, category)):
        os.mkdir(os.path.join(sdf_dir, category))

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
    # 获取归一化点云和质心坐标、缩放系数
    # 采集多视角的单角度扫描点云，使用完整点云的归一化参数
    time1 = datetime.datetime.now()
    pcd1, pcd2, centroid, scale = get_pcd(specs, category, cur_filename)
    time2 = datetime.datetime.now()
    print('get_pcd: ', (time2 - time1).seconds)

    # 在归一化坐标内进行采样，计算采样点的SDF值
    # 所有单角度点云与完整点云共用sdf sample点
    time1 = datetime.datetime.now()
    sdf_samples = get_sdf_samples(specs, category, cur_filename, centroid, scale)
    time2 = datetime.datetime.now()
    print('get_sdf_samples: ', (time2 - time1).seconds)

    time1 = datetime.datetime.now()
    SDF_data = get_sdf_value(pcd1, pcd2, sdf_samples)
    time2 = datetime.datetime.now()
    print('get_sdf_value: ', (time2 - time1).seconds)

    # 可视化
    if specs['visualization']:
        visualization(pcd1, pcd2, SDF_data)
    # 保存点云
    save_pcd(pcd1, pcd2, specs, category, cur_filename)
    # 保存SDF_data
    save_sdf(SDF_data, specs, category, cur_filename)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generateScanData.json'
    specs = parseConfig(configFile_path)
    filename_re = specs['mesh_filename_re']

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["pcd_path"]):
        os.mkdir(specs["pcd_path"])
    if not os.path.isdir(specs["sdf_path"]):
        os.mkdir(specs["sdf_path"])

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
