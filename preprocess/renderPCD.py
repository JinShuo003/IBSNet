"""
渲染残缺点云和ibs面的图像，检查数据是否合理
"""
import open3d as o3d
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt


def parseConfig(config_filepath: str = './config/visualization.json'):
    with open(config_filepath, 'r') as configfile:
        specs = json.load(configfile)

    return specs


def getFilenameTree(specs: dict):
    # 以SDF network作为基准构建文件树
    sdf_path = specs["sdf_dir"]
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    filename_re = specs["filename_re"]

    filename_tree = dict()
    folder_info = os.walk(sdf_path)
    for dir_path, dir_names, filenames in folder_info:
        # 当前是顶级目录，不做处理
        if dir_path == sdf_path:
            continue
        # 获取当前文件夹的类目信息，合法则继续处理
        category = dir_path.split('\\')[-1]
        if not re.match(category_re, category):
            continue
        # 当前类目不在filename_tree中，添加
        if not category in filename_tree:
            filename_tree[category] = dict()
        for filename in filenames:
            if not re.match(filename_re, filename):
                continue
            # 获取场景名
            scene = re.match(scene_re, filename)
            # 如果与场景re不匹配则跳过
            if not scene:
                continue
            scene = scene.group()
            # 当前场景不在filename_tree[category]中，添加
            if not scene in filename_tree[category]:
                filename_tree[category][scene] = list()
            filename_tree[category][scene].append(filename)

    return filename_tree


def getGeometriesPath(specs, filename):
    category_re = specs["category_re"]
    filename_re = specs["filename_re"]
    category = re.match(category_re, filename).group()
    filename = re.match(filename_re, filename).group()

    geometries_path = dict()

    pcd_dir = specs['pcd_dir']
    sdf_dir = specs['sdf_dir']

    pcd1_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_filename = '{}_{}.ply'.format(filename, 1)
    sdf_filename = '{}.npz'.format(filename)

    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)
    geometries_path['sdf_scan'] = os.path.join(sdf_dir, category, sdf_filename)

    return geometries_path


def get_pcd(specs, pcd_filepath, color_key):
    pcd = o3d.io.read_point_cloud(pcd_filepath)
    pcd.paint_uniform_color(specs['visualization_options']['colors'][color_key])

    return pcd


def get_surface_points(specs, sdf_filepath):
    npz = np.load(sdf_filepath)

    data = npz["data"]
    surface_points1 = [points[0:3] for points in data if
                       abs(points[3]) < specs['visualization_options']['sdf_threshold']]
    surface_points2 = [points[0:3] for points in data if
                       abs(points[4]) < specs['visualization_options']['sdf_threshold']]
    surface_points_ibs = [points[0:3] for points in data if
                          abs(points[3] - points[4]) < specs['visualization_options']['sdf_threshold']]

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd_ibs = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(surface_points1)
    pcd2.points = o3d.utility.Vector3dVector(surface_points2)
    pcd_ibs.points = o3d.utility.Vector3dVector(surface_points_ibs)

    pcd1.paint_uniform_color(specs['visualization_options']['colors']['pcd1'])
    pcd2.paint_uniform_color(specs['visualization_options']['colors']['pcd2'])
    pcd_ibs.paint_uniform_color(specs['visualization_options']['colors']['ibs'])

    return pcd1, pcd2, pcd_ibs


class visualizer:
    def __init__(self, specs):
        self.specs = specs

    def handle_current_file(self, specs, filename):
        geometries_path = getGeometriesPath(specs, filename)
        pcd1 = get_pcd(specs, geometries_path['pcd1'], 'pcd1')
        pcd2 = get_pcd(specs, geometries_path['pcd2'], 'pcd2')
        _, _, ibs_scan = get_surface_points(specs, geometries_path['sdf_scan'])
        self.save_to_img(pcd1, pcd2, ibs_scan, filename)

    def save_to_img(self, pcd1, pcd2, ibs, filename):
        img_save_dir = specs["img_save_dir"]
        category_re = specs["category_re"]
        filename_re = specs["filename_re"]
        category = re.match(category_re, filename).group()
        filename = re.match(filename_re, filename).group()

        # 创建保存路径+类别文件夹
        img_save_path = os.path.join(img_save_dir, category)
        if not os.path.isdir(img_save_path):
            os.makedirs(img_save_path)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # 将背景颜色设置为白色
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)
        vis.add_geometry(ibs)
        vis.poll_events()
        vis.update_renderer()
        # 保存图像
        vis.capture_screen_image("{}.png".format(os.path.join(img_save_path, filename)))
        vis.destroy_window()


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/renderPCD.json'
    specs = parseConfig(config_filepath)

    filename_tree = getFilenameTree(specs)

    v = visualizer(specs)
    for category in filename_tree:
        chamfer_distance_scan = 0
        chamfer_distance_network = 0
        for scene in filename_tree[category]:
            print('current scene: ', scene)
            for filename in filename_tree[category][scene]:
                v.handle_current_file(specs, filename)
