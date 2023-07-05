"""
1. 从mesh进行采样，得到完整点云和归一化参数
2. 利用归一化参数在交互区域内和两物体点云的aabb框内随机散点
3. 计算这些点在两物体完整点云sdf场内的sdf值，该值作为间接法网络的gt
4. 保留这些点中sdf之差小于阈值的那部分点（认为位于ibs面上），然后计算所有点在ibs面的sdf场内的sdf值，该值作为直接法网络的gt
5. 从n个视角对完整点云进行扫描，得到n个角度的残缺点云数据，作为网络的输入
6. 计算这些点在两物体残缺点云sdf场内的sdf值，作为对比数据
"""
import copy
import os
import re
import open3d as o3d
import numpy as np
from ordered_set import OrderedSet
import multiprocessing
import json


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def getFilenameTree(specs: dict, base_path: str):
    """以base_path为基准构建文件树，文件树的格式为
    {
    'scene1': ['scene1.1000',
               'scene1.1001'],
    'scene2': ['scene1.1011',
               'scene1.1012']}
    """
    # 构建文件树
    base_path = specs[base_path]
    scene_re = specs["scene_re"]

    filename_tree = dict()
    folder_info = os.walk(base_path)
    for dir_path, dir_names, filenames in folder_info:
        # 顶级目录不做处理
        if dir_path == base_path:
            continue
        category = dir_path.split('\\')[-1]
        if not regular_match(specs["category_re"], category):
            continue
        if not category in filename_tree:
            filename_tree[category] = OrderedSet()
        for filename in filenames:
            if not regular_match(specs["scene_re"], filename):
                continue
            filename = re.match(scene_re, filename).group()
            filename_tree[category].add(filename)
    # 将有序集合转为列表
    for key in filename_tree.keys():
        filename_tree[key] = list(filename_tree[key])
    return filename_tree


def generatePath(specs: dict, path_list: list):
    """检查specs中的path是否存在，不存在则创建"""
    for path in path_list:
        if not os.path.isdir(specs[path]):
            os.makedirs(specs[path])


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)


def getGeometriesPath(specs, scene):
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    category = re.match(category_re, scene).group()
    scene = re.match(scene_re, scene).group()

    geometries_path = dict()

    mesh_dir = specs["mesh_dir"]
    IOUgt_dir = specs["IOUgt_dir"]

    mesh1_filename = '{}_{}.off'.format(scene, 0)
    mesh2_filename = '{}_{}.off'.format(scene, 1)
    IOUgt_filename = '{}.txt'.format(scene)

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometries_path['IOUgt'] = os.path.join(IOUgt_dir, category, IOUgt_filename)

    return geometries_path


def save_mesh(specs, scene, mesh1, mesh2):
    mesh_dir = specs['mesh_normalize_save_dir']
    category = re.match(specs['category_re'], scene).group()
    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    mesh1_filename = '{}_0.obj'.format(scene)
    mesh2_filename = '{}_1.obj'.format(scene)
    mesh1_path = os.path.join(mesh_dir, category, mesh1_filename)
    mesh2_path = os.path.join(mesh_dir, category, mesh2_filename)

    o3d.io.write_triangle_mesh(mesh1_path, mesh1)
    o3d.io.write_triangle_mesh(mesh2_path, mesh2)


def save_IOU(specs, scene, aabb_IOU):
    IOU_dir = specs['IOUgt_dir_normalize_save_dir']
    category = re.match(specs['category_re'], scene).group()
    if not os.path.isdir(os.path.join(IOU_dir, category)):
        os.makedirs(os.path.join(IOU_dir, category))

    aabb_IOU_obj_filename = '{}.obj'.format(scene)
    aabb_IOU_txt_filename = '{}.npy'.format(scene)
    aabb_IOU_obj_path = os.path.join(IOU_dir, category, aabb_IOU_obj_filename)
    aabb_IOU_txt_path = os.path.join(IOU_dir, category, aabb_IOU_txt_filename)

    # 保存aabb_mesh
    aabb_mesh = aabb2mesh(aabb_IOU)
    o3d.io.write_triangle_mesh(aabb_IOU_obj_path, aabb_mesh)

    # 保存aabb_txt
    min_bound = aabb_IOU.get_min_bound()
    max_bound = aabb_IOU.get_max_bound()
    np.save(aabb_IOU_txt_path, np.array([min_bound, max_bound]))


def aabb2mesh(aabb):
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    # 定义AABB框的八个顶点坐标
    aabb_vertices = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],  # 顶点1
        [min_bound[0], min_bound[1], max_bound[2]],  # 顶点2
        [min_bound[0], max_bound[1], min_bound[2]],  # 顶点3
        [min_bound[0], max_bound[1], max_bound[2]],  # 顶点4
        [max_bound[0], min_bound[1], min_bound[2]],  # 顶点5
        [max_bound[0], min_bound[1], max_bound[2]],  # 顶点6
        [max_bound[0], max_bound[1], min_bound[2]],  # 顶点7
        [max_bound[0], max_bound[1], max_bound[2]]  # 顶点8
    ])

    # 定义AABB框的面索引
    aabb_faces = np.array([
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3]
    ])

    # 创建Mesh对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(aabb_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(aabb_faces)
    return mesh


class GeometryHandler:
    def __init__(self):
        pass

    def get_pcd_normalize_para(self, pcd):
        pcd_np = np.asarray(pcd.points)
        # 求点云的中心
        centroid = np.mean(pcd_np, axis=0)
        # 求长轴长度
        scale = np.max(np.sqrt(np.sum(pcd_np ** 2, axis=1)))
        return centroid, scale

    def geometry_transform(self, geometry, centroid, scale):
        coor = self.get_unit_coordinate()
        geometry.translate(-centroid)
        geometry.scale(1 / scale, np.array([0, 0, 0]))

    def get_unit_sphere(self):
        # 创建单位球点云
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.compute_vertex_normals()
        return mesh_sphere

    def get_unit_coordinate(self):
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        coord_frame.compute_vertex_normals()
        return coord_frame

    def get_unit_sphere_pcd(self):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        pcd = mesh_sphere.sample_points_uniformly(256)
        return pcd


class TrainDataGenerator:
    def __init__(self, specs):
        self.geometryHandler = GeometryHandler()
        self.specs = specs
        self.geometries_path = None

    def combine_meshes(self, mesh1, mesh2):
        # 获取第一个Mesh的顶点和面数据
        vertices1 = mesh1.vertices
        faces1 = np.asarray(mesh1.triangles)

        # 获取第二个Mesh的顶点和面数据
        vertices2 = mesh2.vertices
        faces2 = np.asarray(mesh2.triangles)

        # 将第二个Mesh的顶点坐标添加到第一个Mesh的顶点列表中
        combined_vertices = np.concatenate((vertices1, vertices2))

        # 更新第二个Mesh的面索引，使其适应顶点索引的变化
        faces2 += len(vertices1)

        # 将两个Mesh的面数据合并
        combined_faces = np.concatenate((faces1, faces2))

        # 创建一个新的Mesh对象
        combined_mesh = o3d.geometry.TriangleMesh()

        # 设置新的Mesh的顶点和面数据
        combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
        combined_mesh.triangles = o3d.utility.Vector3iVector(combined_faces)

        return combined_mesh

    def get_normalize_para(self, mesh):
        aabb = mesh.get_axis_aligned_bounding_box()
        centroid = aabb.get_center()
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        scale = np.linalg.norm(max_bound - min_bound)
        return centroid, scale/0.8

    def get_IOU(self):
        """获取交互区域的aabb框"""
        with open(self.geometries_path["IOUgt"], 'r') as file:
            data = file.readlines()
            line1 = data[0].strip('\n').strip(' ').split(' ')
            line2 = data[1].strip('\n').strip(' ').split(' ')
            min_bound = np.array([float(item) for item in line1])
            max_bound = np.array([float(item) for item in line2])
            return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    def handle_scene(self, scene):
        """读取mesh，组合后求取归一化参数，然后分别归一化到单位球内，保存结果"""
        self.geometries_path = getGeometriesPath(self.specs, scene)

        mesh1 = o3d.io.read_triangle_mesh(self.geometries_path["mesh1"])
        mesh2 = o3d.io.read_triangle_mesh(self.geometries_path["mesh2"])
        combined_mesh = self.combine_meshes(copy.deepcopy(mesh1), copy.deepcopy(mesh2))
        centroid, scale = self.get_normalize_para(combined_mesh)

        aabb_IOU = self.get_IOU()
        self.geometryHandler.geometry_transform(mesh1, centroid, scale)
        self.geometryHandler.geometry_transform(mesh2, centroid, scale)
        self.geometryHandler.geometry_transform(aabb_IOU, centroid, scale)

        # combined_mesh = self.combine_meshes(copy.deepcopy(mesh1), copy.deepcopy(mesh2))
        # aabb = combined_mesh.get_axis_aligned_bounding_box()
        # aabb.color = (1, 0, 0)
        # sphere = self.geometryHandler.get_unit_sphere_pcd()
        # coor = self.geometryHandler.get_unit_coordinate()
        # o3d.visualization.draw_geometries([sphere, mesh1, mesh2, aabb, coor])

        save_mesh(self.specs, scene, mesh1, mesh2)
        save_IOU(self.specs, scene, aabb_IOU)


def my_process(scene, specs):
    # 获取当前进程信息
    process_name = multiprocessing.current_process().name
    # 执行任务函数的逻辑
    print(f"Running task in process: {process_name}, scene: {scene}")
    # 其他任务操作
    trainDataGenerator = TrainDataGenerator(specs)
    try:
        trainDataGenerator.handle_scene(scene)
        print(f"scene: {scene} succeed")
    except:
        print(f"scene: {scene} failed")


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/normalizeMesh.json'
    specs = parseConfig(config_filepath)
    # 构建文件树
    filename_tree = getFilenameTree(specs, "mesh_dir")
    # 处理文件夹，不存在则创建
    generatePath(specs, ["mesh_normalize_save_dir"])

    # 创建进程池，指定进程数量
    pool = multiprocessing.Pool(processes=10)
    # 参数
    scene_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            scene_list.append(scene)
    # 使用进程池执行任务，返回结果列表
    for scene in scene_list:
        pool.apply_async(my_process, (scene, specs,))

    # 关闭进程池
    pool.close()
    pool.join()
