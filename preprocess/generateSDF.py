"""
从mesh生成sdf groundtruth的工具，配置好./config/generateSDF.json后可以按场景生成sdf的groundtruth
"""
import os
import re
import open3d as o3d
from utils import *
import json


def parseConfig(config_filepath: str = './config/generateSDF.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def generateSDF(specs: dict, category: str, current_pair_name):
    """
    在mesh1和mesh2组成的空间下进行随机散点，计算每个点对于两个mesh的sdf值，生成(x, y, z, sdf1, sdf2)后存储到pcd_dir下
    """
    mesh_dir = specs["mesh_path"]
    IOUgt_dir = specs["IOUgt_path"]
    sample_option = specs["sample_options"]
    mesh_filename_1 = "{}_0.off".format(current_pair_name)
    mesh_filename_2 = "{}_1.off".format(current_pair_name)
    IOUgt_filename = "{}.txt".format(current_pair_name)

    # 获取mesh
    mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, mesh_filename_1))
    mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, category, mesh_filename_2))

    # 获取两物体的aabb框和交互区域的aabb框，在各自范围内按比例散点
    aabb1, aabb2, aabb = getTwoMeshBorder(mesh1, mesh2)
    aabb_IOUgt = getAABBfromTwoPoints(os.path.join(IOUgt_dir, category, IOUgt_filename))

    random_points = getRandomPointsSeparately(aabb1, aabb2, aabb_IOUgt, sample_option)

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

    # 从文件中查询当前场景的缩放系数，将坐标值和sdf值同时除以缩放系数
    scale_filename = '{}_scale.txt'.format(category)
    scale_path = os.path.join(specs['pcd_path'], category, scale_filename)
    scale = 1
    centroid = []
    with open(scale_path, 'r') as scale_file:
        scale_data = scale_file.readlines()
        for line in scale_data:
            if re.match(current_pair_name, line) is not None:
                centroid = line.split(',')[1]
                centroid = [float(item) for item in re.findall('\\d*\\.\\d*', centroid)]
                centroid = np.array(centroid).reshape(3)
                scale = float(line.split(',')[2])
                break
    # 拼接查询点和两个SDF值
    random_points -= centroid
    SDF_data = np.concatenate([random_points, signed_distance1, signed_distance2], axis=1)

    SDF_data /= scale
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
    random_points_mesh_IOUgt_ = np.concatenate([random_points_mesh_IOUgt[0], random_points_mesh_IOUgt[1], random_points_mesh_IOUgt[2]],
                                          axis=1)

    return np.concatenate([random_points_mesh1_, random_points_mesh2_, random_points_mesh_IOUgt_], axis=0)


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
    configFile_path = 'config/generatePCDandSDF.json'
    specs = parseConfig(configFile_path)

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["sdf_path"]):
        os.mkdir(specs["sdf_path"])

    # mesh_dir + categories[i]下保存了对应场景的mesh数据、
    for category in specs['categories']:
        category_dir = os.path.join(specs["mesh_path"], category)
        # 列出当前类别目录下所有文件名
        filename_list = os.listdir(category_dir)

        # 记录已处理过的文件名
        handled_filename_list = set()
        for filename in filename_list:
            file_absPath = os.path.join(category_dir, filename)
            #  跳过非文件
            if not os.path.isfile(file_absPath):
                continue
            # 跳过不匹配正则式的文件
            if re.match(specs["process_filename_re"], filename) is None:
                continue

            # 数据成对出现，处理完一对后将前缀记录到map中，防止重复处理
            current_pair_name = re.match(specs["mesh_filename_re"], filename).group()
            if current_pair_name in handled_filename_list:
                continue
            else:
                handled_filename_list.add(current_pair_name)

            print('current file: ', filename)

            SDF_data = generateSDF(specs, category, current_pair_name)
            sdf_filename = "{}.npz".format(current_pair_name)
            saveSDFData(specs["sdf_path"], category, sdf_filename, SDF_data)
