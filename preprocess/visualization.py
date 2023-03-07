import open3d as o3d
import os
import re
import json

filename_re = 'scene\d\.\d{4}_\d'
filepath_re = '.*scene\d\.\d{4}_\d.*'


def parseConfig(config_filepath: str = './config/visualization.json'):
    with open(config_filepath, 'r') as configfile:
        specs = json.load(configfile)

    return specs


def visualize(specs, category, filename_intersection):
    mesh_path = specs['mesh_path']
    pcd_path = specs['pcd_path']
    sdf_path = specs['sdf_path']
    IOUgt_path = specs['IOUgt_path']
    mesh_path = specs['mesh_path']

    mesh1_filename = filename_intersection + '_{}.off'.format(0)
    mesh1_filename = filename_intersection + '_{}.off'.format(0)

    mesh1_path = os.path.join(mesh_path, category, )
    mesh2_path = os.path.join(mesh_path, category, filename_intersection + '_{}.off'.format(1))
    pcd_path = os.path.join(pcd_path, category, filename_intersection + '_{}.off'.format(1))


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/visualization.json'
    specs = parseConfig(config_filepath)

    for category in specs['categories']:
        category_dir = os.path.join(specs["IOUgt_path"], category)
        # 列出当前类别目录下所有文件名
        filename_list = os.listdir(category_dir)
        for filename in filename_list:
            # 得到公共部分的名字
            filename_intersection = re.match(specs['filename_re'], filename).group()

            # 跳过不匹配正则式的文件
            if re.match(specs["process_filename_re"], filename_intersection) is None:
                continue

            print('current file: ', filename_intersection)
            visualize(specs, category, filename_intersection)
