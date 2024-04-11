"""
绘制残缺程度-重建cd散点图，用来说明不同方法对残缺程度的敏感性
"""

import os
import re

import seaborn
import matplotlib.pyplot as plt

from utils import path_utils


def getGeometriesPath(specs, instance_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, instance_name).group()
    filename = re.match(filename_re, instance_name).group()
    geometries_path = dict()

    incomplete_cd_dir = specs.get("path_options").get("incomplete_cd_dir")
    reconstruct_cd_geometric_dir = specs.get("path_options").get("reconstruct_cd_geometric_dir")
    reconstruct_cd_grasping_field_dir = specs.get("path_options").get("reconstruct_cd_grasping_field_dir")
    reconstruct_cd_IBSNet_dir = specs.get("path_options").get("reconstruct_cd_IBSNet_dir")

    incomplete1_cd_filename = '{}_0.txt'.format(filename)
    incomplete2_cd_filename = '{}_1.txt'.format(filename)
    reconstruct_cd_filename = '{}.txt'.format(filename)

    geometries_path['incomplete1_cd'] = os.path.join(incomplete_cd_dir, category, incomplete1_cd_filename)
    geometries_path['incomplete2_cd'] = os.path.join(incomplete_cd_dir, category, incomplete2_cd_filename)
    geometries_path['reconstruct_cd_geometric'] = os.path.join(reconstruct_cd_geometric_dir, category, reconstruct_cd_filename)
    geometries_path['reconstruct_cd_grasping_field'] = os.path.join(reconstruct_cd_grasping_field_dir, category, reconstruct_cd_filename)
    geometries_path['reconstruct_cd_IBSNet'] = os.path.join(reconstruct_cd_IBSNet_dir, category, reconstruct_cd_filename)

    return geometries_path


def get_data(specs, instance_name):
    geometry_path = getGeometriesPath(specs, instance_name)

    with open(geometry_path.get('incomplete1_cd'), 'r') as f:
        incomplete1_cd = float(f.read())
    with open(geometry_path.get('incomplete2_cd'), 'r') as f:
        incomplete2_cd = float(f.read())
    with open(geometry_path.get('reconstruct_cd_geometric'), 'r') as f:
        reconstruct_cd_geometric = float(f.read())
    with open(geometry_path.get('reconstruct_cd_grasping_field'), 'r') as f:
        reconstruct_cd_grasping_field = float(f.read())
    with open(geometry_path.get('reconstruct_cd_IBSNet'), 'r') as f:
        reconstruct_cd_IBSNet = float(f.read())

    return incomplete1_cd, incomplete2_cd, reconstruct_cd_geometric, reconstruct_cd_grasping_field, reconstruct_cd_IBSNet


if __name__ == '__main__':
    config_filepath = 'configs/generate_histogram.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("base_dir"))

    # 参数
    incomplete_cd_geometric_list = []
    incomplete_cd_grasping_field_list = []
    incomplete_cd_IBSNet_list = []
    reconstruct_cd_geometric_list = []
    reconstruct_cd_grasping_field_list = []
    reconstruct_cd_IBSNet_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                try:
                    incomplete1_cd, incomplete2_cd, reconstruct_cd_geometric, reconstruct_cd_grasping_field, reconstruct_cd_IBSNet = get_data(specs, filename)
                    if reconstruct_cd_geometric != 0.0:
                        incomplete_cd_geometric_list.append(incomplete1_cd)
                        reconstruct_cd_geometric_list.append(reconstruct_cd_geometric)
                    if reconstruct_cd_grasping_field != 0.0:
                        incomplete_cd_grasping_field_list.append(incomplete1_cd)
                        reconstruct_cd_grasping_field_list.append(reconstruct_cd_grasping_field)
                    if reconstruct_cd_IBSNet != 0.0:
                        incomplete_cd_IBSNet_list.append(incomplete1_cd)
                        reconstruct_cd_IBSNet_list.append(reconstruct_cd_IBSNet)
                except Exception as e:
                    print(e)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.set_title('geometric')
    seaborn.scatterplot(x=incomplete_cd_geometric_list, y=reconstruct_cd_geometric_list, ax=ax1)
    ax2.set_title('grasping_field')
    seaborn.scatterplot(x=incomplete_cd_grasping_field_list, y=reconstruct_cd_grasping_field_list, ax=ax2)
    ax3.set_title('IBSNet')
    seaborn.scatterplot(x=incomplete_cd_IBSNet_list, y=reconstruct_cd_IBSNet_list, ax=ax3)

    plt.show()
