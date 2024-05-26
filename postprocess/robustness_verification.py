"""
绘制输入点云残缺程度-cd值的散点图，说明不同方法对残缺程度的敏感性
"""

import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from utils import path_utils


def getGeometriesPath(specs, instance_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, instance_name).group()
    filename = re.match(filename_re, instance_name).group()
    geometries_path = dict()

    incomplete_cd_dir = specs.get("path_options").get("incomplete_cd_dir")
    reconstruct_cd_geometric_dir = specs.get("path_options").get("reconstruct_cd_geometric_dir")
    reconstruct_cd_IMNet_dir = specs.get("path_options").get("reconstruct_cd_IMNet_dir")
    reconstruct_cd_grasping_field_dir = specs.get("path_options").get("reconstruct_cd_grasping_field_dir")
    reconstruct_cd_IBSNet_dir = specs.get("path_options").get("reconstruct_cd_IBSNet_dir")

    incomplete1_cd_filename = '{}_0.txt'.format(filename)
    incomplete2_cd_filename = '{}_1.txt'.format(filename)
    reconstruct_cd_filename = '{}.txt'.format(filename)

    geometries_path['incomplete1_cd'] = os.path.join(incomplete_cd_dir, category, incomplete1_cd_filename)
    geometries_path['incomplete2_cd'] = os.path.join(incomplete_cd_dir, category, incomplete2_cd_filename)
    geometries_path['reconstruct_cd_geometric'] = os.path.join(reconstruct_cd_geometric_dir, category,
                                                               reconstruct_cd_filename)
    geometries_path['reconstruct_cd_IMNet'] = os.path.join(reconstruct_cd_IMNet_dir, category,
                                                               reconstruct_cd_filename)
    geometries_path['reconstruct_cd_grasping_field'] = os.path.join(reconstruct_cd_grasping_field_dir, category,
                                                                    reconstruct_cd_filename)
    geometries_path['reconstruct_cd_IBSNet'] = os.path.join(reconstruct_cd_IBSNet_dir, category,
                                                            reconstruct_cd_filename)

    return geometries_path


def get_data(specs, instance_name):
    geometry_path = getGeometriesPath(specs, instance_name)

    with open(geometry_path.get('incomplete1_cd'), 'r') as f:
        incomplete1_cd = float(f.read())
    with open(geometry_path.get('incomplete2_cd'), 'r') as f:
        incomplete2_cd = float(f.read())
    with open(geometry_path.get('reconstruct_cd_geometric'), 'r') as f:
        reconstruct_cd_geometric = float(f.read())
    with open(geometry_path.get('reconstruct_cd_IMNet'), 'r') as f:
        reconstruct_cd_IMNet = float(f.read())
    with open(geometry_path.get('reconstruct_cd_grasping_field'), 'r') as f:
        reconstruct_cd_grasping_field = float(f.read())
    with open(geometry_path.get('reconstruct_cd_IBSNet'), 'r') as f:
        reconstruct_cd_IBSNet = float(f.read())

    return (incomplete1_cd, incomplete2_cd, reconstruct_cd_geometric,
            reconstruct_cd_IMNet, reconstruct_cd_grasping_field, reconstruct_cd_IBSNet)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    # sns.set_style('darkgrid')

    my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/simsun.ttc")
    config_filepath = 'configs/generate_histogram.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("base_dir"))
    base = '2'

    # 参数
    incomplete_cd_geometric_list = []
    incomplete_cd_IMNet_list = []
    incomplete_cd_grasping_field_list = []
    incomplete_cd_IBSNet_list = []
    reconstruct_cd_geometric_list = []
    reconstruct_cd_IMNet_list = []
    reconstruct_cd_grasping_field_list = []
    reconstruct_cd_IBSNet_list = []

    bar_num = 7
    cd_total_geometric = [0] * bar_num
    cd_total_IMNet = [0] * bar_num
    cd_total_grasping_field = [0] * bar_num
    cd_total_IBSNet = [0] * bar_num
    instance_num_geometric = [0] * bar_num
    instance_num_IMNet = [0] * bar_num
    instance_num_grasping_field = [0] * bar_num
    instance_num_IBSNet = [0] * bar_num

    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                try:
                    (incomplete1_cd, incomplete2_cd, reconstruct_cd_geometric, reconstruct_cd_IMNet,
                     reconstruct_cd_grasping_field, reconstruct_cd_IBSNet) = get_data(specs, filename)
                    incomplete_cd = incomplete1_cd if base == '1' else incomplete2_cd
                    idx = int(incomplete_cd * 40)
                    if idx >= bar_num:
                        continue
                    if reconstruct_cd_geometric != 0.0:
                        cd_total_geometric[idx] += reconstruct_cd_geometric
                        instance_num_geometric[idx] += 1
                        incomplete_cd_geometric_list.append(incomplete_cd)
                        reconstruct_cd_geometric_list.append(reconstruct_cd_geometric)
                    if reconstruct_cd_IMNet != 0.0:
                        cd_total_IMNet[idx] += reconstruct_cd_IMNet
                        instance_num_IMNet[idx] += 1
                        incomplete_cd_IMNet_list.append(incomplete_cd)
                        reconstruct_cd_IMNet_list.append(reconstruct_cd_IMNet)
                    if reconstruct_cd_grasping_field != 0.0:
                        cd_total_grasping_field[idx] += reconstruct_cd_grasping_field
                        instance_num_grasping_field[idx] += 1
                        incomplete_cd_grasping_field_list.append(incomplete_cd)
                        reconstruct_cd_grasping_field_list.append(reconstruct_cd_grasping_field)
                    if reconstruct_cd_IBSNet != 0.0:
                        cd_total_IBSNet[idx] += reconstruct_cd_IBSNet
                        instance_num_IBSNet[idx] += 1
                        incomplete_cd_IBSNet_list.append(incomplete_cd)
                        reconstruct_cd_IBSNet_list.append(reconstruct_cd_IBSNet)
                except Exception as e:
                    print(e)

    cd_total_geometric = [a / b for a, b in zip(cd_total_geometric, instance_num_geometric)]
    cd_total_IMNet = [a / b for a, b in zip(cd_total_IMNet, instance_num_IMNet)]
    cd_total_grasping_field = [a / b for a, b in zip(cd_total_grasping_field, instance_num_grasping_field)]
    cd_total_IBSNet = [a / b for a, b in zip(cd_total_IBSNet, instance_num_IBSNet)]

    x_label = list(range(bar_num))

    data = {'method': ['几何计算'] * bar_num + ['IMNet'] * bar_num + ['Grasping Field'] * bar_num + ['IBSNet'] * bar_num,
            'ibs_accuracy_cd': cd_total_geometric + cd_total_IMNet + cd_total_grasping_field + cd_total_IBSNet,
            'incomplete_level': x_label + x_label + x_label + x_label}
    tag_list = [str(i) for i in range(bar_num)]

    fig = sns.lineplot(x='incomplete_level', y='ibs_accuracy_cd', hue='method', data=data, linewidth=5)

    incompletion_level_mapping = '输入点云残缺程度等级对应区间\n' + '\n'.join(['{}: ({}, {})'.format(i + 1, float(i) / 40, float(i+1) / 40) for i in range(bar_num)])
    plt.xlabel('输入点云残缺程度等级', fontproperties=my_font, fontsize=20)
    plt.ylabel('重建交互平分面到真实值的单向倒角距离', fontproperties=my_font, fontsize=20)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.legend(prop=my_font)
    fig.set_xlim(0, 6.2)
    fig.set_ylim(0.0, 0.055)

    plt.show()
