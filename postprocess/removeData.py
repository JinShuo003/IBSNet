"""手工剔除不合理数据，根据类别名和场景名在各个目录下查找，满足正则则删除"""

import os
import re

import json


def parseConfig(config_filepath: str = './config/generatePointCloud.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def generatePath(specs: dict):
    """返回待检查的各个路径"""
    path_list = []
    category = specs["category_re"]
    path_list.append(os.path.join(specs["pcd_partial_save_dir"], category))
    path_list.append(os.path.join(specs["sdf_indirect_complete_save_dir"], category))
    path_list.append(os.path.join(specs["sdf_indirect_partial_save_dir"], category))
    path_list.append(os.path.join(specs["sdf_direct_save_dir"], category))
    return path_list


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/removeData.json'
    specs = parseConfig(config_filepath)
    # 获取待处理的文件夹
    path_list = generatePath(specs)

    # 检查各个文件夹
    for path in path_list:
        print("current path: ", path)
        # 检查文件夹下的所有文件
        filename_list = os.listdir(path)
        for filename in filename_list:
            # 满足条件，删除
            if re.match(specs["scene_re"], filename):
                print("filename: ", filename)
                os.remove(os.path.join(path, filename))

