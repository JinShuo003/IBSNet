"""
根据./config/generateDataset.json中的设置，划分测试集、训练集，并保存划分好的结果
"""
import json
import os
import re
from ordered_set import OrderedSet
import torch
from torch.utils.data import DataLoader, Dataset


def parse_config(config_filepath: str = './config/generateSDF.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def get_instance_filenames(specs):
    """
    将sdf采样点路径下每一个类别文件夹中的文件名按类别存入instance_filename_list
    """
    sdf_sample_path = specs["sdf_sample_path"]
    filename_re = specs["filename_re"]
    categories = specs["categories"]
    # 获取每一个实例的文件名
    instance_filename_list = list()
    for category in categories:
        cur_list = list()
        filename_list = os.listdir(os.path.join(sdf_sample_path, category))
        for filename in filename_list:
            instance_filename = re.match(filename_re, filename).group()
            cur_list.append(instance_filename)
        instance_filename_list.append(cur_list)
    return instance_filename_list


def partition_dataset(instance_filename_list, specs):
    """
    依次将每一种类别下的所有文件按配置文件中的比例划分为训练集、测试集、验证集
    :param instance_filenames: 实例名列表
    :param category_set: 类别集合
    :param specs: 配置信息
    :return: 三个SubSet的列表，每个SubSet对应一个类目，记录了原始数据和下标
    """
    train_dataset = []
    test_dataset = []
    # 每个类别分别划分，保证比例均匀
    for i in range(instance_filename_list.__len__()):
        # 计算训练集、测试集、验证集大小
        train_size = int(len(instance_filename_list[i]) * specs["partition_option"]["train_dataset_proportion"])
        test_size = len(instance_filename_list[i]) - train_size
        print("train: {}\ntest: {}\n".format(train_size, test_size))

        _train_dataset, _test_dataset = torch.utils.data.random_split(instance_filename_list[i], [train_size, test_size])

        train_dataset.append(_train_dataset)
        test_dataset.append(_test_dataset)

    return train_dataset, test_dataset


def generate_split_file(dataset_path, dir_name, filename, dataset, category_set, specs):
    """
    根据数据集划分结果生成对应的文件存储划分信息
    """
    dataset_name = specs["dataset_name"]
    # 若目录不存在则创建目录
    if not os.path.isdir(os.path.join(dataset_path, dir_name)):
        os.mkdir(os.path.join(dataset_path, dir_name))
    # 生成数据
    split_data = {dataset_name: {}}
    for category in category_set:
        split_data[dataset_name][category] = []
    for i in range(dataset.__len__()):
        for index in dataset[i].indices:
            split_data[dataset_name]["scene{}".format(i+1)].append(dataset[i].dataset[index])
    split_json = json.dumps(split_data, indent=1)

    # 生成文件
    split_path = os.path.join(dataset_path, dir_name, "{}.json".format(filename))
    if os.path.isfile(split_path):
        os.remove(split_path)

    # 写入总体的split文件
    with open(split_path, 'w', newline='\n') as f:
        f.write(split_json)

    # 写入每个场景的split文件
    for key in split_data[dataset_name]:
        scene_data = {dataset_name: {}}
        scene_data[dataset_name][key] = split_data[dataset_name][key]
        scene_json = json.dumps(scene_data, indent=1)
        # 生成文件
        scene_path = os.path.join(dataset_path, dir_name, "{}_{}.json".format(filename, key))
        if os.path.isfile(scene_path):
            os.remove(scene_path)
        # 写入总体的split文件
        with open(scene_path, 'w', newline='\n') as f:
            f.write(scene_json)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generateDataset.json'
    specs = parse_config(configFile_path)

    dataset_path = specs["dataset_path"]
    train_split_dirname = specs["train_split_dirname"]
    test_split_dirname = specs["test_split_dirname"]
    categories = specs["categories"]

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["dataset_path"]):
        os.mkdir(specs["dataset_path"])

    # 获取所有实例名
    instance_filename_list = get_instance_filenames(specs)
    # 划分训练集、测试集
    train_dataset, test_dataset = partition_dataset(instance_filename_list, specs)
    # 将划分结果写成文件
    generate_split_file(dataset_path, train_split_dirname, "train", train_dataset, categories, specs)
    generate_split_file(dataset_path, test_split_dirname, "test", test_dataset, categories, specs)
