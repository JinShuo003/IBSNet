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


def get_instance_filenames(mesh_path, filename_re):
    """
    将路径下所有满足正则式的文件名添加到集合中
    :param mesh_path: mesh文件路径
    :param filename_re: 文件名正则式
    :return 文件名列表（无重复元素）
    """
    # 获取每一个实例的文件名
    instance_filenames = OrderedSet()
    filename_list = os.listdir(mesh_path)
    for filename in filename_list:
        instance_filename = re.match(filename_re, filename).group()
        instance_filenames.append(instance_filename)
    return list(instance_filenames)


def get_category_name(instance_filenames, category_re):
    """
    根据所有实例名获取类别集合
    :param instance_filenames: 实例名列表
    :param category_re: 用于切分类别名的正则式
    :return: 类别集合
    """
    # 根据所有实例名获取所有类别
    category_set = OrderedSet()
    for filename in instance_filenames:
        category_name = re.match(category_re, filename).group()
        if category_name not in category_set:
            category_set.add(category_name)
    return category_set


def partition_dataset(instance_filenames, category_set, specs):
    """
    依次将每一种类别下的所有文件按配置文件中的比例划分为训练集、测试集、验证集
    :param instance_filenames: 实例名列表
    :param category_set: 类别集合
    :param specs: 配置信息
    :return: 三个SubSet的列表，每个SubSet对应一个类目，记录了原始数据和下标
    """
    train_dataset = []
    validate_dataset = []
    test_dataset = []
    # 每个类别分别划分，保证比例均匀
    for category in category_set:
        # 挑出属于当前类别的文件名
        temp_filenames = []
        for filename in instance_filenames:
            if re.match(category, filename):
                temp_filenames.append(filename)
        # 计算训练集、测试集、验证集大小
        train_size = int(len(temp_filenames) * specs["partition_option"]["train_dataset_proportion"])
        test_size = int(len(temp_filenames) * specs["partition_option"]["test_dataset_proportion"])
        validate_size = len(temp_filenames) - test_size - train_size
        # print("train: {}\nvalidate: {}\ntest: {}\n".format(train_size, validate_size, test_size))

        _train_dataset, _validate_dataset, _test_dataset = \
            torch.utils.data.random_split(temp_filenames, [train_size, validate_size, test_size])

        train_dataset.append(_train_dataset)
        validate_dataset.append(_validate_dataset)
        test_dataset.append(_test_dataset)

    return train_dataset, validate_dataset, test_dataset


def generate_split_file(dataset_path, filename, dataset, category_set, specs):
    """
    根据数据集划分结果生成对应的文件存储划分信息
    :param dataset_path: 文件保存路径
    :param filename: 文件名
    :param dataset: SubSet的列表
    :param category_set: 类别集合
    :param specs: 配置信息
    """
    # 生成数据
    split_data = {"IBSNet": {}}
    for category in category_set:
        split_data["IBSNet"][category] = []
    for item in dataset:
        for index in item.indices:
            category = re.match(specs["category_re"], item.dataset[index]).group()
            split_data["IBSNet"][category].append(item.dataset[index])
    split_data = json.dumps(split_data, indent=1)

    # 生成文件
    split_path = os.path.join(dataset_path, filename)
    if os.path.isfile(split_path):
        os.remove(split_path)

    with open(split_path, 'w', newline='\n') as f:
        f.write(split_data)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'config/generateDataset.json'
    specs = parse_config(configFile_path)

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["dataset_path"]):
        os.mkdir(specs["dataset_path"])

    # 获取所有实例名
    instance_filenames = get_instance_filenames(specs["mesh_path"], specs["filename_re"])
    # 获取类别集合
    category_set = get_category_name(instance_filenames, specs["category_re"])
    # 划分训练集、测试集、验证集
    train_dataset, validate_dataset, test_dataset = partition_dataset(instance_filenames, category_set, specs)
    # 将划分结果写成文件
    generate_split_file(specs["dataset_path"], "train.json", train_dataset, category_set, specs)
    generate_split_file(specs["dataset_path"], "validate.json", validate_dataset, category_set, specs)
    generate_split_file(specs["dataset_path"], "test.json", test_dataset, category_set, specs)
    # 将data/sdf中的文件按照场景分开
