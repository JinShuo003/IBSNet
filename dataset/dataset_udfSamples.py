"""
数据集，能够同时加载一对残缺点云及对应的查询点数据
"""
import logging
import os
import re

import numpy as np
import open3d as o3d
import torch
import torch.utils.data

import workspace as ws


def get_instance_filenames(data_source, split):
    scene_patten = ws.scene_patten
    npzfiles = []
    pcd1files = []
    pcd2files = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                scene_name = re.match(scene_patten, instance_name).group()
                instance_filename = os.path.join(dataset, class_name, scene_name + ".npz")
                pcd1_filename = os.path.join(dataset, class_name, instance_name + "_0.ply")
                pcd2_filename = os.path.join(dataset, class_name, instance_name + "_1.ply")
                if not os.path.isfile(os.path.join(data_source, ws.udf_samples_subdir, instance_filename)):
                    logging.warning("Requested non-existent file '{}'".format(instance_filename))
                npzfiles += [instance_filename]
                pcd1files += [pcd1_filename]
                pcd2files += [pcd2_filename]

    return npzfiles, pcd1files, pcd2files


def unpack_udf_samples(filename):
    data = np.load(filename)['data']
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def get_pcd_data(pcd_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    xyz_load = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
    return xyz_load


class UDFSamples(torch.utils.data.Dataset):
    def __init__(self, data_source, split):
        self.data_source = data_source
        self.npyfiles, self.pcd1files, self.pcd2files = get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        udf_filename = os.path.join(self.data_source, ws.udf_samples_subdir, self.npyfiles[idx])
        pcd1_filename = os.path.join(self.data_source, ws.pcd_samples_subdir, self.pcd1files[idx])
        pcd2_filename = os.path.join(self.data_source, ws.pcd_samples_subdir, self.pcd2files[idx])

        pcd1 = get_pcd_data(pcd1_filename)
        pcd2 = get_pcd_data(pcd2_filename)
        sdf_data = unpack_udf_samples(udf_filename)

        return pcd1, pcd2, sdf_data, idx
