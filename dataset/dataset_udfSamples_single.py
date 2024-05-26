"""
单物体数据集，一次仅加载一个残缺点云及对应的查询点数据
"""
import logging
import os
import re

import numpy as np
import open3d as o3d
import torch
import torch.utils.data

import workspace as ws


def get_instance_filenames(data_source, split, objIdx):
    scene_patten = ws.scene_patten
    npzfiles = []
    pcdfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                scene_name = re.match(scene_patten, instance_name).group()
                instance_filename = os.path.join(dataset, class_name, scene_name + ".npz")
                pcd_filename = os.path.join(dataset, class_name, instance_name + "_{}.ply".format(objIdx))
                if not os.path.isfile(os.path.join(data_source, ws.udf_samples_subdir, instance_filename)):
                    logging.warning("Requested non-existent file '{}'".format(instance_filename))
                npzfiles += [instance_filename]
                pcdfiles += [pcd_filename]

    return npzfiles, pcdfiles


def unpack_udf_samples(filename):
    data = np.load(filename)['data']
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def get_pcd_data(pcd_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    xyz_load = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
    return xyz_load


class UDFSamples(torch.utils.data.Dataset):
    def __init__(self, data_source, split, objIdx):
        self.data_source = data_source
        self.objIdx = objIdx
        self.npyfiles, self.pcdfiles = get_instance_filenames(data_source, split, objIdx)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        udf_filename = os.path.join(self.data_source, ws.udf_samples_subdir, self.npyfiles[idx])
        pcd_filename = os.path.join(self.data_source, ws.pcd_samples_subdir, self.pcdfiles[idx])

        pcd = get_pcd_data(pcd_filename)
        sdf_data = unpack_udf_samples(udf_filename)

        return pcd, sdf_data, idx
