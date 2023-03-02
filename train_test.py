import torch.utils.data as data_utils
import deep_sdf
import deep_sdf.workspace as ws
import json


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


if __name__ == '__main__':
    experiment_directory = './data'
    specs = ws.load_experiment_specifications(experiment_directory)
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )
    # 219，训练集共有219个数字
    print("length of sdf_dataset: ", sdf_dataset.__len__())

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    # 54，将训练集划分为了4个Batch
    print("length of sdf_loader: ", sdf_loader.__len__())

    for pcd1, pcd2, sdf_data, indices in sdf_loader:
        sdf_data = sdf_data.reshape(scene_per_batch, -1, 5)
        # num_sdf_samples = sdf_data.shape[0]

        pcd1.requires_grad = False
        pcd2.requires_grad = False
        sdf_data.requires_grad = False

        xyz = sdf_data[:, :, 0:3]

        sdf_gt1 = sdf_data[:, :, 3].unsqueeze(2)
        sdf_gt2 = sdf_data[:, :, 4].unsqueeze(2)

        print(indices)

        break
