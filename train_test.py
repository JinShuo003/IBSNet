import torch.utils.data as data_utils
import deep_sdf
import deep_sdf.workspace as ws
import json
from torch.utils.tensorboard import SummaryWriter

from networks.models import *


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):
    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def main_function():

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    experiment_directory = './data'
    specs = ws.load_experiment_specifications(experiment_directory)
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    latent_size = specs["CodeLength"]
    epoch_num = specs["NumEpochs"]
    writer = SummaryWriter('./logs')

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

    encoder_obj1 = ResnetPointnet()
    encoder_obj2 = ResnetPointnet()
    decoder = Decoder(latent_size, **specs["NetworkSpecs"])
    IBS_Net = IBSNet(encoder_obj1, encoder_obj2, decoder, num_samp_per_scene).cuda()

    input_pcd1_shape = torch.randn(1, 1024, 3)
    input_pcd2_shape = torch.randn(1, 1024, 3)
    input_xyz_shape = torch.randn(1, 30000, 3)

    print('here')
    if torch.cuda.is_available():
        IBS_Net = IBS_Net.cuda()
        input_pcd1_shape = input_pcd1_shape.cuda()
        input_pcd2_shape = input_pcd2_shape.cuda()
        input_xyz_shape = input_xyz_shape.cuda()
        print('here')

    writer.add_graph(IBS_Net, (input_pcd1_shape, input_pcd2_shape, input_xyz_shape))

    lr_schedules = get_learning_rate_schedules(specs)

    optimizer = torch.optim.Adam(
        [
            {
                "params": IBS_Net.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            }
        ]
    )

    loss_udf1 = torch.nn.L1Loss(reduction="sum")
    loss_udf2 = torch.nn.L1Loss(reduction="sum")

    for epoch in range(epoch_num):
        print('-------------epoch: {}---------------'.format(epoch))
        IBS_Net.train()

        adjust_learning_rate(lr_schedules, optimizer, epoch)
        print('-------------learning rate: {}---------------'.format(lr_schedules[0].get_learning_rate(epoch)))

        epoch_total_loss = 0
        for pcd1, pcd2, sdf_data, indices in sdf_loader:
            sdf_data = sdf_data.reshape(scene_per_batch, -1, 5)

            pcd1.requires_grad = False
            pcd2.requires_grad = False
            sdf_data.requires_grad = False

            xyz = sdf_data[:, :, 0:3]

            pcd1 = pcd1.cuda()
            pcd2 = pcd1.cuda()
            xyz = xyz.cuda()

            udf_gt1 = sdf_data[:, :, 3].unsqueeze(2)
            udf_gt2 = sdf_data[:, :, 4].unsqueeze(2)

            out = IBS_Net(pcd1, pcd2, xyz)
            udf_pred1 = out[:, :, 0].unsqueeze(2)
            udf_pred2 = out[:, :, 1].unsqueeze(2)

            loss_1 = loss_udf1(udf_pred1, udf_gt1.cuda()) / num_samp_per_scene
            loss_2 = loss_udf2(udf_pred2, udf_gt2.cuda()) / num_samp_per_scene

            loss = loss_1 + loss_2
            epoch_total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("loss", epoch_total_loss, epoch)
        print('-------------epoch_total_loss: {}---------------'.format(epoch_total_loss))

    writer.close()


if __name__ == '__main__':
    main_function()
