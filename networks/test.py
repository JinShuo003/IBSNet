import torch
from torch import nn
from deep_sdf import workspace
from models import *
import numpy as np

work_dir = "../data"
specs = workspace.load_experiment_specifications(work_dir)


obj1 = torch.Tensor(10, 1024, 3)
print(obj1.shape)

# obj2 = torch.Tensor(10, 500, 3)
# xyz = torch.Tensor(10, specs["SamplesPerScene"], 3)

encoder_obj1 = ResnetPointnet()
out = encoder_obj1(obj1)
print(out.shape)

# encoder_obj2 = ResnetPointnet()
#
# decoder = Decoder(specs["CodeLength"], **specs["NetworkSpecs"])
# ibs_net = IBSNet(encoder_obj1, encoder_obj2, decoder, specs["SamplesPerScene"])
#
# out = ibs_net(obj1, obj2, xyz)
# print(out.shape)


