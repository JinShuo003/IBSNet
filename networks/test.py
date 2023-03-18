import torch
from torch import nn
from deep_sdf import workspace
from models import *
import numpy as np

work_dir = "../data"
specs = workspace.load_experiment_specifications(work_dir)

sdf_data = np.array([[[1, 2, 3, 4, 5],
                     [3, 5, 1, 2, 4],
                     [5, 1, 0, 9, 4],
                     [9, 8, 1, 2, 3]],
                    [[7, 2, 10, 4, 5],
                     [9, 5, 1, 2, 4],
                     [6, 1, 0, 9, 4],
                     [-2, 8, 1, 2, 3]]])
sdf_data = torch.from_numpy(sdf_data)
sdf_data = sdf_data.reshape(-1, 5)
print(sdf_data.shape)

x_obj1 = np.array([[1, 2, 3, 4, 5, 6],
                   [10, 9, 8, 7, 6, 5]])
x_obj1 = torch.from_numpy(x_obj1)

latent_obj1 = x_obj1.repeat_interleave(4, dim=0)
print(latent_obj1.shape)


