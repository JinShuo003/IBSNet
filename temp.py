import numpy as np

points = []

points_np = np.array(points, dtype=np.float32)
print(points_np.shape)

points_np = points_np.reshape(-1, 3)
print(points_np.shape)