import open3d as o3d
from matplotlib import pyplot as plt

pcd = o3d.io.read_point_cloud('airplane.pcd')
o3d.visualization.draw_geometries([pcd])
