import numpy as np
import trimesh
import open3d as o3d


def o3d2trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return tri_mesh


def trimesh2o3d(tri_mesh):
    vertices = tri_mesh.vertices
    triangles = tri_mesh.faces

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return o3d_mesh