import open3d as o3d
from matplotlib import pyplot as plt

sphere = o3d.geometry.TriangleMesh.create_sphere(4.0)
sphere.compute_vertex_normals()
cylinder = o3d.geometry.TriangleMesh.create_cylinder(1.0, 4.0, 30, 4)
cylinder.compute_triangle_normals()
cylinder.translate([6, 2, 0.0])

render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = 'defaultLit'

render.scene.add_geometry("sphere1", sphere, mat)
render.scene.add_geometry("cylinder1", cylinder, mat)
render.setup_camera(45, [0, 0, 0], [0, 0, -25.0], [0, 1, 0])

cimg = render.render_to_image()
dimg = render.render_to_depth_image()
plt.imsave("test_cimg", cimg)
plt.imsave("test_dimg", dimg)
#
# plt.subplot(1, 2, 1)
# plt.imshow(cimg)
# plt.subplot(1, 2, 2)
# plt.imshow(dimg)
# plt.show()