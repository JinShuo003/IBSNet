# blender packages
import logging
# built-in modules
import sys
from itertools import product
from pathlib import Path

import bpy
# third-party packages
import numpy as np
from bpy.types import (
    Scene, Material, Object
)


def init_scene():
    """
    Initialize a scene with the basic rendering configurations.
    """
    # the bpy.context module is usually read-only, so we access the current scene through bpy.data
    scene_name: str = bpy.context.scene.name
    scene: Scene = bpy.data.scenes[scene_name]
    scene.render.engine = 'BLENDER_EEVEE'
    # scene.render.engine = 'CYCLES'
    # output image settings
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = True  # transparent background
    # remove the default cube and lights created by blender
    for obj in bpy.data.objects:
        if obj.name != 'Camera':
            logging.info(f'remove object {obj.name} from the scene')
            bpy.data.objects.remove(obj)


def create_materials():
    """
    Create materials for rendering the input point cloud / output mesh
    """
    params = [
        {'name': 'pointcloud1', 'color': (0.165, 0.564, 0.921, 1.0), 'transparent': False},
        {'name': 'pointcloud2', 'color': (1, 0.8, 0.1, 1.0), 'transparent': False},
        {'name': 'ibs', 'color': (0.5, 0.25, 0.8, 1.0), 'transparent': False}
    ]
    global protected_material_names
    protected_material_names = [param['name'] for param in params]
    roughness = 0.5
    for param in params:
        # create a new material and enable nodes
        bpy.data.materials.new(name=param['name'])
        material: Material = bpy.data.materials[param['name']]
        material.use_nodes = True

        nodes: bpy_prop_collection = material.node_tree.nodes
        links: bpy_prop_collection = material.node_tree.links
        # remove the default Principle BSDF node in the material's node tree
        for node in nodes:
            if node.type != 'OUTPUT_MATERIAL':
                nodes.remove(node)
        # add a Diffuse BSDF node
        BSDF_node = nodes.new('ShaderNodeBsdfDiffuse')
        BSDF_node.inputs['Color'].default_value = param['color']
        BSDF_node.inputs['Roughness'].default_value = roughness
        output_node: ShaderNodeOutputMaterial = nodes['Material Output']
        if param['transparent']:
            # for a transparent material, create a Mix Shader node and enable color
            # blending
            transparent_node = nodes.new('ShaderNodeBsdfTransparent')
            mix_node = nodes.new('ShaderNodeMixShader')
            mix_node.inputs['Fac'].default_value = 0.5

            # here we have to use index instead of key to access the 'Shader' input
            # of a Mix Shader node, because there are two input slots with the same
            # name 'Shader' and we need to use both of them
            links.new(BSDF_node.outputs['BSDF'], mix_node.inputs[1])
            links.new(transparent_node.outputs['BSDF'], mix_node.inputs[2])
            links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

            material.blend_method = 'BLEND'
            material.shadow_method = 'CLIP'
        else:
            # for a non-transparent material, link the Diffuse BSDF node's output
            # with the output node's input
            links.new(BSDF_node.outputs['BSDF'], output_node.inputs['Surface'])

        logging.info('Diffuse BSDF material {} has been created'.format(param['name']))


def init_camera():
    """
    Set the camera's position
    """
    camera_obj: Object = bpy.data.objects['Camera']
    # the location is obtained through GUI
    camera_obj.location = (0.7359, -0.6926, 0.4958)


def init_lights(scale_factor: float = 1):
    """
    Set lights for rendering.
    By default, this function will place
      - two sun lights above the object
      - one point light below the object
    The object is assumed to be normalized, i.e. it can be enclosed by a unit cube
    centered at (0, 0, 0).
    To render larger objects, pass the `scale_factor` parameter explicitly to scale
    the locations of lights.
    """
    # all parameters are obtained through blender GUI
    # the unit of angle is radians as blender API default setting
    params = [
        {
            'name': 'sun light 1', 'type': 'SUN',
            'location': np.array([3.638, 1.674, 4.329]), 'energy': 5.0, 'angle': 0.199
        },
        {
            'name': 'sun light 2', 'type': 'SUN',
            'location': np.array([0.449, -3.534, 1.797]), 'energy': 1.83, 'angle': 0.009
        },
        {
            'name': 'point light 1', 'type': 'POINT',
            'location': np.array([-2.163, -0.381, -2.685]), 'energy': 500
        }
    ]
    for param in params:
        light = bpy.data.lights.new(name=param['name'], type=param['type'])
        light.energy = param['energy']
        if param['type'] == 'SUN':
            light.angle = param['angle']
        light_obj = bpy.data.objects.new(name=param['name'], object_data=light)
        light_obj.location = param['location'] * scale_factor
        bpy.context.collection.objects.link(light_obj)


def creat_pointcloud_modifier(modifier_name: str, material_name: str, sphere_radius: int = 0.005):
    """
    Create the geometry nodes as a modifier for point clouds.
    This modifier will expand each point to a ico sphere for rendering.
    """
    # create a node group and enable it as a geometry modifier
    geom_nodes = bpy.data.node_groups.new('{}'.format(modifier_name), 'GeometryNodeTree')
    geom_nodes.is_modifier = True
    nodes = geom_nodes.nodes
    links = geom_nodes.links
    interface = geom_nodes.interface
    interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # create all node with their properties set
    input_node = nodes.new('NodeGroupInput')
    output_node = nodes.new('NodeGroupOutput')
    mesh_to_points_node = nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points_node.mode = 'VERTICES'
    ico_sphere_node = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere_node.inputs['Radius'].default_value = sphere_radius
    ico_sphere_node.inputs['Subdivisions'].default_value = 3  # control the smoothness of the ico sphere
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    material_node = nodes.new('GeometryNodeReplaceMaterial')
    # only set the New slot of the Replace Material node because we actually
    # use it to set the material of output instances (spheres), the Old slot
    # is not used.
    material_node.inputs['New'].default_value = bpy.data.materials['{}'.format(material_name)]

    # link the nodes
    links.new(input_node.outputs['Geometry'], mesh_to_points_node.inputs['Mesh'])
    # the PLY file are imported as mesh, so we need to replace each vertex in
    # the mesh with a point, then we will have a real point cloud in Blender
    links.new(mesh_to_points_node.outputs['Points'], instance_node.inputs['Points'])
    # use the pre-defined ico sphere as the template instance. with the
    # Instance On Points node we can instantiate an instance at each point in
    # the point cloud
    links.new(ico_sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], material_node.inputs['Geometry'])
    links.new(material_node.outputs['Geometry'], output_node.inputs['Geometry'])


def track_object(obj: Object):
    """
    Let the camera track the specified object's center.
    By setting the tracking constraint, we can easily make the camera orient to
    the target object we want to render. This is less flexible but easier than
    setting the rotation manually.
    """
    camera: Object = bpy.data.objects['Camera']
    # the Track To constraint can keep the up direction of the camera better
    # than the Damp Track constraint, allowing placing the camera in the half-
    # space where x < 0
    camera.constraints.new('TRACK_TO')
    constraint = camera.constraints['Track To']
    constraint.target = obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'


def clear_imported_objects():
    """
    Remove the imported mesh and point cloud from the current scene, together
    with the materials automatically created by Blender when importing a mesh.
    """
    for obj in bpy.data.objects:
        # after application of geometry nodes, the point cloud data will also
        # be mesh
        if obj.type == 'MESH':
            logging.info(f'remove object {obj.name} from the current scene')
            bpy.data.objects.remove(obj)

    for material in bpy.data.materials:
        if material.name not in protected_material_names:
            logging.info(f'remove material {material.name}')
            bpy.data.materials.remove(material)


def launch_render(pcd_path: str, ibs_path: str, output_path: str, filename: str, render_options: dict, additional_info: str):
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    init_scene()
    create_materials()
    init_lights()
    creat_pointcloud_modifier('pointcloud1 modifier', 'pointcloud1', 0.007)
    creat_pointcloud_modifier('pointcloud2 modifier', 'pointcloud2', 0.007)
    creat_pointcloud_modifier('ibs modifier', 'ibs', 0.007)

    camera_obj: Object = bpy.data.objects['Camera']
    radius = 1.5
    camera_location = np.array([-0.1, 0.7, 0.3])
    camera_location /= np.linalg.norm(camera_location)
    camera_location *= radius
    camera_obj.location = camera_location

    pcd_path = Path(pcd_path)
    ibs_path = Path(ibs_path)
    output_path = Path(output_path)
    ply1_basename = "{}_0".format(filename)
    ply2_basename = "{}_1".format(filename)
    ibs_basename = "{}".format(filename)
    ply1_filename = "{}.ply".format(ply1_basename)
    ply2_filename = "{}.ply".format(ply2_basename)
    ibs_filename = "{}.ply".format(ibs_basename)
    ply1_filepath = pcd_path / ply1_filename
    ply2_filepath = pcd_path / ply2_filename
    ibs_filepath = ibs_path / ibs_filename

    if render_options.get('pcd1'):
        bpy.ops.wm.ply_import(filepath=ply1_filepath.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
        pointcloud1 = bpy.data.objects[ply1_basename]
        modifier = pointcloud1.modifiers.new('modifier', 'NODES')
        modifier.node_group = bpy.data.node_groups['pointcloud1 modifier']

    if render_options.get('pcd2'):
        bpy.ops.wm.ply_import(filepath=ply2_filepath.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
        pointcloud2 = bpy.data.objects[ply2_basename]
        modifier = pointcloud2.modifiers.new('modifier', 'NODES')
        modifier.node_group = bpy.data.node_groups['pointcloud2 modifier']

    if render_options.get('ibs'):
        bpy.ops.wm.ply_import(filepath=ibs_filepath.as_posix(), forward_axis='NEGATIVE_Z', up_axis='Y')
        ibs = bpy.data.objects[ibs_basename]
        modifier = ibs.modifiers.new('modifier', 'NODES')
        modifier.node_group = bpy.data.node_groups['ibs modifier']

    for view_index, sign in enumerate(product(np.array([1, -1]), repeat=3)):
        camera_obj.location = camera_location * sign
        track_object(pointcloud1)
        bpy.context.scene.render.filepath = (output_path / f'{view_index}_{additional_info}.png').as_posix()
        bpy.ops.render.render(write_still=True)
        break


if __name__ == '__main__':
    import re

    filename_list = [
        # "scene1.1016_view7",
        # "scene2.1033_view4"
        # "scene3.1007_view1"
        # "scene4.1015_view10",
        # "scene4.1015_view14",
        # "scene5.1028_view2",
        # "scene6.1001_view0"
        "scene6.1019_view9"
    ]

    for filename in filename_list:
        category = re.match("scene\\d", filename).group()
        scene_name = re.match("scene\\d.\\d{4}", filename).group()

        render_options = {'pcd1': True, 'pcd2': True, 'ibs': True}
        pcd_path = "D:\\dataset\\IBSNet\\trainData\\pcdComplete\\INTE\\{}".format(category)
        ibs_path = "D:\\dataset\\IBSNet\\evaluateData\\IBS_pcd_complete\\{}".format(category)
        output_path = "D:\\dataset\\IBSNet\\render\\{}\\{}".format(scene_name, filename)
        # gt complete
        launch_render(pcd_path, ibs_path, output_path, scene_name, render_options, "gt")

        render_options = {'pcd1': True, 'pcd2': True, 'ibs': True}
        pcd_path = "D:\\dataset\\IBSNet\\trainData\\pcdScan\\diffUDF\\{}".format(category)
        ibs_path = "D:\\dataset\\IBSNet\\evaluateData\\IBS_pcd_scan\\{}".format(category)
        output_path = "D:\\dataset\\IBSNet\\render\\{}\\{}".format(scene_name, filename)
        # input geometric
        launch_render(pcd_path, ibs_path, output_path, filename, render_options, "input")

        # render_options = {'pcd1': True, 'pcd2': True, 'ibs': True}
        # pcd_path = "D:\\dataset\\IBSNet\\trainData\\pcdScan\\diffUDF\\{}".format(category)
        # ibs_path = "D:\\dataset\\IBSNet\\test_result\\Grasping_field\\{}".format(category)
        # output_path = "D:\\dataset\\IBSNet\\render\\{}\\{}".format(scene_name, filename)
        # launch_render(pcd_path, ibs_path, output_path, filename, render_options, "Grasping_field")

        render_options = {'pcd1': True, 'pcd2': True, 'ibs': True}
        pcd_path = "D:\\dataset\\IBSNet\\trainData\\pcdScan\\diffUDF\\{}".format(category)
        ibs_path = "D:\\dataset\\IBSNet\\test_result\\IBSNet_transformer_IM_lr5e4_l2\\{}".format(category)
        output_path = "D:\\dataset\\IBSNet\\render\\{}\\{}".format(scene_name, filename)
        launch_render(pcd_path, ibs_path, output_path, filename, render_options, "IBSNet")
