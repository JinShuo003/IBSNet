import math
import os.path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd
import pyperclip

from utils import path_utils


class App:
    def __init__(self, specs):
        self.specs = specs
        self.filename_tree = None
        self.obj1_path = None
        self.obj2_path = None
        self.mesh1 = None
        self.mesh2 = None

        self.key_mesh_1 = "mesh_1"
        self.key_mesh_2 = "mesh_gt_2"
        self.key_pcd_partial_1 = "pcd_partial_1"
        self.key_pcd_partial_2 = "pcd_partial_2"
        self.key_ibs_gt = "ibs_gt"
        self.key_ibs_geometric = "ibs_geometric"
        self.key_ibs_grasping_field = "ibs_grasping_field"
        self.key_ibs_IBSNet = "ibs_IBSNet"

        self.TAG_obj1 = "pcd1"
        self.TAG_obj2 = "pcd2"
        self.TAG_IBS = "IBS"

        self.obj_material = rendering.MaterialRecord()
        self.obj_material.shader = 'defaultLit'
        self.material = {
            "obj": self.obj_material
        }

        gui.Application.instance.initialize()

        self.scene_gt_str = "complete"
        self.scene_geometric_str = "partial"
        self.scene_grasping_field_str = specs.get("sub_window_3_name")
        self.scene_IBSNet_str = specs.get("sub_window_4_name")

        self.sub_window_width = specs.get("window_size_options").get("sub_window_width")
        self.sub_window_height = specs.get("window_size_options").get("sub_window_height")
        self.tool_bar_width = specs.get("window_size_options").get("tool_bar_width")
        self.window_width = self.sub_window_width * 2 + self.tool_bar_width
        self.window_height = self.sub_window_height * 2
        self.window = gui.Application.instance.create_window("visualization", self.window_width, self.window_height)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_key(self.on_key)

        self.em = self.window.theme.font_size

        # ground truth窗口
        self.scene_gt = gui.SceneWidget()
        self.scene_gt.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_gt_text = gui.Label("gt")

        # 传统方法窗口
        self.scene_geometric = gui.SceneWidget()
        self.scene_geometric.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_geometric_text = gui.Label("geometric")

        # grasping field窗口
        self.scene_grasping_field = gui.SceneWidget()
        self.scene_grasping_field.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_grasping_field_text = gui.Label(self.specs.get("sub_window_3_name"))

        # 补全结果2窗口
        self.scene_IBSNet = gui.SceneWidget()
        self.scene_IBSNet.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_IBSNet_text = gui.Label(self.specs.get("sub_window_4_name"))

        # 视点
        self.view_point_radius = 0.75
        self.default_view_point = np.array([0, 0, self.view_point_radius])
        self.default_theta = 90
        self.default_phi = 90
        self.current_view_point = np.array([0, 0, self.view_point_radius])
        self.current_theta = 90
        self.current_phi = 90

        # 窗口是否显示的标记
        self.flag_grasping_field_window_show = False
        self.flag_IBSNet_window_show = True

        # 几何体
        self.mesh_gt_1 = None
        self.mesh_gt_2 = None
        self.ibs_gt = None
        self.pcd_partial_1 = None
        self.pcd_partial_2 = None
        self.ibs_geometric = None
        self.ibs_grasping_field = None
        self.ibs_IBSNet = None

        # 工具栏
        self.tool_bar_layout = gui.Vert()

        # 文件路径输入区域
        self.data_dir_editor_layout = None
        self.ibs_grasping_field_dir_editor_text = None
        self.ibs_grasping_field_dir_editor = None
        self.ibs_IBSNet_dir_editor_text = None
        self.ibs_IBSNet_dir_editor = None
        self.data_dir_confirm_btn = None

        # 数据选择区域
        self.geometry_select_layout = None
        self.selected_category = -1
        self.category_selector_layout = None
        self.category_selector_text = None
        self.category_selector = None
        self.selected_scene = -1
        self.scene_selector_layout = None
        self.scene_selector_text = None
        self.scene_selector = None
        self.selected_view = None
        self.view_selector_layout = None
        self.view_selector_text = None
        self.view_selector = None
        self.btn_load = None

        # 数据切换区域
        self.data_switch_layout = None
        self.category_switch_area = None
        self.category_switch_text = None
        self.category_switch_btn_area = None
        self.btn_pre_category = None
        self.btn_next_category = None
        self.scene_switch_area = None
        self.scene_switch_text = None
        self.scene_switch_btn_area = None
        self.btn_pre_scene = None
        self.btn_next_scene = None
        self.view_switch_area = None
        self.view_switch_text = None
        self.view_switch_btn_area = None
        self.btn_pre_view = None
        self.btn_next_view = None

        # 可见性
        self.show_obj1_checked = False
        self.show_obj2_checked = False
        self.show_ibs_checked = False
        self.visible_control_layout = None
        self.visible_text = None
        self.visible_control_checkbox_layout = None
        self.show_obj1_checkbox_text = None
        self.show_obj2_checkbox_text = None
        self.show_ibs_checkbox_text = None
        self.show_obj1_checkbox = None
        self.show_obj2_checkbox = None
        self.show_ibs_checkbox = None

        # 数据信息
        self.data_info_layout = None
        self.category_info = None
        self.scene_info = None
        self.view_info = None
        self.btn_copy_cur_data_info = None
        self.category_info_patten = "category: {}"
        self.scene_info_patten = "scene: {}"
        self.view_info_patten = "view: {}"
        self.value_patten = "{:.2f}"

        self.init_data_dir_editor_area()
        self.init_geometry_select_area()
        self.init_data_switch_area()
        self.init_visible_control_area()
        self.init_data_info_area()

        self.tool_bar_layout.add_child(self.data_dir_editor_layout)
        self.tool_bar_layout.add_child(self.geometry_select_layout)
        self.tool_bar_layout.add_child(self.data_switch_layout)
        self.tool_bar_layout.add_child(self.visible_control_layout)
        self.tool_bar_layout.add_child(self.data_info_layout)

        self.window.add_child(self.scene_gt)
        self.window.add_child(self.scene_geometric)
        self.window.add_child(self.scene_grasping_field)
        self.window.add_child(self.scene_IBSNet)
        self.window.add_child(self.tool_bar_layout)
        self.window.add_child(self.scene_gt_text)
        self.window.add_child(self.scene_geometric_text)
        self.window.add_child(self.scene_grasping_field_text)
        self.window.add_child(self.scene_IBSNet_text)

    def on_data_dir_comfirm_btn_clicked(self):
        # 根据用户填写的文件路径构建目录树
        filename_tree_dir = self.specs.get("path_options").get("filename_tree_dir")
        dir = self.specs.get("path_options").get("geometries_dir").get(filename_tree_dir)
        if not os.path.exists(dir):
            self.show_message_dialog("warning", "The directory not exist")
            return
        self.filename_tree = path_utils.get_filename_tree(self.specs, dir)

        # 类别
        self.category_selector.clear_items()
        for category in self.filename_tree.keys():
            self.category_selector.add_item(category)
        if self.category_selector.number_of_items > 0:
            self.selected_category = self.category_selector.selected_index
            print("seleted category: ", self.selected_category)
        else:
            self.selected_category = -1
            print("seleted category: ", self.selected_category)
            self.show_message_dialog("warning", "No category in this directory")
            return

        # 场景
        self.scene_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        for key in self.filename_tree.get(selected_category).keys():
            self.scene_selector.add_item(key)
        if self.scene_selector.number_of_items > 0:
            self.selected_scene = self.scene_selector.selected_index
            print("seleted scene: ", self.selected_scene)
        else:
            self.selected_scene = -1
            print("seleted scene: ", self.selected_scene)
            self.show_message_dialog("warning", "No scene in this directory")
            return

        # 视角
        self.view_selector.clear_items()
        selected_scene = self.scene_selector.get_item(self.selected_scene)
        for key in self.filename_tree.get(selected_category).get(selected_scene):
            self.view_selector.add_item(key)
        if self.view_selector.number_of_items > 0:
            self.selected_view = self.view_selector.selected_index
            print("seleted view: ", self.selected_view)
        else:
            self.selected_view = -1
            print("seleted view: ", self.selected_view)
            self.show_message_dialog("warning", "No view in this directory")
            return

    def on_category_selection_changed(self, val, idx):
        if self.selected_category == idx:
            return
        self.selected_category = idx
        print("seleted category: ", self.selected_category)
        self.scene_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        for key in self.filename_tree.get(selected_category).keys():
            self.scene_selector.add_item(key)

        self.selected_scene = 0
        print("seleted scene: ", self.selected_scene)
        self.view_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        selected_scene = self.scene_selector.get_item(self.selected_scene)
        for key in self.filename_tree.get(selected_category).get(selected_scene):
            self.view_selector.add_item(key)
        self.selected_view = 0
        print("seleted view: ", self.selected_view)

    def on_scene_selection_changed(self, val, idx):
        if self.selected_scene == idx:
            return
        self.selected_scene = idx
        print("seleted scene: ", self.selected_scene)
        self.view_selector.clear_items()
        selected_category = self.category_selector.get_item(self.selected_category)
        selected_scene = self.scene_selector.get_item(self.selected_scene)
        for key in self.filename_tree.get(selected_category).get(selected_scene):
            self.view_selector.add_item(key)
        self.selected_view = 0
        print("seleted view: ", self.selected_view)

    def on_view_selection_changed(self, val, idx):
        if self.selected_view == idx:
            return
        self.selected_view = idx
        print("seleted view: ", self.selected_view)

    def clear_all_window(self):
        if self.scene_gt:
            self.scene_gt.scene.clear_geometry()
        if self.scene_geometric:
            self.scene_geometric.scene.clear_geometry()
        if self.scene_grasping_field:
            self.scene_grasping_field.scene.clear_geometry()
        if self.scene_IBSNet:
            self.scene_IBSNet.scene.clear_geometry()

    def paint_color(self, geometry, color):
        if geometry is None:
            return
        geometry.paint_uniform_color(color)

    def load_data(self):
        if self.category_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a category first")
            return
        if self.scene_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a scene first")
            return
        if self.view_selector.selected_index < 0:
            self.show_message_dialog("warning", "Please select a view first")
            return

        geometry_path_dict = self.get_geometry_path()
        self.mesh_gt_1 = self.read_mesh(geometry_path_dict.get(self.key_mesh_1))
        self.mesh_gt_2 = self.read_mesh(geometry_path_dict.get(self.key_mesh_2))
        self.pcd_partial_1 = self.read_pcd(geometry_path_dict.get(self.key_pcd_partial_1))
        self.pcd_partial_2 = self.read_pcd(geometry_path_dict.get(self.key_pcd_partial_2))
        self.ibs_gt = self.read_pcd(geometry_path_dict.get(self.key_ibs_gt))
        self.ibs_geometric = self.read_pcd(geometry_path_dict.get(self.key_ibs_geometric))
        self.ibs_grasping_field = self.read_pcd(geometry_path_dict.get(self.key_ibs_grasping_field))
        self.ibs_IBSNet = self.read_pcd(geometry_path_dict.get(self.key_ibs_IBSNet))

        self.paint_color(self.mesh_gt_1, (0.7, 0.2, 0.2))
        self.paint_color(self.mesh_gt_2, (0.2, 0.7, 0.2))
        self.paint_color(self.pcd_partial_1, (0.7, 0.2, 0.2))
        self.paint_color(self.pcd_partial_2, (0.2, 0.7, 0.2))
        self.paint_color(self.ibs_gt, (0.2, 0.2, 0.7))
        self.paint_color(self.ibs_geometric, (0.2, 0.2, 0.7))
        self.paint_color(self.ibs_grasping_field, (0.2, 0.2, 0.7))
        self.paint_color(self.ibs_IBSNet, (0.2, 0.2, 0.7))

        self.clear_all_window()
        if self.show_obj1_checked:
            self.add_object(self.scene_gt, self.TAG_obj1, self.mesh_gt_1)
            self.add_object(self.scene_geometric, self.TAG_obj1, self.pcd_partial_1)
            self.add_object(self.scene_grasping_field, self.TAG_obj1, self.pcd_partial_1)
            self.add_object(self.scene_IBSNet, self.TAG_obj1, self.pcd_partial_1)
        if self.show_obj2_checked:
            self.add_object(self.scene_gt, self.TAG_obj2, self.mesh_gt_2)
            self.add_object(self.scene_geometric, self.TAG_obj2, self.pcd_partial_2)
            self.add_object(self.scene_grasping_field, self.TAG_obj2, self.pcd_partial_2)
            self.add_object(self.scene_IBSNet, self.TAG_obj2, self.pcd_partial_2)
        if self.show_ibs_checked:
            self.add_object(self.scene_gt, self.TAG_IBS, self.ibs_gt)
            self.add_object(self.scene_geometric, self.TAG_IBS, self.ibs_geometric)
            self.add_object(self.scene_grasping_field, self.TAG_IBS, self.ibs_grasping_field)
            self.add_object(self.scene_IBSNet, self.TAG_IBS, self.ibs_IBSNet)

    def on_load_btn_clicked(self):
        self.load_data()
        self.update_info_area()
        self.update_all_camera(np.array([0, 0, self.view_point_radius]))

    def update_metrics_area(self, metrics_tag, metrics_l, metrics_r, metrics_l_container, metrics_r_container,
                            metrics_compare_func, better_color, worse_color):
        metrics_num = len(metrics_tag)
        for i in range(metrics_num):
            metrics_l_container[i].text = self.value_patten.format(self.metrics_scale[metrics_tag[i]] * metrics_l[i])
            metrics_r_container[i].text = self.value_patten.format(self.metrics_scale[metrics_tag[i]] * metrics_r[i])
            is_left_better = metrics_compare_func[i](metrics_l[i], metrics_r[i])
            metrics_l_container[i].background_color = better_color if is_left_better else worse_color
            metrics_r_container[i].background_color = better_color if not is_left_better else worse_color

    def greater_than(self, a, b):
        return a > b

    def less_than(self, a, b):
        return a < b

    def update_info_area(self):
        self.category_info.text = self.category_info_patten.format(
            self.category_selector.get_item(self.selected_category))
        self.scene_info.text = self.scene_info_patten.format(self.scene_selector.get_item(self.selected_scene))
        self.view_info.text = self.view_info_patten.format(self.view_selector.get_item(self.selected_view))

    def on_pre_category_btn_clicked(self):
        if self.selected_category <= 0:
            return
        self.on_category_selection_changed(self.category_selector.get_item(self.selected_category - 1),
                                           self.selected_category - 1)
        self.load_data()
        self.update_info_area()
        self.update_all_camera(np.array([0, 0, self.view_point_radius]))

    def on_next_category_btn_clicked(self):
        if self.selected_category >= self.category_selector.number_of_items - 1:
            return
        self.on_category_selection_changed(self.category_selector.get_item(self.selected_category + 1),
                                           self.selected_category + 1)
        self.load_data()
        self.update_info_area()
        self.update_all_camera(np.array([0, 0, self.view_point_radius]))

    def on_pre_scene_btn_clicked(self):
        if self.selected_scene <= 0:
            return
        self.on_scene_selection_changed(self.scene_selector.get_item(self.selected_scene - 1), self.selected_scene - 1)
        self.load_data()
        self.update_info_area()

    def on_next_scene_btn_clicked(self):
        if self.selected_scene >= self.scene_selector.number_of_items - 1:
            return
        self.on_scene_selection_changed(self.scene_selector.get_item(self.selected_scene + 1), self.selected_scene + 1)
        self.load_data()
        self.update_info_area()

    def on_pre_view_btn_clicked(self):
        if self.selected_view <= 0:
            return
        self.on_view_selection_changed(self.view_selector.get_item(self.selected_view - 1), self.selected_view - 1)
        self.load_data()
        self.update_info_area()

    def on_next_view_btn_clicked(self):
        if self.selected_view >= self.view_selector.number_of_items - 1:
            return
        self.on_view_selection_changed(self.view_selector.get_item(self.selected_view + 1), self.selected_view + 1)
        self.load_data()
        self.update_info_area()

    def on_copy_data_info_clicked(self):
        cur_filename = self.view_selector.get_item(self.selected_view)
        pyperclip.copy(cur_filename)

    def on_show_pcd1_checked(self, is_checked):
        print("show pcd1 checked: ", is_checked)
        self.show_obj1_checked = is_checked
        if is_checked:
            self.add_object(self.scene_gt, self.TAG_obj1, self.mesh_gt_1)
            self.add_object(self.scene_geometric, self.TAG_obj1, self.pcd_partial_1)
            self.add_object(self.scene_grasping_field, self.TAG_obj1, self.pcd_partial_1)
            self.add_object(self.scene_IBSNet, self.TAG_obj1, self.pcd_partial_1)
        else:
            self.remove_object(self.scene_gt, self.TAG_obj1)
            self.remove_object(self.scene_geometric, self.TAG_obj1)
            self.remove_object(self.scene_grasping_field, self.TAG_obj1)
            self.remove_object(self.scene_IBSNet, self.TAG_obj1)

    def on_show_pcd2_checked(self, is_checked):
        print("show pcd2checked: ", is_checked)
        self.show_obj2_checked = is_checked
        if is_checked:
            self.add_object(self.scene_gt, self.TAG_obj2, self.mesh_gt_2)
            self.add_object(self.scene_geometric, self.TAG_obj2, self.pcd_partial_2)
            self.add_object(self.scene_grasping_field, self.TAG_obj2, self.pcd_partial_2)
            self.add_object(self.scene_IBSNet, self.TAG_obj2, self.pcd_partial_2)
        else:
            self.remove_object(self.scene_gt, self.TAG_obj2)
            self.remove_object(self.scene_geometric, self.TAG_obj2)
            self.remove_object(self.scene_grasping_field, self.TAG_obj2)
            self.remove_object(self.scene_IBSNet, self.TAG_obj2)

    def on_show_ibs_checked(self, is_checked):
        print("show ibs checked: ", is_checked)
        self.show_ibs_checked = is_checked
        if is_checked:
            self.add_object(self.scene_gt, self.TAG_IBS, self.ibs_gt)
            self.add_object(self.scene_geometric, self.TAG_IBS, self.ibs_geometric)
            self.add_object(self.scene_grasping_field, self.TAG_IBS, self.ibs_grasping_field)
            self.add_object(self.scene_IBSNet, self.TAG_IBS, self.ibs_IBSNet)
        else:
            self.remove_object(self.scene_gt, self.TAG_IBS)
            self.remove_object(self.scene_geometric, self.TAG_IBS)
            self.remove_object(self.scene_grasping_field, self.TAG_IBS)
            self.remove_object(self.scene_IBSNet, self.TAG_IBS)

    def read_mesh(self, mesh_path):
        if not os.path.exists(mesh_path):
            return None
        if not os.path.isfile(mesh_path):
            self.show_message_dialog("warning", "{} is not a file".format(mesh_path))
            return None
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if np.asarray(mesh.vertices).shape[0] == 0:
            self.show_message_dialog("warning", "{} is not a mesh file".format(mesh_path))
            return None
        mesh.compute_vertex_normals()

        return mesh

    def read_pcd(self, pcd_path):
        if not os.path.exists(pcd_path):
            return None
        if not os.path.isfile(pcd_path):
            self.show_message_dialog("warning", "{} is not a file".format(pcd_path))
            return None
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.normals = o3d.utility.Vector3dVector(np.array([]).reshape(0, 3))
        if np.asarray(pcd.points).shape[0] == 0:
            self.show_message_dialog("warning", "{} is not a pcd file".format(pcd_path))
            return None
        return pcd

    def remove_object(self, scene, name):
        if scene is None:
            return
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)

    def add_object(self, scene, name, geometry):
        if scene is None or geometry is None:
            return
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)
        scene.scene.add_geometry(name, geometry, self.material.get("obj"))

    def reset_all_camera(self):
        self.current_theta = self.default_theta
        self.current_phi = self.default_phi
        self.current_view_point = self.default_view_point
        self.update_all_camera(self.default_view_point)

    def update_all_camera(self, eye):
        self.update_camera(self.scene_gt, eye)
        self.update_camera(self.scene_geometric, eye)
        self.update_camera(self.scene_grasping_field, eye)
        self.update_camera(self.scene_IBSNet, eye)

    def update_camera(self, scene, eye):
        scene.look_at(np.array([0, 0, 0]), eye, np.array([0, 1, 0]))

    def update_view_point(self, theta, phi):
        x = self.view_point_radius * math.sin(math.radians(phi)) * math.cos(math.radians(theta))
        z = self.view_point_radius * math.sin(math.radians(phi)) * math.sin(math.radians(theta))
        y = self.view_point_radius * math.cos(math.radians(phi))
        return np.array([x, y, z])

    def on_layout(self, layout_context):
        print("on layout")
        r = self.window.content_rect

        if self.scene_gt:
            self.scene_gt.frame = gui.Rect(r.x, r.y, self.sub_window_width, self.sub_window_height)
            self.scene_gt_text.frame = gui.Rect(r.x, r.y,
                                                len(self.scene_gt_str) * 15, 0)
        if self.scene_geometric:
            self.scene_geometric.frame = gui.Rect(r.x + self.sub_window_width, r.y, self.sub_window_width,
                                                  self.sub_window_height)
            self.scene_geometric_text.frame = gui.Rect(r.x + self.sub_window_width, r.y,
                                                       len(self.scene_geometric_str) * 15, 0)
        if self.scene_grasping_field:
            self.scene_grasping_field.frame = gui.Rect(r.x, r.y + self.sub_window_height, self.sub_window_width,
                                                       self.sub_window_height)
            self.scene_grasping_field_text.frame = gui.Rect(r.x, r.y + self.sub_window_height,
                                                            len(self.scene_grasping_field_str) * 15, 0)
        if self.scene_IBSNet:
            self.scene_IBSNet.frame = gui.Rect(r.x + self.sub_window_width, r.y + self.sub_window_height,
                                               self.sub_window_width, self.sub_window_height)
            self.scene_IBSNet_text.frame = gui.Rect(r.x + self.sub_window_width, r.y + self.sub_window_height,
                                                    len(self.scene_IBSNet_str) * 15, 0)

        tool_bar_layout_x = r.x + self.sub_window_width * 2
        tool_bar_layout_y = r.y
        tool_bar_layout_width = self.tool_bar_width
        tool_bar_layout_height = r.height
        self.tool_bar_layout.frame = gui.Rect(tool_bar_layout_x, tool_bar_layout_y, tool_bar_layout_width, tool_bar_layout_height)

    def on_key(self, key_event):
        if key_event.type == o3d.visualization.gui.KeyEvent.Type.UP:
            return
        # 切换类别
        if key_event.key == o3d.visualization.gui.KeyName.Q:
            self.on_pre_category_btn_clicked()
        if key_event.key == o3d.visualization.gui.KeyName.E:
            self.on_next_category_btn_clicked()

        # 切换场景
        if key_event.key == o3d.visualization.gui.KeyName.UP:
            self.on_pre_scene_btn_clicked()
        if key_event.key == o3d.visualization.gui.KeyName.DOWN:
            self.on_next_scene_btn_clicked()

        # 切换视角
        if key_event.key == o3d.visualization.gui.KeyName.RIGHT:
            self.on_next_view_btn_clicked()
        if key_event.key == o3d.visualization.gui.KeyName.LEFT:
            self.on_pre_view_btn_clicked()

        # 可见性
        if key_event.key == o3d.visualization.gui.KeyName.ONE:
            self.on_show_pcd1_checked(not self.show_obj1_checked)
        if key_event.key == o3d.visualization.gui.KeyName.TWO:
            self.on_show_pcd2_checked(not self.show_obj2_checked)
        if key_event.key == o3d.visualization.gui.KeyName.THREE:
            self.on_show_ibs_checked(not self.show_ibs_checked)

        # 切换视角
        if key_event.key == o3d.visualization.gui.KeyName.F1:
            self.update_all_camera(np.array([0, 0, self.view_point_radius]))
        if key_event.key == o3d.visualization.gui.KeyName.F2:
            self.update_all_camera(np.array([0, 0, -self.view_point_radius]))
        if key_event.key == o3d.visualization.gui.KeyName.F3:
            self.update_all_camera(np.array([self.view_point_radius, 0, 0]))
        if key_event.key == o3d.visualization.gui.KeyName.F4:
            self.update_all_camera(np.array([-self.view_point_radius, 0, 0]))

        # 变换视角
        if key_event.key == o3d.visualization.gui.KeyName.W:
            self.current_phi = self.current_phi + 5
            if self.current_phi >= 175:
                self.current_phi = 175
            self.current_view_point = self.update_view_point(self.current_theta, self.current_phi)
            self.update_all_camera(self.current_view_point)
        if key_event.key == o3d.visualization.gui.KeyName.S:
            self.current_phi = self.current_phi - 5
            if self.current_phi <= 5:
                self.current_phi = 5
            self.current_view_point = self.update_view_point(self.current_theta, self.current_phi)
            self.update_all_camera(self.current_view_point)
        if key_event.key == o3d.visualization.gui.KeyName.A:
            self.current_theta = (self.current_theta - 5) % 360
            self.current_view_point = self.update_view_point(self.current_theta, self.current_phi)
            self.update_all_camera(self.current_view_point)
        if key_event.key == o3d.visualization.gui.KeyName.D:
            self.current_theta = (self.current_theta + 5) % 360
            self.current_view_point = self.update_view_point(self.current_theta, self.current_phi)
            self.update_all_camera(self.current_view_point)

    def on_dialog_ok(self):
        self.window.close_dialog()

    def init_data_dir_editor_area(self):
        self.data_dir_editor_layout = gui.Vert(0, gui.Margins(self.em, self.em/2, self.em, self.em/2))

        self.ibs_grasping_field_dir_editor_text = gui.Label("ibs_grasping_field_dir")
        self.ibs_grasping_field_dir_editor = gui.TextEdit()
        self.ibs_grasping_field_dir_editor.text_value = self.specs.get("path_options").get("geometries_dir").get("ibs_grasping_field_dir")

        self.ibs_IBSNet_dir_editor_text = gui.Label("ibs_IBSNet_dir")
        self.ibs_IBSNet_dir_editor = gui.TextEdit()
        self.ibs_IBSNet_dir_editor.text_value = self.specs.get("path_options").get("geometries_dir").get("ibs_IBSNet_dir")

        self.data_dir_confirm_btn = gui.Button("confirm")
        self.data_dir_confirm_btn.set_on_clicked(self.on_data_dir_comfirm_btn_clicked)

        self.data_dir_editor_layout.add_child(self.ibs_grasping_field_dir_editor_text)
        self.data_dir_editor_layout.add_child(self.ibs_grasping_field_dir_editor)
        self.data_dir_editor_layout.add_fixed(self.em / 2)
        self.data_dir_editor_layout.add_child(self.ibs_IBSNet_dir_editor_text)
        self.data_dir_editor_layout.add_child(self.ibs_IBSNet_dir_editor)
        self.data_dir_editor_layout.add_fixed(self.em / 2)
        self.data_dir_editor_layout.add_child(self.data_dir_confirm_btn)

    def init_geometry_select_area(self):
        self.geometry_select_layout = gui.Vert(self.em / 2, gui.Margins(self.em, self.em/2, self.em, self.em/2))

        # 类别
        self.selected_category = -1
        self.category_selector_layout = gui.Vert()
        self.category_selector_text = gui.Label("category")
        self.category_selector = gui.Combobox()
        self.category_selector.set_on_selection_changed(self.on_category_selection_changed)
        self.category_selector_layout.add_child(self.category_selector_text)
        self.category_selector_layout.add_child(self.category_selector)

        # 场景
        self.selected_scene = -1
        self.scene_selector_layout = gui.Vert()
        self.scene_selector_text = gui.Label("scene")
        self.scene_selector = gui.Combobox()
        self.scene_selector.set_on_selection_changed(self.on_scene_selection_changed)
        self.scene_selector_layout.add_child(self.scene_selector_text)
        self.scene_selector_layout.add_child(self.scene_selector)

        # 场景
        self.selected_view = -1
        self.view_selector_layout = gui.Vert()
        self.view_selector_text = gui.Label("view")
        self.view_selector = gui.Combobox()
        self.view_selector.set_on_selection_changed(self.on_view_selection_changed)
        self.view_selector_layout.add_child(self.view_selector_text)
        self.view_selector_layout.add_child(self.view_selector)

        # 确认
        self.btn_load = gui.Button("load data")
        self.btn_load.set_on_clicked(self.on_load_btn_clicked)

        self.geometry_select_layout.add_child(self.category_selector_layout)
        self.geometry_select_layout.add_child(self.scene_selector_layout)
        self.geometry_select_layout.add_child(self.view_selector_layout)
        self.geometry_select_layout.add_child(self.btn_load)

    def init_data_switch_area(self):
        self.data_switch_layout = gui.Vert(0, gui.Margins(self.em, self.em/2, self.em, self.em/2))

        self.category_switch_area = gui.Vert()
        self.category_switch_text = gui.Label("switch category")
        self.btn_pre_category = gui.Button("previous category")
        self.btn_pre_category.set_on_clicked(self.on_pre_category_btn_clicked)
        self.btn_next_category = gui.Button("next category")
        self.btn_next_category.set_on_clicked(self.on_next_category_btn_clicked)
        self.category_switch_area.add_child(self.category_switch_text)
        self.category_switch_area.add_child(self.btn_pre_category)
        self.category_switch_area.add_child(self.btn_next_category)

        self.scene_switch_area = gui.Vert()
        self.scene_switch_text = gui.Label("switch scene")
        self.btn_pre_scene = gui.Button("previous scene")
        self.btn_pre_scene.set_on_clicked(self.on_pre_scene_btn_clicked)
        self.btn_next_scene = gui.Button("next scene")
        self.btn_next_scene.set_on_clicked(self.on_next_scene_btn_clicked)
        self.scene_switch_area.add_child(self.scene_switch_text)
        self.scene_switch_area.add_child(self.btn_pre_scene)
        self.scene_switch_area.add_child(self.btn_next_scene)

        self.view_switch_area = gui.Vert()
        self.view_switch_text = gui.Label("switch view")
        self.btn_pre_view = gui.Button("previous view")
        self.btn_pre_view.set_on_clicked(self.on_pre_view_btn_clicked)
        self.btn_next_view = gui.Button("next view")
        self.btn_next_view.set_on_clicked(self.on_next_view_btn_clicked)
        self.view_switch_area.add_child(self.view_switch_text)
        self.view_switch_area.add_child(self.btn_pre_view)
        self.view_switch_area.add_child(self.btn_next_view)

        self.data_switch_layout.add_child(self.category_switch_area)
        self.data_switch_layout.add_fixed(self.em)
        self.data_switch_layout.add_child(self.scene_switch_area)
        self.data_switch_layout.add_fixed(self.em)
        self.data_switch_layout.add_child(self.view_switch_area)

    def init_visible_control_area(self):
        self.visible_control_layout = gui.Vert(0, gui.Margins(self.em, self.em/2, self.em, self.em/2))

        self.visible_text = gui.Label("visible")

        self.visible_control_checkbox_layout = gui.Horiz()
        self.show_obj1_checkbox_text = gui.Label("obj1")
        self.show_obj2_checkbox_text = gui.Label("obj2")
        self.show_ibs_checkbox_text = gui.Label("ibs")
        self.show_obj1_checkbox = gui.Checkbox("")
        self.show_obj1_checkbox.checked = True
        self.show_obj1_checked = True
        self.show_obj2_checkbox = gui.Checkbox("")
        self.show_obj2_checkbox.checked = True
        self.show_obj2_checked = True
        self.show_ibs_checkbox = gui.Checkbox("")
        self.show_ibs_checkbox.checked = True
        self.show_ibs_checked = True
        self.show_obj1_checkbox.set_on_checked(self.on_show_pcd1_checked)
        self.show_obj2_checkbox.set_on_checked(self.on_show_pcd2_checked)
        self.show_ibs_checkbox.set_on_checked(self.on_show_ibs_checked)

        self.visible_control_checkbox_layout.add_child(self.show_obj1_checkbox_text)
        self.visible_control_checkbox_layout.add_child(self.show_obj1_checkbox)
        self.visible_control_checkbox_layout.add_stretch()
        self.visible_control_checkbox_layout.add_child(self.show_obj2_checkbox_text)
        self.visible_control_checkbox_layout.add_child(self.show_obj2_checkbox)
        self.visible_control_checkbox_layout.add_stretch()
        self.visible_control_checkbox_layout.add_child(self.show_ibs_checkbox_text)
        self.visible_control_checkbox_layout.add_child(self.show_ibs_checkbox)

        self.visible_control_layout.add_child(self.visible_text)
        self.visible_control_layout.add_child(self.visible_control_checkbox_layout)

    def init_data_info_area(self):
        self.data_info_layout = gui.Vert(0, gui.Margins(self.em, self.em/2, self.em, self.em/2))

        self.category_info = gui.Label("category: {}".format(""))
        self.scene_info = gui.Label("scene: {}".format(""))
        self.view_info = gui.Label("view: {}".format(""))
        self.btn_copy_cur_data_info = gui.Button("copy")
        self.btn_copy_cur_data_info.set_on_clicked(self.on_copy_data_info_clicked)

        self.data_info_layout.add_child(self.category_info)
        self.data_info_layout.add_child(self.scene_info)
        self.data_info_layout.add_child(self.view_info)
        self.data_info_layout.add_child(self.btn_copy_cur_data_info)

    def get_geometry_path(self):
        mesh_dir = self.specs.get("path_options").get("geometries_dir").get("mesh_dir")
        pcd_partial_dir = self.specs.get("path_options").get("geometries_dir").get("pcd_partial_dir")
        ibs_gt_dir = self.specs.get("path_options").get("geometries_dir").get("ibs_gt_dir")
        ibs_geometric_dir = self.specs.get("path_options").get("geometries_dir").get("ibs_geometric_dir")
        ibs_grasping_field_dir = self.specs.get("path_options").get("geometries_dir").get("ibs_grasping_field_dir")
        ibs_IBSNet_dir = self.specs.get("path_options").get("geometries_dir").get("ibs_IBSNet_dir")

        category = self.category_selector.get_item(self.selected_category)
        scene = self.scene_selector.get_item(self.selected_scene)
        view = self.view_selector.get_item(self.selected_view)

        mesh_1_filename = "{}_0.obj".format(scene)
        mesh_2_filename = "{}_1.obj".format(scene)
        pcd_partial_1_filename = "{}_0.ply".format(view)
        pcd_partial_2_filename = "{}_1.ply".format(view)
        ibs_gt_filename = "{}.ply".format(scene)
        ibs_filename = "{}.ply".format(view)

        mesh_1_path = os.path.join(mesh_dir, category, mesh_1_filename)
        mesh_2_path = os.path.join(mesh_dir, category, mesh_2_filename)
        pcd_partial_1_path = os.path.join(pcd_partial_dir, category, pcd_partial_1_filename)
        pcd_partial_2_path = os.path.join(pcd_partial_dir, category, pcd_partial_2_filename)
        ibs_gt_path = os.path.join(ibs_gt_dir, category, ibs_gt_filename)
        ibs_geometric_path = os.path.join(ibs_geometric_dir, category, ibs_filename)
        ibs_grasping_field_path = os.path.join(ibs_grasping_field_dir, category, ibs_filename)
        ibs_IBSNet_path = os.path.join(ibs_IBSNet_dir, category, ibs_filename)

        geometry_path_dict = {
            self.key_mesh_1: mesh_1_path,
            self.key_mesh_2: mesh_2_path,
            self.key_pcd_partial_1: pcd_partial_1_path,
            self.key_pcd_partial_2: pcd_partial_2_path,
            self.key_ibs_gt: ibs_gt_path,
            self.key_ibs_geometric: ibs_geometric_path,
            self.key_ibs_grasping_field: ibs_grasping_field_path,
            self.key_ibs_IBSNet: ibs_IBSNet_path
        }

        return geometry_path_dict

    def show_message_dialog(self, title, message):
        dlg = gui.Dialog(title)
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self.on_dialog_ok)

        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        dlg_layout.add_child(button_layout)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def run(self):
        try:
            gui.Application.instance.run()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    config_filepath = 'configs/IBSNet_result_visualize_utils.json'
    specs = path_utils.read_config(config_filepath)

    app = App(specs)
    app.run()
