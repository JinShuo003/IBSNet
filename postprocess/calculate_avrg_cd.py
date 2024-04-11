"""
统计各类别平均cd
"""
import json
import logging
import os

from utils import path_utils


def get_cd(specs, category, instance_name):
    cd_data_dir = specs.get("path_options").get("cd_data_dir")
    cd_filepath = os.path.join(cd_data_dir, category, '{}.txt'.format(instance_name))
    with open(cd_filepath, 'r') as f:
        cd = f.read()
        return float(cd)


if __name__ == '__main__':
    config_filepath = 'configs/calculate_avrg_cd.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("cd_data_dir"))
    path_utils.generate_path(specs.get("path_options").get("cd_avrg_save_dir"))

    logger = logging.getLogger("calculate_avrg_cd")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # 参数
    view_list = []
    cd_dict = {}
    cd_dict['total'] = 0
    vie_num_total = 0
    for category in filename_tree:
        cd_dict[category] = 0
        view_num = 0
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                cd = get_cd(specs, category, filename)
                if cd != 0.0:
                    cd_dict['total'] += cd
                    cd_dict[category] += cd
                    view_num += 1
                    vie_num_total += 1
        cd_dict[category] /= view_num
    cd_dict['total'] /= vie_num_total

    cd_avrg_save_dir = specs.get("path_options").get("cd_avrg_save_dir")
    cd_json = json.dumps(cd_dict, indent=1)

    cd_path = os.path.join(cd_avrg_save_dir, "cd_avrg.json")

    with open(cd_path, 'w', newline='\n') as f:
        f.write(cd_json)
