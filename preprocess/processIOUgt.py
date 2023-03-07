import os
import re
import shutil

file_dir = '../data/boundingBox'
filename_obj_re = "scene\\d\\.\\d{4}.*obj"
filename_txt_re = "scene\\d\\.\\d{4}.*txt"
filename_re = "scene\\d\\.\\d{4}"
catrgory_re = "scene\\d"
categories = ["scene1", "scene2", "scene3", "scene4", "scene5", "scene6", "scene7", "scene8", "scene9"]

# for category in categories:
#     if not os.path.isdir(os.path.join(file_dir, category)):
#         os.mkdir(os.path.join(file_dir, category))


filename_list = os.listdir(file_dir)
for filename in filename_list:
    if not os.path.isfile(os.path.join(file_dir, filename)):
        continue
    src_path = os.path.join(file_dir, filename)
    category = re.match(catrgory_re, filename).group()
    dest_path = os.path.join(file_dir, category, filename)

    shutil.copy(src_path, dest_path)

