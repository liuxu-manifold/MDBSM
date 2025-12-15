from glob import glob
import os

def update(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

# your_dataset_path = "/data/zjl/192-torch/pytorch-coviar-master/data/AFEW_origin/AFEW_imges_align_resize/"
# all_txt_file = glob(os.path.join('AFEW_*.txt'))
# for txt_file in all_txt_file:
#     update(txt_file, "/home/user/datasets/AFEW_Face/", your_dataset_path)


your_dataset_path = "/data/zjl/192-torch/pytorch-coviar-master/data/MAFW/"
all_txt_file = glob(os.path.join('/data/zjl/192-torch/pytorch-coviar-master/data/datalists/MAFW-txt/*.txt'))
for txt_file in all_txt_file:
    update(txt_file, "/data/zjl/192-torch/pytorch-coviar-master/data/", your_dataset_path)



# your_dataset_path = ".../DFEW_Face/"
# all_txt_file = glob(os.path.join('DFEW_*.txt'))
# for txt_file in all_txt_file:
#     update(txt_file, "/home/user/datasets/DFEW_Face/", your_dataset_path)

# your_dataset_path = ".../FERV39K/"
# all_txt_file = glob(os.path.join('FERV39K_*.txt'))
# for txt_file in all_txt_file:
#     update(txt_file, "/home/user/datasets/FERV39K/", your_dataset_path)