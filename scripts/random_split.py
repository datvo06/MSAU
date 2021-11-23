import os
import glob
import random

path_prefix = "./path_prefix"
data_path = "/path/to/training/json"

def random_split(dir, train_ratio):
    file_list = []
    data_dir = os.path.join(data_path, dir)

    print(data_dir)

    for file_path in glob.glob(data_dir + "/*.json"):
        filename = os.path.basename(file_path)
        basename = filename.split('.')[0]
        ext = filename.split('.')[-1]
        file_list.append(basename + '.' + ext)

    random.shuffle(file_list)

    split_index = int(train_ratio * len(file_list))
    train_list = file_list[:split_index]
    val_list = file_list[split_index:]

    train_list = [path_prefix.format(dir) + filename for filename in train_list]
    val_list = [path_prefix.format(dir) + filename for filename in val_list]

    return train_list, val_list


train_list, val_list = random_split('', 0.75)

with open(data_path + '/train.lst', 'w') as f:
    for filename in train_list:
        f.writelines(filename + "\n")

with open(data_path + '/val.lst', 'w') as f:
    for filename in val_list:
        f.writelines(filename + "\n")
