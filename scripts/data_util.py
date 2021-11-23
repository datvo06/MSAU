import json
import os

def value_in_range(value, min_v, max_v):
    return value >= min_v and value <= max_v


def glob_folder(path, extension, use_dirname=False):
    file_map = {}
    for dirpath, subdir, filenames in os.walk(path):
        for filepath in filenames:
            if filepath.endswith("{}".format(extension)):
                if use_dirname:
                    dirname = os.path.basename(dirpath)
                    basename = dirname
                else:
                    basename = os.path.basename(filepath).split('.')[0]
                if basename not in file_map:
                    file_map[basename] = os.path.join(dirpath, filepath)
                else:
                    print('Duplicated label file: {}, existing file: {}'.format(os.path.join(dirpath, filepath),
                                                                                file_map[basename]))
    return file_map


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline=color)


def export_label_json(export_path, label_lines, im_shape):
    json_dict = {'img_shape': im_shape, 'lines': []}
    for line in label_lines:
        box, text, type_idx, value_idx = line
        json_dict['lines'].append({'box': box, 'text': text, 'type': type_idx, 'value': value_idx})
    with open(export_path, 'w') as f:
        json.dump(json_dict, f)
