#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import os
import json
from PIL import Image, ImageDraw, ImageFont
from .morph_util import scale_rect, threshold_and_upscale_map, union_boxes, IoU
import cv2

FONT_PATH = './resources/Dengb.ttf'

def read_image_list(pathToList, prefix=None):
    '''

    :param pathToList:
    :return:
    '''
    f = open(pathToList, 'r')
    filenames = []
    for line in f:
        if line[0] != '#':
            if line[-1] == '\n':
                filename = line[:-1]
            else:
                filename = line
            if prefix is not None:
                filename = prefix + filename
            filenames.append(filename)
        else:
            if len(line) < 3:
                break
    f.close()
    return filenames#

def glob_folder(path, extension):
    file_map = {}
    for dirpath, subdir, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith("{}".format(extension)):
                basename = os.path.basename(filename).split('.')[0]
                if basename not in file_map:
                    file_map[basename] = os.path.join(dirpath, filename)
                else:
                    print('Duplicated file name: {}, existing file: {}'.format(os.path.join(dirpath, filename), file_map[basename]))
    return file_map



def sort_box_reading_order(boxes):
   """ Sort cell list to create the right reading order using their locations

   :param boxes: list of cells to sort
   :returns: a list of cell lists in the right reading order that contain no key or start with a key and contain no other key
   :rtype: list of lists of cells

   """
   sorted_list = []
   if len(boxes) == 0:
       return boxes

   while len(boxes) > 1:
       topleft_box = boxes[0]

       for box in boxes[1:]:

           topleft_box_pos = topleft_box['box']
           topleft_box_center_x = (
                                      topleft_box_pos[0] + topleft_box_pos[2]) / 2
           topleft_box_center_y = (
                                      topleft_box_pos[1] + topleft_box_pos[3]) / 2

           x1, y1, x2, y2 = box['box']
           box_center_x = (x1 + x2) / 2
           box_center_y = (y1 + y2) / 2
           cell_h = y2 - y1

           if box_center_y <= topleft_box_center_y - cell_h / 2:
               topleft_box = box
               continue

           if box_center_x < topleft_box_pos[2] and box_center_y < topleft_box_pos[3]:
               topleft_box = box
               continue

       sorted_list.append(topleft_box)
       boxes.remove(topleft_box)

   sorted_list.append(boxes[0])

   return sorted_list

def to_categorical(target_vector, n_labels):
    return np.eye(n_labels, dtype='B')[target_vector]

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)


def visualize_kv_map(img, kv_mask, draw, font, value_id, text, box, color='red'):
    x1, y1, x2, y2 = box
    draw_rectangle(draw, ((x1, y1), (x2, y2)), color=color, width=3)
    draw.text((x1, y2 + 5), '{}'.format(value_id), fill=color, font=font)

    if value_id[0] == 'v':
    # if int(value_id) % 2 == 0:
        # draw.text((x2 + 12, y1 + 5), '{}'.format(value_id), fill="magenta", font=font)
        # draw.text((x1, y2 + 5), '{}'.format(value_id), fill="magenta", font=font)
        draw.text((x1, y1 + 5), text, fill="green", font=font)


def visualize_kv_results(pred_mask, values, input_im, debug_im, visualize_shape, visualize_scale, n_class, class_names=None,
                         ca_text='', eval_results=None, correct_answers=None):
    if len(debug_im.shape) < 3:
        debug_im = cv2.cvtColor(debug_im, cv2.COLOR_GRAY2BGR)
    # origin_im = cv2.resize(debug_im, visualize_shape[::-1])
    origin_im = debug_im
    debug_im = origin_im.copy()

    input_bg = 1 - threshold_and_upscale_map(visualize_shape, (input_im[:, :, 0] * 255).astype('uint8'),
                                             threshold=80)
    input_missing = threshold_and_upscale_map(visualize_shape, (input_im[:, :, 1] * 255).astype('uint8'),
                                              threshold=80)
    # debug_im[input_bg > 0] = [0, 255, 0]
    # debug_im[input_missing > 0] = [255, 0, 0]

    debug_im = cv2.addWeighted(origin_im, 0.3, debug_im, 0.7, 0.0)

    output_bg = 1 - threshold_and_upscale_map(visualize_shape, (pred_mask[:, :, 0] * 255).astype('uint8'), threshold=240)
    debug_im[output_bg > 0] = [0, 255, 255] #[0, 120, 255]

    c_maps = []
    for value_id in range(1, n_class):
        c_map = threshold_and_upscale_map(visualize_shape, (pred_mask[:, :, value_id] * 255).astype('uint8'),
                                          threshold=120)
        c_maps.append(c_map)
        if value_id % 2 == 1:
            debug_im[c_map > 0] = [0, 0, 255]
        else:
            debug_im[c_map > 0] = [255, 0, 0] #[0, 255, 255]

    debug_im = cv2.addWeighted(origin_im, 0.6, debug_im, 0.4, 0.0)
    clone_img = Image.fromarray(debug_im[:, :, ::-1])
    draw = ImageDraw.Draw(clone_img)
    font = ImageFont.truetype(
        FONT_PATH,
        size=15,
        encoding='utf-8-sig')

    for value_id in range(1, n_class):
        text = values[value_id][0]

        if values[value_id][1] is None: continue

        for box in values[value_id][1]:

            color = 'magenta'

            if eval_results is not None:
                eval_results[value_id]['num_pred'] += 1

                if correct_answers is not None:
                    if value_id in correct_answers:
                        gt_boxes = correct_answers[value_id][0][:1]
                    else:
                        gt_boxes = []

                    is_matched = False
                    for gt_box in gt_boxes:
                        if IoU(box, gt_box) > 0.7:
                            is_matched = True
                            break

                    if is_matched:
                        eval_results[value_id]['num_correct'] += 1
                        color = 'blue'

            box = scale_rect(box, visualize_scale)

            if class_names is not None:
                visualize_kv_map(debug_im, c_maps[value_id - 1], draw, font, class_names[value_id - 1], text, box, color=color)
            else:
                visualize_kv_map(debug_im, c_maps[value_id - 1], draw, font, str(value_id - 1), text, box, color=color)

    draw.text((5, 15), ca_text, fill="green", font=font)

    return clone_img


def visualize_gt_boxes(debug_im, gt_boxes, color = 'red'):
    clone_img = Image.fromarray(debug_im[:, :, ::-1])
    draw = ImageDraw.Draw(clone_img)
    font = ImageFont.truetype(
        FONT_PATH,
        size=15,
        encoding='utf-8-sig')

    for box, value_id in gt_boxes:
        x1, y1, x2, y2 = box
        draw_rectangle(draw, ((x1, y1), (x2, y2)), color=color, width=3)
        draw.text((x2 + 6, y1 + 5), 'v{}'.format(value_id), fill=color, font=font)

    return np.array(clone_img)[:, :, ::-1]


def read_json_gt(json_path, scale=1.0, offset=(0, 0)):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    im_shape = [int(d * scale) for d in json_dict['img_shape']]
    label_lines = json_dict['lines']

    value_boxes = {}
    offset_x, offset_y = offset

    for line_idx, line in enumerate(label_lines):
        box, text, value_idx, type_idx = (line[k] for k in ['box', 'text', 'value', 'type'])
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 - offset_x, y1 - offset_y, x2 - offset_x, y2 - offset_y
        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        line['box'] = [x1, y1, x2, y2]

        if value_idx > 0 and type_idx > 0: #== 2:
            value_idx = value_idx + 1
            if value_idx in value_boxes:
                value_boxes[value_idx].append(line)
            else:
                value_boxes[value_idx] = [line]

    correct_answers = {}

    for value_id, boxes in value_boxes.items():
        boxes = sort_box_reading_order(boxes)
        merged_box = [union_boxes([b['box'] for b in boxes]), ] + [b['box'] for b in boxes] #[b['box'] for b in boxes] #
        value_text = ''.join([b['text'] for b in boxes])
        if value_id not in [1,]:
            correct_answers[value_id] = (merged_box, value_text)

    # print("CA")
    # for value_id, item in correct_answers.items():
    #     print(value_id, item[1])

    return correct_answers


def write_csv_report_by_row(file_list, kv_results, output_path, ca_map=None):
    with open(output_path, 'w') as csvfile:
        report = csv.writer(csvfile, delimiter=',')
        keys = kv_results[0].keys()
        headers = ['file_name', 'field_name', 'predict', 'correct_answer', 'T/F']

        report.writerow(headers)
        for f, item in zip(file_list, kv_results):
            f = os.path.basename(f).split('.')[0]
            if ca_map is not None:
                ca_row = ca_map[f]
                for k in keys:
                    report.writerow([f, k, item[k], ca_row[k][1], ca_row[k][0]])
            else:
                for k in keys:
                    report.writerow([f, k, item[k], '(none)', '(none)'])


def write_csv_report(file_list, kv_results, output_path, ca_map=None):
    with open(output_path, 'w') as csvfile:
        report = csv.writer(csvfile, delimiter=',')
        keys = kv_results[0].keys()
        headers = ['filename',]
        for k in kv_results[0].keys():
            headers.append(k)
            if ca_map is not None:
                headers.append('correct_answer')

        report.writerow(headers)
        for f, item in zip(file_list, kv_results):
            f = os.path.basename(f).split('.')[0]
            if ca_map is not None:
                ca_row = ca_map[f]
            results = []
            for k in keys:
                results.append(item[k])
                if ca_map is not None:
                    results.append(ca_row[k][1])
            report.writerow([f,] + results)

