import json
import torch
import cv2
import os
import numpy as np

from .morph_util import r_closing, area, intersect_boxes, union_boxes, r_dilation, \
    connected_components, xcenter, ycenter, max_dim, filter_overlap_boxes, filter_overlap_boxes_bigger, scale_rect

from .generic_util import sort_box_reading_order, to_categorical, \
    visualize_kv_results, read_image_list, glob_folder, read_json_gt, visualize_gt_boxes
from .postprocess import CLASS_NAMES, post_process_kv


class KVModel():
    """
        Inference Key-Value model

        """

    default_config = {
        'scale': 3.0,
        'charset': '',
        'model_kv': '',
        'n_class': 0
    }

    def __init__(self):
        self.net = None
        self.scale = None
        self.tok_to_id, self.id_to_tok = None, None
        self.blank_idx = 1
        self.n_token = 1
        self.charset = ''
        self.n_class = 1

    def load(self, **config):
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net.load_weights(config['model_weight'])
        self.net.eval()
        scale = self.default_config['scale']
        self.scale = scale
        path_charset = config["charset"]

        if path_charset is not None:
            with open(path_charset, 'r') as f:
                self.charset = ' ' + '$' + f.read()
                self.blank_idx = 1
                self.tok_to_id = {tok: idx for idx, tok in
                                  enumerate(self.charset)}
                self.id_to_tok = {idx: tok for tok, idx in self.tok_to_id.items()}
                self.n_token = len(self.tok_to_id)
        else:
            self.charset = None

        self.n_class = config['n_class']

    ### load json input from layout & ocr models
    @staticmethod
    def _read_json_layout_ocr(json_path):

        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        new_dict = json_dict

        # new_dict = {'img_shape': [], 'lines': []}
        # for name, item in json_dict.items():
        #     if 'line' not in name: continue
        #     y1, x1, y2, x2 = item['location']
        #     if 'text' in item:
        #         text = item['text']
        #     elif 'value' in item:
        #         text = item['value']
        #     else:
        #         text = ''
        #     new_dict['lines'].append({'box': [x1, y1, x2, y2], 'text': text, 'value': 0, 'type': 0})

        return new_dict

    ### create input mask
    def _generate_masks_from_label(self, label_path):

        json_dict = self._read_json_layout_ocr(label_path)

        label_lines = json_dict['lines']
        line_heights = [l['box'][3] - l['box'][1] for l in label_lines]

        min_x, min_y, max_x, max_y = min([l['box'][0] for l in label_lines]), min([l['box'][1] for l in label_lines]), \
                                     max([l['box'][2] for l in label_lines]), max([l['box'][3] for l in label_lines])
        bounding_box = (min_x, min_y, max_x, max_y)

        median_h = np.median(line_heights)

        bg_pad = int(median_h * 3)
        min_x, min_y = min_x - bg_pad, min_y - bg_pad
        max_x, max_y = max_x + bg_pad, max_y + bg_pad

        scale = 3.0 / median_h
        v_scale = 1.0
        h_scale = 1.0

        w, h = max_x - min_x, max_y - min_y
        im_shape = [int(h * scale * v_scale), int(w * scale * h_scale)]

        input_mask = np.zeros(list(im_shape), dtype='uint16')
        output_mask = np.zeros(list(im_shape), dtype='uint16')

        line_id_mask = np.zeros(list(im_shape), dtype='uint16')
        character_id_mask = np.zeros(list(im_shape), dtype='uint16')

        for line_idx, line in enumerate(label_lines):
            box, text, type_idx, value_idx = (line[k] for k in ['box', 'text', 'type', 'value'])
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y

            # cw, ch = (x2 - x1) * scale, (y2 - y1) * scale
            # x1, y1 = int(x1 * scale * h_scale), int(y1 * scale * v_scale)
            # x2, y2 = int(x1 + cw), int(y1 + ch)

            x1, y1, x2, y2 = int(x1 * scale * h_scale), int(y1 * scale * v_scale), int(x2 * scale * h_scale), int(
                y2 * scale * v_scale)

            line['box'] = [x1, y1, x2, y2]
            text = ''.join([c if not c.isdigit() else '0' for c in text])

            if len(text) > 0:

                char_full_w = max(1.0 * (x2 - x1) / len(text), 1.0)
                # if len(text) == 1:
                #     char_w = max(1.0 * char_full_w, 1.0)
                # else:
                char_w = max(0.9 * char_full_w, 1.0)
                char_w = min(char_w, int((y2 - y1) * 1.2))
                line_id_mask[y1:y2, x1:x2] = line_idx + 1

                for idx, c in enumerate(text):
                    offset = x1 + idx * char_full_w
                    char_id = self.tok_to_id[c] if c in self.tok_to_id else self.blank_idx

                    start_x, end_x = int(offset), int(offset + char_w)
                    input_mask[y1:y2, start_x:end_x] = char_id

                    line_id_mask[y1:y2, start_x:end_x] = line_idx + 1
                    character_id_mask[y1:y2, start_x:end_x] = idx + 1

        return input_mask, line_id_mask, character_id_mask, label_lines, scale, bg_pad, bounding_box

    @staticmethod
    def _extract_value(line_mask, char_mask, label_lines, pred_mask, num_classes):
        n_class = pred_mask.shape[2]

        ### define special fields
        multiple_lines_fields = [5, 11]
        non_count_overlap_fields = []
        contain_one_line_fields = []

        num_lines = len(label_lines)

        values = [('', None, None, None)] * n_class
        pred_class = np.argmax(pred_mask, axis=-1)
        new_pred_mask = np.zeros(pred_mask.shape)
        new_pred_mask[:, :, 0] = pred_mask[:, :, 0]
        line_used_count = [0] * (num_lines + 1)


        line_ids_for_field = [[] for _ in range(num_classes + 1)]
        boxes_for_field = [[] for _ in range(num_classes + 1)]

        for idx, l in enumerate(label_lines):
            l['id'] = idx + 1

        for c in range(2, n_class):
            c_map = pred_class == c  # pred_mask[:, :, c] > 0.25 #
            c_map = r_closing(c_map, (1, 3))
            labels, objects = connected_components(c_map)

            if len(objects) == 0:
                continue

            if c in multiple_lines_fields:
                max_area_id_list = np.argsort([-ycenter(o) for o in objects])
            else:
                max_area_id_list = np.argsort([area(o) for o in objects])

            if len(max_area_id_list) == 0:
                continue
            else:
                max_area_id = max_area_id_list[-1]
                alt_max_area_ids = []
                if area(objects[max_area_id]) < 5:
                    continue

                if c in multiple_lines_fields and len(max_area_id_list) > 1:
                # if len(max_area_id_list) > 1:
                    for alt_max_area_id in max_area_id_list[:-1]:
                        if area(objects[alt_max_area_id]) > 5:
                            alt_max_area_ids.append(alt_max_area_id)
                            box = objects[alt_max_area_id]
                            boxes_for_field[c].append([box[1].start, box[0].start, box[1].stop, box[0].stop])

            box = objects[max_area_id]
            boxes_for_field[c].append([box[1].start, box[0].start, box[1].stop, box[0].stop])
            line_ids = [idx for idx in np.unique(line_mask[labels == max_area_id + 1]) if idx > 0]

            if len(alt_max_area_ids) > 0:
                for alt_max_area_id in alt_max_area_ids:
                    line_ids += [idx for idx in np.unique(line_mask[labels == alt_max_area_id + 1]) if idx > 0]
                    new_pred_mask[:, :, c][labels == alt_max_area_id + 1] = 1

            line_ids_for_field[c] = list(set(line_ids))

            if c not in non_count_overlap_fields:
                for idx in line_ids:
                    line_used_count[idx] += 1

            new_pred_mask[:, :, c][labels == max_area_id + 1] = 1

        for c in range(2, n_class):
            line_ids = line_ids_for_field[c]
            if len(line_ids) == 0: continue
            value = ''
            lines = sort_box_reading_order([label_lines[i - 1] for i in line_ids if i > 0])
            line_boxes = []

            for line in lines:
                line_id = line['id']
                line_boxes.append(line['box'])
                if line_used_count[line_id] <= 1:
                    value += line['text']
                else:
                    x1, y1, x2, y2 = line['box']
                    char_mask_line = set(np.unique(char_mask[y1:y2, x1:x2][new_pred_mask[:, :, c][y1:y2, x1:x2] > 0]))
                    if 0 in char_mask_line: char_mask_line.remove(0)
                    if len(char_mask_line) == 0:
                        continue
                    char_pos_min, char_pos_max = min(char_mask_line), max(char_mask_line)
                    if char_pos_max > len(line['text']) - 3:
                        char_pos_max = len(line['text']) + 1
                    value += line['text'][char_pos_min - 2 if char_pos_min >= 2 else 0: char_pos_max - 1]

                if c in contain_one_line_fields and len(value) > 2:
                    break
                if c in multiple_lines_fields:
                    value += '\n'

            if len(value) > 0 and value[-1] == '\n': value = value[:-1]
            merged_box = union_boxes(line_boxes)

            intersect_box = intersect_boxes(boxes_for_field[c] + [merged_box,])
            union_box = union_boxes(boxes_for_field[c] + [merged_box,])
            list_boxes = [boxes_for_field[c][-1]] # boxes_for_field[c]

            values[c] = (value, list_boxes, intersect_box, union_box)

        # boxes = [v[1] if v[1] is not None else [0, 0, 0, 0] for v in values]
        # is_overlaps = filter_overlap_boxes_bigger(boxes, return_indices=True, intersect_thres=0.7)
        # values = [v if not is_overlaps[id] else ('', v[1]) for id, v in enumerate(values)]

        return values, new_pred_mask

    #### main function, ca is only used for displaying DEBUG text, and can be safely removed
    def predict(self, data, debug_info=None, label_path=None, eval_results=None):
        json_path, debug_im = data

        input_im, line_mask, char_mask, label_lines, scale, bg_pad, (
        min_x, min_y, max_x, max_y) = self._generate_masks_from_label(json_path)

        debug_im = debug_im[min_y:max_y, min_x:max_x]
        debug_im = cv2.copyMakeBorder(debug_im, bg_pad, bg_pad, bg_pad, bg_pad, borderType=cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))

        input_im = to_categorical(input_im, self.n_token)
        batch_x = np.expand_dims(input_im, 0)
        batch_x = torch.from_numpy(batch_x).transpose(1, -1).transpose(2, 3)
        if torch.cuda.is_available():
            batch_x = batch_x.float().cuda()

        visualize_scale = 6
        visualize_shape = tuple((d * visualize_scale for d in input_im.shape[:2]))
        debug_im = cv2.resize(debug_im, visualize_shape[::-1])

        ## correct answer evaluation
        correct_answers = None
        if label_path is not None:
            try:
                correct_answers = read_json_gt(label_path, scale=scale, offset=(min_x - bg_pad, min_y - bg_pad))
            except IOError as e:
                print('Error reading CA', e)

        if correct_answers is not None:
            gt_boxes = []
            for value_id in correct_answers:
                eval_results[value_id]['num_label'] += 1 #len(correct_answers[value_id][0])

                for box in correct_answers[value_id][0][1:]:
                    box = scale_rect(box, visualize_scale)
                    gt_boxes.append((box, value_id))

            debug_im = visualize_gt_boxes(debug_im, gt_boxes)

        # Run model
        a_pred = None
        with torch.set_grad_enabled(False):
            a_pred, _, _ = self.net(batch_x)
            a_pred = torch.transpose(a_pred, 1, -1).transpose(1, 2)
            if torch.cuda.is_available():
                a_pred = a_pred.cpu().numpy()

        values, pred_mask = self._extract_value(
            line_mask, char_mask, label_lines, a_pred[0], self.n_class
        )

        kv_results = post_process_kv(values)

        if debug_info is not None:
            ca_text, ca_map = debug_info
        else:
            ca_text = ''
            ca_map = None

        ca_text += '\nPRED\n'
        # all_true = True
        for k in sorted(kv_results):
            if ca_map is not None and k in ca_map:
                status = '' if kv_results[k] == ca_map[k][1] else '(F)'
                # if status == '(F)': all_true = False
            else:
                status = '-'
            ca_text += '{} : {}  {}\n'.format(k, kv_results[k][:20].replace('\n', '') if len(
                kv_results[k]) > 0 else '(empty)', status)

        debug_im = visualize_kv_results(pred_mask, values, input_im, debug_im, visualize_shape, visualize_scale,
                                        self.n_class, ca_text=ca_text, class_names=CLASS_NAMES,
                                        eval_results=eval_results, correct_answers=correct_answers)

        return kv_results, debug_im  # , all_true


    def run_test(self, list_inf, out_dir, label_dir=None, img_dir=None):

        eval_results = [{'num_pred': 0, 'num_correct': 0, 'num_label': 0} for _ in range(self.n_class)]
        kv_results = []

        for file_path in list_inf:
            base_path, ext = os.path.splitext(file_path)
            basename = os.path.basename(file_path).split('.')[0]

            if img_dir is None:
                img = cv2.imread(base_path + ".jpg")
            else:
                img = cv2.imread(os.path.join(img_dir, basename + '.jpg'))

            if img is None:
                continue

            if label_dir is not None:
                label_path = os.path.join(label_dir, basename + '.json')
            else:
                label_path = None

            ca_text = ''
            ca_dict = None

            result, debug_im = self.predict((file_path, img), debug_info=(ca_text, ca_dict),
                                            label_path=label_path, eval_results=eval_results)
            print(basename)
            print(result)
            kv_results.append(result)

            debug_im.save(os.path.join(out_dir, basename) + ".jpg")

        if label_dir is not None:
            for c, count in enumerate(eval_results):
                if count['num_pred'] > 0 or count['num_label'] > 0:
                    print(c, count)

            recall = 1.0 * np.sum([c['num_correct'] for c in eval_results]) / np.sum(
                [c['num_label'] for c in eval_results])
            precision = 1.0 * np.sum([c['num_correct'] for c in eval_results]) / np.sum(
                [c['num_pred'] for c in eval_results])
            f1 = 2 * recall * precision / (recall + precision)

            print('Precision : {}   Recall : {}    F1-score : {}'.format(precision, recall, f1))

        return kv_results
