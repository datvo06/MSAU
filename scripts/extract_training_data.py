import argparse
import codecs
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import Counter

from scripts.data_util import *

import cv2

FONT_PATH = './resources/Dengb.ttf'
CLASS_LIST = ['sample_class_name_1', 'sample_class_name_2', ]

class DataExtractor:
    def __init__(self, output_dir, debug=False):
        self.label_dir = ''
        self.image_dir = ''
        self.predict_dir = ''

        self.output_dir = output_dir

        self.debug_dir = os.path.join(self.output_dir, 'debug_ims')

        # for path in [output_dir, self.debug_dir]:
        #     if not os.path.isdir(path):
        #         os.mkdir(path)

        self._debug = debug
        self._all_chars = []
        self.key_set = []
        self.class_names = ['nul'] * (2 * len(CLASS_LIST))
        for id, key in enumerate(CLASS_LIST):
            self.class_names[2 * id] = 'k_' + key
            self.class_names[2 * id + 1] = 'v_' + key

        print(self.class_names)

        self.max_value_id = 0

    def process(self, label_dir, image_dir):
        image_map = glob_folder(image_dir, 'jpg')
        print('Got {} images'.format(len(image_map)))
        fail_count = 0
        label_map = glob_folder(label_dir, 'json', use_dirname=False)
        count = {}
        label_list = list(label_map.items())
        N = len(label_list)

        for idx, (label_name, label_path) in enumerate(label_list):
            if label_name not in image_map:
                print('Matching image not found in data folder. Skipped ({})'.format(label_name))
                fail_count += 1
                continue
            image_file = image_map[label_name]
            print('------------------------------------')
            print('{}/{} Handle {} label'.format(idx + 1, N, label_path))
            print(image_file)
            path_name = os.path.dirname(image_file)
            if path_name in count:
                count[path_name] += 1
            else:
                count[path_name] = 1
            if not os.path.exists(image_file):
                print("Do not have image with same name for label {}".format(label_name))
                continue

            with codecs.open(label_path, 'r', 'utf-8-sig') as f:
                content = json.load(f)

            if '_via_img_metadata' in content:
                content = content["_via_img_metadata"]

            data = content[list(content.keys())[0]]
            self._parse(data, image_file)

        ## export most frequent chars
        top_char_count = 300
        char_count = Counter(self._all_chars)
        if ' ' in char_count:
            char_count.pop(' ')
        charset = [c[0] for c in char_count.most_common(top_char_count)]
        with open(os.path.join(self.debug_dir, 'charset.txt'), 'w') as f:
            for c in sorted(charset):
                f.write(c)

        print('Got {} label files. Failed {} / {}'.format(len(label_map), fail_count, len(label_map)))
        print('Total {} classes: {}'.format(len(self.key_set) + 1, self.key_set))
        print('Max value id:', self.max_value_id)
        print('Class names:', self.class_names)

    def _is_latin_label(self, label):
        alpha_count = len([c for c in label if c.isalpha()])
        is_latin = alpha_count > 1 and (alpha_count / len(label)) > 0.3
        return is_latin

    def _parse(self, data, img_path):
        filename = os.path.basename(img_path).split('.')[0]
        regions = data['regions']

        img = cv2.imread(img_path)
        clone_img = Image.fromarray(img)
        draw = ImageDraw.Draw(clone_img)
        font = ImageFont.truetype(
            FONT_PATH,
            size=28,
            encoding='utf-8-sig')

        label_lines = []

        for rg in regions:
            region_attr = rg['region_attributes']
            shape_attr = rg['shape_attributes']

            try:
                if shape_attr['name'] == 'polygon':
                    x1, y1, x2, y2 = min(shape_attr['all_points_x']), min(shape_attr['all_points_y']), max(
                        shape_attr['all_points_x']), max(shape_attr['all_points_y'])
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                else:
                    x, y, w, h = shape_attr['x'], shape_attr['y'], shape_attr['width'], shape_attr['height']

            except KeyError as e:
                print(e, shape_attr)
                continue

            box = [x, y, x + w, y + h]

            if 'label' in region_attr:
                text = region_attr['label']
            else:
                text = ''

            # text = text.replace('\n', '')
            text = ''.join(['0' if c.isdigit() else c for c in text])

            if 'type' in region_attr:
                type = region_attr['type']
            else:
                type = ''
            type = type.replace(' ', '_')

            if 'formal_key' in region_attr:
                matching_key = region_attr['formal_key']
            else:
                matching_key = ''

            matching_key = matching_key.replace(' ', '').replace('\n', '').replace('\u3000', '').replace('__', '_')
            # matching_key = '_'.join(matching_key.split('_')[:2])

            if matching_key not in CLASS_LIST:
                matching_key = ''
                type = 'other'

            if type in ['key', 'value'] and len(matching_key) > 0:
                if matching_key not in self.key_set:
                    self.key_set.append(matching_key)

                key_idx = self.key_set.index(matching_key)  #CLASS_LIST.index(matching_key)
                if type == 'key':
                    value_idx = 2 * key_idx + 1
                else:
                    value_idx = 2 * key_idx + 2

                if type == 'key':
                    print(matching_key, text)

            else:
                value_idx = 0

            if value_idx > self.max_value_id:
                self.max_value_id = value_idx

            type_idx = {'other': 0, 'key': 1, 'value': 2, 'common_key': 0, 'master': 0, 'master_key': 0}[type]

            label_lines.append((box, text, type_idx, value_idx, matching_key))

            if type in ['key', 'value'] and len(matching_key) > 0:
                self._all_chars += list(text) * 10
            else:
                self._all_chars += list(text)

        for box, text, type_idx, value_idx, matching_key in label_lines:
            x1, y1, x2, y2 = box
            color = "green" if type_idx == 1 else "red"
            if len(text) > 0:
                if value_idx > 0:
                    draw_rectangle(draw, ((x1, y1), (x2, y2)), color=color, width=3)
                    draw.text((x2 + 5, y1 + 10), "{}({})".format(matching_key, value_idx), fill=color, font=font)
                elif type_idx > 0:
                    draw_rectangle(draw, ((x1, y1), (x2, y2)), color='blue', width=3)

        im_shape = img.shape[:2]
        export_label_json(os.path.join(self.output_dir, filename + '.json'),
                          [(b, o, t, v) for b, o, t, v, _ in label_lines], im_shape)

        if self._debug:
            linecut_debug_path = os.path.join(self.debug_dir, filename + '.jpg')
            clone_img.save(linecut_debug_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir',
                        default='path_to_label_dir')
    parser.add_argument('--image_dir',
                        default='path_to_image_dir')
    parser.add_argument('--save_dir',
                        default='path_to_output_dir')
    args = parser.parse_args()

    extractor = DataExtractor(args.save_dir, True)
    extractor.process(args.label_dir, args.image_dir)