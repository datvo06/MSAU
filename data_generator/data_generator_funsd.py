from queue import Queue
import threading
from random import uniform
from random import shuffle
import numpy as np
import math
from scipy import ndimage
import json
import random
import time
import glob

from utils.path_util import read_image_list
from utils.image_util import affine_transform, elastic_transform, to_categorical, \
    gaussian_radius, draw_gaussian, draw_line
from utils.generic_util import decision
from inference.morph_util import scale_rect


class DataGenerator(object):
    def __init__(self, path_list_train, path_list_val, n_classes, threadNum=2, queueCapacity=128, path_charset=None,
                 prefix=None, kwargs_dat={}):

        """Define data generator parameters / flags

        Args:
          path_list_train: Path to training file list
          path_list_val: Path to validation file list
          n_classes: Number of output classes
          threadNum: Number of concurrent threads to generate data
          queueCapacity: Maximum queue capacity (reduce this if you have out-of-memory issue)
          path_charset: Path to character set file
          prefix: [Optional] Prefix path to each file in the dataset (in case of relative path)
          kwargs_dat: Additional keyword arguments for data augmentation (geometric transforms, text error rate, ...)

        Returns:
          a DataGenerator instance
        """

        self.n_classes = n_classes
        self.list_train = None
        self.size_train = 0
        self.q_train = None
        self.list_val = None
        self.size_val = 0
        self.q_val = None

        """
        Augmentation parameters
        """

        # batch size for trainging & validation
        self.batchsize_tr = kwargs_dat.get("batchsize_tr", 1)
        self.batchsize_val = kwargs_dat.get("batchsize_val", 1)

        # use affine transformation
        self.affine_tr = kwargs_dat.get("affine_tr", False)
        self.affine_val = kwargs_dat.get("affine_val", False)
        self.affine_value = kwargs_dat.get("affine_value", 0.025)

        # use elastic transformation
        self.elastic_tr = kwargs_dat.get("elastic_tr", False)
        self.elastic_val = kwargs_dat.get("elastic_val", False)
        self.elastic_value_x = kwargs_dat.get("elastic_val_x", 0.0002)
        self.elastic_value_y = kwargs_dat.get("elastic_value_y", 0.0002)

        # use random rotation
        self.rotate_tr = kwargs_dat.get("rotate_tr", False)
        self.rotate_val = kwargs_dat.get("rotate_val", False)
        self.rotateMod90_tr = kwargs_dat.get("rotateMod90_tr", False)
        self.rotateMod90_val = kwargs_dat.get("rotateMod90_val", False)

        # use random text errors
        self.text_err_train = kwargs_dat.get("text_err_train", 0.0)
        self.text_err_val = kwargs_dat.get("text_err_val", 0.0)

        # use random text scale
        self.scale_min = kwargs_dat.get("scale_min", 2.0)
        self.scale_max = kwargs_dat.get("scale_max", 4.0)
        self.scale_val = kwargs_dat.get("scale_val", 3.0)

        # ensure one-hot encoding consistency after geometric distortions
        self.one_hot_encoding = kwargs_dat.get("one_hot_encoding", True)
        self.dominating_channel = kwargs_dat.get("dominating_channel", 1)

        # shuffle dataset after each epoch
        self.shuffle = kwargs_dat.get("shuffle", True)

        self.threadNum = threadNum
        self.queueCapacity = queueCapacity
        self.stopTrain = threading.Event()
        self.stopVal = threading.Event()

        # load character set from file
        if path_charset is not None:
            with open(path_charset, 'r') as f:
                self.charset = '◫' + '⎅' + f.read()
                self.blank_idx = 1
                self.tok_to_id = {tok: idx for idx, tok in enumerate(self.charset)}
                self.id_to_tok = {idx: tok for tok, idx in self.tok_to_id.items()}
                self.n_token = len(self.tok_to_id)
                print('n token', self.n_token)
        else:
            self.charset = None

        self.label_to_id = {
            'question': 1,
            'answer': 2,
            'header': 3,
            'other': 0
        }
        self.id_to_label = {idx: tok for tok, idx in self.label_to_id.items()}

        # start data generator thread(s) to fill the training queue
        if path_list_train != None:
            self.list_train = glob.glob(path_list_train + '/*.json')
            self.size_train = len(self.list_train)


        # start data generator thread(s) to fill the validation queue
        if path_list_val != None:
            self.list_val = glob.glob(path_list_val + '/*.json')
            self.size_val = len(self.list_val)

        self.q_train, self.threads_tr = self._get_list_queue(self.list_train, self.threadNum, self.queueCapacity,
                                                             self.stopTrain, self.batchsize_tr,
                                                             self.scale_min, self.scale_max, self.affine_tr,
                                                             self.elastic_tr, self.rotate_tr, self.rotateMod90_tr,
                                                             self.text_err_train)

        self.restart_val_runner()

    def next_data(self, list):
        """Return next data from the queues
        """

        if list is 'val':
            q = self.q_val
        else:
            q = self.q_train
        if q is None:
            return None, None
        return q.get()


    def generate_charset(self, output_path):
        all_text = []
        for file_path in self.list_val + self.list_train:
            print('Reading ', file_path)
            with open(file_path, 'r') as f:
                json_dict = json.load(f)
                for item in json_dict['form']:
                    all_text += list(item['text'].replace('\n', ''))

        all_text = sorted(set(all_text))
        with open(output_path, 'w') as f:
            for c in all_text:
                f.write(c)


    def _get_list_queue(self, aList, threadNum, queueCapacity, stopEvent, batch_size, min_scale, max_scale, affine,
                        elastic, rotate, rotateMod90, text_err):
        """Create a queue and add dedicated generator thread(s) to fill it
        """

        q = Queue(maxsize=queueCapacity)
        threads = []
        for t in range(threadNum):
            threads.append(threading.Thread(target=self._fillQueue, args=(
                q, aList[:], stopEvent, batch_size, min_scale, max_scale, affine, elastic, rotate,
                rotateMod90, text_err)))
        for t in threads:
            t.start()
        return q, threads


    def _generate_bounding_box_corner_point(self, bboxes, edges, origin_shape, output_shape, n_class, use_gaussian_bump=True, gaussian_rad=1, gaussian_iou=0.7):

        # number of the available corner points
        max_tag_len = len(edges)
        n_box_class = 3

        heatmaps_tl = np.zeros((output_shape[0], output_shape[1], n_box_class), dtype=np.float32)
        heatmaps_br = np.zeros((output_shape[0], output_shape[1], n_box_class), dtype=np.float32)
        heatmaps_center = np.zeros((output_shape[0], output_shape[1], n_box_class), dtype=np.float32)
        vertices = np.zeros((max_tag_len, n_class), dtype=np.float32)
        offsets_tl = np.zeros((max_tag_len, 2), dtype=np.float32)
        offsets_br = np.zeros((max_tag_len, 2), dtype=np.float32)
        tags_tl = np.zeros((max_tag_len), dtype=np.int64)
        tags_br = np.zeros((max_tag_len), dtype=np.int64)
        tags_mask = np.zeros((max_tag_len), dtype=np.float32)
        boxes = np.zeros((max_tag_len, 4), dtype=np.int64)
        ratio = np.ones((max_tag_len, 2), dtype=np.float32)
        tag_lens = 0

        width_ratio = output_shape[1] / origin_shape[1]
        height_ratio = output_shape[0] / origin_shape[0]

        bboxes_list = bboxes.values()
        converted_bboxes = {}

        for ind, bb_item in enumerate(bboxes_list):
            category = bb_item[1]
            box = bb_item[0]
            link = bb_item=[-1]

            xtl_ori, ytl_ori = box[0], box[1]
            xbr_ori, ybr_ori = box[2], box[3]

            fxtl = (xtl_ori * width_ratio)
            fytl = (ytl_ori * height_ratio)
            fxbr = (xbr_ori * width_ratio)
            fybr = (ybr_ori * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            xc = (xtl + xbr) // 2
            yc = (ytl + ybr) // 2

            converted_bboxes[ind] = [xtl, ytl, xbr, ybr]

            if category > 0:
                category -= 1
                if use_gaussian_bump:
                    width = box[2] - box[0]
                    height = box[3] - box[1]

                    width = math.ceil(width * width_ratio)
                    height = math.ceil(height * height_ratio)

                    if gaussian_rad == -1:
                        radius = gaussian_radius((height, width), gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = gaussian_rad

                    draw_gaussian(heatmaps_tl[:, :, category], [xtl, ytl], radius)
                    draw_gaussian(heatmaps_br[:, :, category], [xbr, ybr], radius)
                    draw_gaussian(heatmaps_center[:, :, category], [xc, yc], radius)

                else:
                    heatmaps_tl[ytl, xtl, category] = 1
                    heatmaps_br[ybr, xbr, category] = 1
                    heatmaps_center[yc, xc, category] = 1

        for id, bb_item in bboxes.items():
            link = bb_item[-1]
            # if len(link) == 0 and decision(0.2):
            #     tag_ind = tag_lens
            #     xbr, ybr = converted_bboxes[id][:2]
            #
            #     max_pos = output_shape[1] * output_shape[0] - 1
            #     tags_tl[tag_ind] = min(xbr * output_shape[1] + ybr, max_pos)
            #     tags_br[tag_ind] = min(xbr * output_shape[1] + ybr, max_pos)

            for item in link:
                target_id = item[1]
                if target_id == id: continue

                tag_ind = tag_lens
                xbr, ybr = converted_bboxes[id][:2]
                xtl, ytl = converted_bboxes[target_id][:2]
                tag_lens += 1

                shift_vector = [xbr - xtl, ybr - ytl]
                if shift_vector[0] > shift_vector[1]:
                    offsets_tl[tag_ind, :] = shift_vector
                    offsets_br[tag_ind, :] = 0
                else:
                    offsets_tl[tag_ind, :] = 0
                    offsets_br[tag_ind, :] = shift_vector

                # offsets_br[tag_ind, :] = [xtl - xbr, ytl - ybr]

                ## limit tag index to max shape
                max_pos = output_shape[1] * output_shape[0] - 1
                tags_tl[tag_ind] = min(ytl * output_shape[1] + xtl, max_pos)
                tags_br[tag_ind] = min(ytl * output_shape[1] + xtl, max_pos) #min(ybr * output_shape[1] + xbr, max_pos)

                # tags_mask[tag_ind] = category
                # boxes[tag_ind] = [xtl_ori, ytl_ori, xbr_ori, ybr_ori]
                ratio[tag_ind] = [width_ratio, height_ratio]
                tag_lens += 1

        tags_mask[:tag_lens] = 1

        return tags_tl, tags_br, heatmaps_tl, heatmaps_br, \
               heatmaps_center, tags_mask, offsets_tl, offsets_br, boxes, ratio, vertices


    def _generate_masks_from_label(self, label_path, scale_min, scale_max, text_err=0.0):
        """Read text-line information from input JSON and generate the character grid

        Args:
          label_path: Path to input JSON
          scale_min: Minimum text height (in pixel)
          scale_max: Maximum text height (in pixel)
          text_err: Text error rate (chance of a character being randomly replaced)

        Returns:
          3-element tuple: input character grid, output mask for key (auxiliary output),
                            output mask for key & value (final output)
        """

        with open(label_path, 'r') as f:
            json_dict = json.load(f)

        label_lines = json_dict['form']
        if len(label_lines) == 0: print(label_path)
        line_heights = [l['box'][3] - l['box'][1] for l in label_lines]
        min_x, min_y, max_x, max_y = min([l['box'][0] for l in label_lines]), min([l['box'][1] for l in label_lines]), \
                                     max([l['box'][2] for l in label_lines]), max([l['box'][3] for l in label_lines])

        median_h = np.median(line_heights)

        if scale_min != scale_max:
            v_scale = uniform(0.8, 1.2)
            h_scale = uniform(0.9, 1.1)
            random_pad = int(uniform(median_h, median_h * 3))
        else:
            v_scale = 1.0
            h_scale = 1.0
            random_pad = median_h * 2

        min_x, min_y = min_x - random_pad, min_y - random_pad
        max_x, max_y = max_x + random_pad, max_y + random_pad

        scale = uniform(scale_min, scale_max) / median_h

        w, h = max_x - min_x, max_y - min_y

        im_shape = [int(h * scale * v_scale), int(w * scale * h_scale)]

        input_mask = np.zeros(list(im_shape), dtype='uint16')
        output_mask = np.zeros(list(im_shape), dtype='uint16')
        # output_mask_aux = np.zeros(list(im_shape), dtype='uint16')
        output_mask_type = np.zeros(list(im_shape), dtype='uint16')

        line_mask = np.zeros(list(im_shape), dtype='B')
        char_sep_mask = np.zeros(list(im_shape), dtype='B')

        bboxes = dict()
        edges = []

        for line in label_lines:
            id, box, text, label, link = (line[k] for k in ['id', 'box', 'text', 'label', 'linking'])
            label_id = self.label_to_id[label]

            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y

            x1, y1, x2, y2 = int(x1 * scale * h_scale), int(y1 * scale * v_scale), int(x2 * scale * h_scale), int(
                y2 * scale * v_scale)

            text_BoW = np.zeros(self.n_token)
            bboxes[id] = ([(x1, y2, x2, y2),
                           label_id,
                           text,
                           text_BoW,
                           link])

            edges += link

            if len(text) > 0:
                # generate output masks
                output_mask[y2-1:y2, x1:x2] = label_id
                # output_mask_aux[y2-1:y2, x1:x2] = self.label_to_id[label]
                # output_mask_type[y2-1:y2, x1:x2] = type_idx
                line_mask[y2:y2+1, x1:x2] = 1

                char_full_w = max(1.0 * (x2 - x1) / len(text), 1.0)
                char_w = max(0.9 * char_full_w, 1.0)
                char_w = min(char_w, int((y2 - y1) * 1.0))

                for idx, c in enumerate(text):
                    text_err_t = text_err
                    if text_err > 0 and (decision(text_err_t)):
                        c = random.choice(range(self.n_token))
                    offset = x1 + idx * char_full_w
                    char_id = self.tok_to_id[c] if c in self.tok_to_id else self.blank_idx

                    start_x, end_x = int(offset), int(offset + char_w)
                    input_mask[y1:y2, start_x:end_x] = char_id
                    char_sep_mask[y1:y2, end_x-1:end_x] = char_id

        input_mask = to_categorical(input_mask, self.n_token)
        input_mask = np.dstack([input_mask, line_mask, char_sep_mask])

        output_mask = to_categorical(output_mask, self.n_classes)
        output_mask_aux = output_mask
        output_mask_type = to_categorical(output_mask_type, 3)

        return input_mask, output_mask, output_mask_aux, output_mask_type, bboxes, edges


    def _fillQueue(self, q, aList, stopEvent, batch_size, min_scale, max_scale, affine, elastic, rotate, rotateMod90,
                   text_err):
        """Main function to generate new input-output pair an put it into the queue

        Args:
          q: Output queue
          aList: List of input JSON(s)
          stopEvent: Thread stop event
          batch_size: Batch-size
          min_scale: Minimum text height
          max_scale: Maximum text height
          affine: Use affine transform
          elastic: Use elastic transform
          rotate: Use random rotation
          rotateMod90: Use random rotation (constrained to multiple of 90 degree)

        Returns:
          None
        """

        if self.shuffle:
            shuffle(aList)
        aIdx = 0
        curPair = None

        while (not stopEvent.is_set()):
            if curPair is None:

                # start_t = time.time()
                if aIdx == len(aList):
                    if self.shuffle:
                        shuffle(aList)
                    aIdx = 0
                try:
                    path = aList[aIdx]
                except IndexError:
                    print(aIdx, len(aList))
                    continue

                input_mask, output_mask, output_mask_aux, output_mask_type, bboxes, edges = self._generate_masks_from_label(path, min_scale, max_scale, text_err)
                origin_shape = input_mask.shape
                bbox_out_shape = [math.ceil(k / 1) for k in origin_shape]
                tags_tl, tags_br, heatmaps_tl, heatmaps_br, heatmaps_center, tags_mask, \
                    offsets_tl, offsets_br, _, _, vertices = self._generate_bounding_box_corner_point(bboxes, edges, origin_shape, bbox_out_shape, self.n_token + 4)

                aImg, aTgt, aTgtAux, aTgtType = input_mask, output_mask, output_mask_aux, output_mask_type

                curPair = [np.expand_dims(aImg, 0), np.expand_dims(aTgt, 0),
                           np.expand_dims(aTgtAux, 0), np.expand_dims(aTgtType, 0),
                           np.expand_dims(tags_tl, 0), np.expand_dims(tags_br, 0),
                           np.expand_dims(heatmaps_tl, 0), np.expand_dims(heatmaps_br, 0),
                           np.expand_dims(heatmaps_center, 0),
                           np.expand_dims(tags_mask, 0),
                           np.expand_dims(offsets_tl, 0), np.expand_dims(offsets_br, 0),
                           np.expand_dims(vertices, 0),
                           [(scale_rect(r[0], 0.5), r[1]) for r in bboxes.values()]]

                # end_time = time.time() - start_t
                # print('end generate', end_time)

            try:
                q.put(curPair, timeout=10)
                curPair = None
                aIdx += 1
            except Exception as e:
                continue


    def stop_all(self):
        """Stop all data generator threads
        """
        self.stopTrain.set()
        self.stopVal.set()

    def restart_val_runner(self):
        """Restart validation runner
        """
        if self.list_val != None:
            self.stopVal.set()
            self.stopVal = threading.Event()
            self.q_val, self.threads_val = self._get_list_queue(self.list_val, self.threadNum // 2, self.queueCapacity // 4, self.stopVal,
                                                                self.batchsize_val,
                                                                self.scale_val, self.scale_val, self.affine_val,
                                                                self.elastic_val, self.rotate_val,
                                                                self.rotateMod90_val,
                                                                self.text_err_val)
