import torch
from queue import Queue
import threading
import os
from random import uniform
from random import shuffle
import numpy as np
from scipy import ndimage
import json
import random

from utils.path_util import read_image_list
from utils.image_util import affine_transform, elastic_transform, \
    to_categorical
from utils.generic_util import decision


class DataGenerator(object):
    def __init__(self, path_list_train, path_list_val, n_classes,
                 threadNum=4, queueCapacity=64, path_charset=None,
                 prefix=None, kwargs_dat={}):

        """Define data generator parameters / flags

        Args:
          path_list_train: Path to training file list
          path_list_val: Path to validation file list
          n_classes: Number of output classes
          threadNum: Number of concurrent threads to generate data
          queueCapacity: Maximum queue capacity
                        (reduce this if you have out-of-memory issue)
          path_charset: Path to character set file
          prefix: [Optional] Prefix path to each file in the dataset
                        (in case of relative path)
          kwargs_dat: Additional keyword arguments for data augmentation
                        (geometric transforms, text error rate, ...)

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
        self.text_err_train = kwargs_dat.get("text_err_train", 0.1)
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
                self.charset = ' ' + '$' + f.read()
                self.blank_idx = 1
                self.tok_to_id = {tok: idx
                                  for idx, tok in enumerate(self.charset)}
                self.id_to_tok = {idx: tok
                                  for tok, idx in self.tok_to_id.items()}
                self.n_token = len(self.tok_to_id)  # self.n_classes #
                print('n token', self.n_token)
        else:
            self.charset = None

        # start data generator thread(s) to fill the training queue
        if path_list_train is not None:
            self.list_train = read_image_list(path_list_train, prefix=prefix)
            self.size_train = len(self.list_train)
            self.q_train, self.threads_tr = self._get_list_queue(
                self.list_train, self.threadNum, self.queueCapacity,
                self.stopTrain, self.batchsize_tr,
                self.scale_min, self.scale_max, self.affine_tr,
                self.elastic_tr, self.rotate_tr, self.rotateMod90_tr,
                self.text_err_train)

        # start data generator thread(s) to fill the validation queue
        if path_list_val is not None:
            self.list_val = read_image_list(path_list_val, prefix=prefix)
            self.size_val = len(self.list_val)
            self.q_val, self.threads_val = self._get_list_queue(
                self.list_val, 2, 64, self.stopVal, self.batchsize_val,
                self.scale_val, self.scale_val, self.affine_val,
                self.elastic_val, self.rotate_val, self.rotateMod90_val,
                self.text_err_val)

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

    #
    def _get_list_queue(self, aList, threadNum, queueCapacity,
                        stopEvent, batch_size, min_scale, max_scale, affine,
                        elastic, rotate, rotateMod90, text_err):
        """Create a queue and add dedicated generator thread(s) to fill it
        """

        q = Queue(maxsize=queueCapacity)
        threads = []
        for t in range(threadNum):
            threads.append(threading.Thread(target=self._fillQueue, args=(
                q, aList[:], stopEvent, batch_size, min_scale, max_scale,
                affine, elastic, rotate, rotateMod90, text_err)))
        for t in threads:
            t.start()
        return q, threads

    def _generate_masks_from_label(self, label_path,
                                   scale_min, scale_max, text_err=0.0):
        """Read text-line information from input JSON and
        generate the character grid

        Args:
          label_path: Path to input JSON
          scale_min: Minimum text height (in pixel)
          scale_max: Maximum text height (in pixel)
          text_err: Text error rate
                    (chance of a character being randomly replaced)

        Returns:
          3-element tuple: input character grid, output mask for key
                           (auxiliary output),
                           output mask for key & value (final output)
        """

        with open(label_path, 'r') as f:
            json_dict = json.load(f)

        label_lines = json_dict['lines']
        if len(label_lines) == 0:
            print(label_path)
        line_heights = [l['box'][3] - l['box'][1] for l in label_lines]
        min_x, min_y, max_x, max_y = min([l['box'][0] for l in label_lines]), min([l['box'][1] for l in label_lines]), \
                                     max([l['box'][2] for l in label_lines]), max([l['box'][3] for l in label_lines])

        median_h = np.median(line_heights)

        if scale_min != scale_max:
            v_scale = uniform(0.8, 1.2)
            h_scale = uniform(0.8, 1.2)
            random_pad = int(uniform(median_h, median_h * 5))
        else:
            v_scale = 1.0
            h_scale = 1.0
            random_pad = median_h * 3

        min_x, min_y = min_x - random_pad, min_y - random_pad
        max_x, max_y = max_x + random_pad, max_y + random_pad

        scale = uniform(scale_min, scale_max) / median_h

        w, h = max_x - min_x, max_y - min_y

        im_shape = [int(h * scale * v_scale), int(w * scale * h_scale)]

        input_mask = np.zeros(list(im_shape), dtype='uint16')
        output_mask = np.zeros(list(im_shape), dtype='uint16')
        output_mask_type = np.zeros(list(im_shape), dtype='uint16')

        for line in label_lines:
            box, text, type_idx, value_idx = (line[k] for k in ['box', 'text', 'type', 'value'])
            # box, text, value_idx = (line[k] for k in ['box', 'text', 'value'])

            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y

            # bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
            # x1, y1 = int(x1 * scale * h_scale), int(y1 * scale * v_scale)
            # x2, y2 = int(x1 + bw), int(y1 + bh)

            x1, y1, x2, y2 = int(x1 * scale * h_scale), int(y1 * scale * v_scale), int(x2 * scale * h_scale), int(
                y2 * scale * v_scale)

            # text = ''.join([c if not c.isdigit() else '0' for c in text])

            if len(text) > 0:
                # generate output masks
                output_mask[y1:y2, x1:x2] = value_idx + 1 if value_idx > 0 else 0 #and value_idx > 0 else 0
                output_mask_type[y1:y2, x1:x2] = value_idx + 1 if type_idx == 1 else (1 if type_idx == 2 else 0)

                char_full_w = max(1.0 * (x2 - x1) / len(text), 1.0)
                char_w = max(0.9 * char_full_w, 1.0)
                # if decision(0.5):
                char_w = min(char_w, int((y2 - y1) * 1.2))

                for idx, c in enumerate(text):
                    text_err_t = 2 * text_err if type_idx == 2 else text_err
                    if decision(text_err_t):
                        c = random.choice(range(self.n_token))
                    offset = x1 + idx * char_full_w
                    char_id = self.tok_to_id[c] if c in self.tok_to_id else self.blank_idx
                    input_mask[y1:y2, int(offset): int(offset + char_w)] = char_id

        input_mask = to_categorical(input_mask, self.n_token)
        output_mask = to_categorical(output_mask, self.n_classes)
        output_mask_type = to_categorical(output_mask_type, self.n_classes)

        return input_mask * 255, output_mask * 255, output_mask_type * 255

    def _fillQueue(self, q, aList, stopEvent, batch_size, min_scale, max_scale,
                   affine, elastic, rotate, rotateMod90,
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
                imgs = []
                tgts = []
                maxH = 0
                maxW = 0

                while len(imgs) < batch_size:
                    if aIdx == len(aList):
                        if self.shuffle:
                            shuffle(aList)
                        aIdx = 0
                    try:
                        path = aList[aIdx]
                    except IndexError:
                        print(aIdx, len(aList))
                    aScale = uniform(min_scale, max_scale)
                    imgChannels = self.n_token
                    tgtChannels = self.n_classes

                    filename, file_extension = os.path.splitext(path)
                    input_mask, output_mask, output_mask_aux =\
                        self._generate_masks_from_label(
                            path, min_scale, max_scale, text_err
                        )
                    maps = np.concatenate(
                        (input_mask, output_mask, output_mask_aux), axis=2)
                    res = maps

                    if affine:
                        res = affine_transform(res, self.affine_value)
                    if elastic:
                        res = elastic_transform(
                            res, self.elastic_value_x, self.elastic_value_y
                        )
                    if rotate or rotateMod90:
                        angle = uniform(-20, 20)
                        if rotateMod90:
                            if angle < 0:
                                angle = -45.0
                            elif angle < 45:
                                angle = -45.0
                            elif angle < 90.0:
                                angle = 45.0
                            else:
                                angle = 90.0
                        res = ndimage.interpolation.rotate(res, angle)

                    aImg = res[:, :, 0:imgChannels]
                    aTgt = res[:, :, imgChannels:imgChannels + tgtChannels]
                    aTgtAux = res[:, :, imgChannels + tgtChannels:]

                    aImg = np.where(aImg > 64, 1.0, 0.0)
                    aTgt = np.where(aTgt > 64, 1.0, 0.0)
                    aTgtAux = np.where(aTgtAux > 64, 1.0, 0.0)

                    if self.one_hot_encoding:
                        for tTgt in [aTgt, aTgtAux]:
                            aMap = tTgt[:, :, self.dominating_channel]
                            for aM in range(1, tgtChannels):
                                if aM == self.dominating_channel:
                                    continue
                                else:
                                    tMap = np.logical_and(tTgt[:, :, aM], np.logical_not(aMap))
                                    aMap = np.logical_or(aMap, tMap)
                                    tTgt[:, :, aM] = tMap
                            tTgt[:, :, 0] = np.logical_not(aMap)

                    imgs.append(aImg)
                    width = aImg.shape[1]
                    height = aImg.shape[0]
                    maxW = max(width, maxW)
                    maxH = max(height, maxH)
                    tgts.append(aTgt)

                curPair = [
                    np.expand_dims(aImg, 0), np.expand_dims(aTgt, 0),
                    np.expand_dims(aTgtAux, 0)]
                curPair = [
                    torch.transpose(torch.transpose(torch.from_numpy(curPair[0]), 1, -1), 2, 3),
                    torch.transpose(torch.transpose(torch.from_numpy(curPair[1]), 1, -1), 2, 3),
                    torch.transpose(torch.transpose(torch.from_numpy(curPair[2]), 1, -1), 2, 3)
                ]

            try:
                q.put(curPair, timeout=1)
                curPair = None
                aIdx += 1
            except:
                continue

    def stop_all(self):
        """Stop all data generator threads
        """
        self.stopTrain.set()
        self.stopVal.set()

    def restart_val_runner(self):
        """Restart validation runner
        """
        if self.list_val is not None:
            self.stopVal.set()
            self.stopVal = threading.Event()
            self.q_val, self.threads_val = self._get_list_queue(
                self.list_val, 1, 100, self.stopVal,
                self.batchsize_val,
                self.scale_val, self.scale_val, self.affine_val,
                self.elastic_val, self.rotate_val,
                self.rotateMod90_val,
                self.text_err_val)
