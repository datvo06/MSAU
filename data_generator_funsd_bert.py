from __future__ import division, print_function
from queue import Queue
import pickle
import threading
from random import uniform
from random import shuffle
import numpy as np
import math
import torch
from scipy import ndimage
import json
import random
import time
import glob

from utils.path_util import read_image_list
from utils.generic_util import decision
from inference.morph_util import scale_rect
from torch.utils.data import Dataset


def getitem_box_bert(dataset_instance, idx):
    return {
        "ocr_values": [cell.ocr_value
                       for cell in dataset_instance.inp_list[idx]['cells']],
        "feats": dataset_instance.inp_list[idx]['transformer_feature'],
        # "feats": torch.Tensor(self.inp_list[idx]['pos_feats']).unsqueeze(0),
        "label": dataset_instance.inp_list[idx]['labels']
    }


def getitem_box_bow(dataset_instance, idx):
    return {
        "ocr_values": [cell.ocr_value
                       for cell in dataset_instance.inp_list[idx]['cells']],
        "feats": dataset_instance.inp_list[idx]['bow_feature'].toarray(),
        # "feats": torch.Tensor(self.inp_list[idx]['pos_feats']).unsqueeze(0),
        "label": dataset_instance.inp_list[idx]['labels']
    }


def get_min_max_x_y_w_h(cell_lists):
    heights = [cell.h for cell in cell_lists]
    widths = [cell.w for cell in cell_lists]
    ys_top = [cell.y+cell.h for cell in cell_lists]
    xs_right = [cell.x+cell.w for cell in cell_lists]
    min_h = min(heights)
    min_w = min(widths)
    max_y = max(ys_top)
    max_x = max(xs_right)

    min_x = min([cell.x for cell in cell_lists])
    min_y = min([cell.y for cell in cell_lists])
    return min_x, min_y, max_x, max_y, min_w, min_h


def get_box_mask_box_label(dataset_instance, idx):
    cell_lists = dataset_instance.inp_list[idx]['cells']
    box_dict = dataset_instance.getitem_box(dataset_instance, idx)
    # Convert into torch/msau mask
    # first, find the shortest cell, horizontally
    # then, find the shortest cell, vertically
    # use them as scale down ratio of the bboxs
    # fill the bboxs with the label
    min_x, min_y, max_x, max_y, min_w, min_h = get_min_max_x_y_w_h(cell_lists)
    new_max_x = int((max_x-min_x)/min_w)+1
    new_max_y = int((max_y-min_y)/min_h)+1

    label_mask = np.zeros((new_max_y, new_max_x)).astype('uint8')
    inp_grid = np.zeros((box_dict['feats'].shape[-1],
                        new_max_y, new_max_x))
    for cell_idx, cell in enumerate(cell_lists):
        new_x = int((cell.x - min_x)/min_w)
        new_y = int((cell.y - min_y)/min_h)
        new_w = max(int(cell.w/min_w), 1)
        new_h = max(int(cell.h/min_h), 1)
        inp_grid[:, new_y:new_y+new_h, new_x:new_x+new_w] =\
            np.expand_dims(
                np.expand_dims(box_dict['feats'][cell_idx], -1), -1)
        label_mask[new_y:new_y+new_h, new_x:new_x+new_w] =\
            box_dict['label'][cell_idx] + 1
    return {
        "ocr_values": box_dict['ocr_values'],
        "mask": inp_grid,
        "label": label_mask,
    }


def get_box_mask_1_pixel_label(dataset_instance, idx):
    cell_lists = dataset_instance.inp_list[idx]['cells']
    box_dict = dataset_instance.getitem_box(dataset_instance, idx)
    # Convert into torch/msau mask
    # first, find the shortest cell, horizontally
    # then, find the shortest cell, vertically
    # use them as scale down ratio of the bboxs
    # fill the bboxs with the label
    min_x, min_y, max_x, max_y, min_w, min_h = get_min_max_x_y_w_h(cell_lists)
    new_max_x = int((max_x-min_x)/min_w)+1
    new_max_y = int((max_y-min_y)/min_h)+1

    label_mask = np.zeros((new_max_y, new_max_x)).astype('uint8')
    inp_grid = np.zeros((box_dict['feats'].shape[-1],
                        new_max_y, new_max_x))
    for cell_idx, cell in enumerate(cell_lists):
        new_x = int((cell.x - min_x)/min_w)
        new_y = int((cell.y - min_y)/min_h)
        new_w = max(int(cell.w/min_w), 1)
        new_h = max(int(cell.h/min_h), 1)
        inp_grid[:, new_y:new_y+new_h, new_x:new_x+new_w] =\
            np.expand_dims(
                np.expand_dims(box_dict['feats'][cell_idx], -1), -1)
        label_mask[new_y, new_x] = box_dict['label'][idx] + 1
    return {
        "ocr_values": box_dict['ocr_values'],
        "mask": inp_grid,
        "label": label_mask,
    }


def get_1px_mask_1_px_label(dataset_instance, idx):
    cell_lists = dataset_instance.inp_list[idx]['cells']
    box_dict = dataset_instance.getitem_box(dataset_instance, idx)
    min_x, min_y, max_x, max_y, min_w, min_h = get_min_max_x_y_w_h(cell_lists)
    new_max_x = int((max_x-min_x)/min_w)+1
    new_max_y = int((max_y-min_y)/min_h)+1

    label_mask = np.zeros((new_max_y, new_max_x)).astype('uint8')
    inp_grid = np.zeros((box_dict['feats'].shape[-1],
                        new_max_y, new_max_x))
    for cell_idx, cell in enumerate(cell_lists):
        new_x = int((cell.x - min_x)/min_w)
        new_y = int((cell.y - min_y)/min_h)
        inp_grid[:, new_y, new_x] = box_dict['feats'][idx]
        label_mask[new_y, new_x] = box_dict['label'][idx] + 1
    return {
        "ocr_values": box_dict['ocr_values'],
        "mask": inp_grid,
        "label": label_mask,
    }


def get_box_mask_box_label_word(dataset_instance, idx):
    instance_dict = dataset_instance.inp_list[idx]
    cell_lists = instance_dict['cells_word']
    cell_lists_textline = instance_dict['cells']
    min_x, min_y, max_x, max_y, min_w, min_h = get_min_max_x_y_w_h(cell_lists)
    new_max_x = int((max_x-min_x)/min_w)+1
    new_max_y = int((max_y-min_y)/min_h)+1
    scaling_ratios = [cell.w/len(cell.ocr_value) for cell in cell_lists]
    min_scale = min(scaling_ratios)
    label_mask = np.zeros((new_max_y, new_max_x)).astype('uint8')
    inp_grid = np.zeros((instance_dict['charset_feature'][0].shape[-1],
                        new_max_y, new_max_x))
    for cell_idx, cell in enumerate(cell_lists):
        new_x = int((cell.x - min_x)/min_scale)
        new_y = int((cell.y - min_y)/min_h)
        new_w = max(int(cell.w/min_scale), 1)
        new_h = max(int(cell.h/min_h), 1)
        per_char_width = max(int(new_w/len(cell.ocr_value)), 1)
        for j, char in enumerate(cell.ocr_value):
            inp_grid[:, new_y:new_y+new_h,
                     new_x+per_char_width*j:new_x+per_char_width*(j+1)] =\
                instance_dict['charset_feature'][idx][j]
    for cell_idx, cell in enumerate(cell_lists_textline):
        new_x = int((cell.x - min_x)/min_w)
        new_y = int((cell.y - min_y)/min_h)
        new_w = max(int(cell.w/min_w), 1)
        new_h = max(int(cell.h/min_h), 1)
        label_mask[new_y:new_y+new_h, new_x:new_x+new_w] =\
            instance_dict['label'][cell_idx] + 1

    return {
        "ocr_values": [cell.ocr_value for cell in cell_lists],
        "mask": inp_grid,
        "label": label_mask,
    }


class FUNSDMaskDataLoader(Dataset):
    '''
    Note: The None class will be automatically be added
    '''
    def __init__(self, funsd_pickle_path, labels_dict=None,
                 getitem_box=getitem_box_bert,
                 getitem_mask=get_box_mask_box_label):
        self.inp_list = pickle.load(open(funsd_pickle_path, 'rb'))
        if labels_dict is None:
            self.labels = dict(
                [label, idx]
                for idx, label in enumerate(
                    list(
                        set(self.inp_list[0]['labels']))))

            json.dump(self.labels, open('labels', 'w'))
        else:
            self.labels = labels_dict
        for each_dict in self.inp_list:
            each_dict['labels'] = np.array(
                [self.labels[label]
                 for label in each_dict['labels']])
        self.getitem_box = getitem_box
        self.getitem_mask = get_box_mask_box_label

    def __len__(self):
        return len(self.inp_list)

    def getitem(self, idx):
        item_dict = self.getitem_mask(self, idx)
        return {
            "ocr_values": item_dict['ocr_values'],
            "mask": torch.Tensor(item_dict['mask']).unsqueeze(0),
            "label": torch.Tensor(item_dict['label']).unsqueeze(0),
        }

    def __getitem__(self, idx):
        if type(idx) != int:
            list_idx = idx[:]
            return_lists = [self.getitem(idx) for idx in list_idx]
        else:
            return_lists = self.getitem(idx)
        return return_lists


class FUNSDCharGridDataLoaderBoxMaskBoxLabel(FUNSDMaskDataLoader):
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBertDataLoaderBoxMaskBoxLabel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=None,
            getitem_mask=get_box_mask_box_label_word)


class FUNSDBertDataLoaderBoxMaskBoxLabel(FUNSDMaskDataLoader):
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBertDataLoaderBoxMaskBoxLabel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=getitem_box_bert,
            getitem_mask=get_box_mask_box_label)


class FUNSDBertDataLoaderBoxMaskPxLabel(FUNSDMaskDataLoader):
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBertDataLoaderBoxMaskPxLabel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=getitem_box_bert,
            getitem_mask=get_box_mask_1_pixel_label)


class FUNSDBertDataLoader1pixel(FUNSDMaskDataLoader):
    '''
    Note: The None class will be automatically be added
    '''
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBertDataLoader1pixel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=getitem_box_bert,
            getitem_mask=get_1px_mask_1_px_label)


class FUNSDBOWDataLoaderBoxMaskBoxLabel(FUNSDMaskDataLoader):
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBOWDataLoaderBoxMaskBoxLabel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=getitem_box_bow,
            getitem_mask=get_box_mask_box_label)


class FUNSDBOWDataLoaderBoxMaskPxLabel(FUNSDMaskDataLoader):
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBOWDataLoaderBoxMaskPxLabel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=getitem_box_bow,
            getitem_mask=get_box_mask_1_pixel_label)


class FUNSDBOWDataLoader1pixel(FUNSDMaskDataLoader):
    '''
    Note: The None class will be automatically be added
    '''
    def __init__(self, funsd_pickle_path, labels_dict=None):
        super(FUNSDBOWDataLoader1pixel, self).__init__(
            funsd_pickle_path, labels_dict=labels_dict,
            getitem_box=getitem_box_bow,
            getitem_mask=get_1px_mask_1_px_label)
