import logging
from math import ceil, floor

import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.measurements import label as cc_label
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import skeletonize


def connected_components(image, thres=0):
    if thres > 0:
        binary = image > thres
    else:
        binary = image

    labels, _ = cc_label(binary)
    objects = find_objects(labels)

    return labels, objects


def width(s):
    return s[1].stop - s[1].start


def height(s):
    return s[0].stop - s[0].start


def area(s):
    return (s[1].stop - s[1].start) * (s[0].stop - s[0].start)


def min_dim(b):
    if (width(b) > height(b)):
        return height(b)
    else:
        return width(b)


def max_dim(b):
    if (width(b) > height(b)):
        return width(b)
    else:
        return height(b)


def xcenter(s):
    return np.mean([s[1].stop, s[1].start])


def ycenter(s):
    return np.mean([s[0].stop, s[0].start])


def aspect_normalized(s):
    asp = height(s) * 1.0 / width(s)
    if asp < 1: asp = 1.0 / asp
    return asp


def r_dilation(image, size, origin=0):
    """Dilation with rectangular structuring element using maximum_filter"""
    return maximum_filter(image, size, origin=origin, mode='constant')


def r_erosion(image, size, origin=0):
    """Erosion with rectangular structuring element using maximum_filter"""
    return minimum_filter(image, size, origin=origin, mode='constant')


def r_opening(image, size, origin=0):
    """Opening with rectangular structuring element using maximum/minimum filter"""
    image = r_erosion(image, size, origin=origin)
    return r_dilation(image, size, origin=origin)


def r_closing(image, size, origin=0):
    """Closing with rectangular structuring element using maximum/minimum filter"""
    image = r_dilation(image, size, origin=0)
    return r_erosion(image, size, origin=0)

def intersect_boxes(boxes):
    if len(boxes) < 1:
        return None
    x1, y1, x2, y2 = boxes[0]
    for b in boxes[1:]:
        x1, y1 = max(x1, b[0]), max(y1, b[1])
        x2, y2 = min(x2, b[2]), min(y2, b[3])

    return [x1, y1, x2, y2]

def union_boxes(boxes):
    if len(boxes) < 1:
        return None
    x1, y1, x2, y2 = boxes[0]
    for b in boxes[1:]:
        x1, y1 = min(x1, b[0]), min(y1, b[1])
        x2, y2 = max(x2, b[2]), max(y2, b[3])

    return [x1, y1, x2, y2]

def filter_overlap_boxes(boxes, with_meta = False, return_indices = False):
    if (len(boxes) < 2):
        return boxes

    is_overlap = [False] * len(boxes)

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if (i == j): continue
            if with_meta:
                x1, y1, x2, y2 = boxes[i][0]
                x3, y3, x4, y4 = boxes[j][0]
            else:
                x1, y1, x2, y2 = boxes[i]
                x3, y3, x4, y4 = boxes[j]
            if (is_overlap[j] == False and abs(x1-x2) <= abs(x3-x4)) and (x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4):
                is_overlap[i] = True
                break

    # print('overlap', len([i for i in is_overlap if i]))
    if return_indices:
        return is_overlap

    return [boxes[i] for i in range(len(boxes)) if not is_overlap[i]]

def filter_overlap_boxes_bigger(boxes, with_meta=False, intersect_thres = 0.9, min_area=0, return_indices = False):
    if (len(boxes) < 2):
        return boxes

    is_overlap = [False] * len(boxes)

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if (i == j): continue
            if with_meta:
                rect_a = boxes[i][0]
                rect_b = boxes[j][0]
            else:
                rect_a = boxes[i]
                rect_b = boxes[j]
            intersect_a = intersect_area(rect_a, rect_b, min_thresh=0)
            area_i = rect_area(rect_a)
            area_j = rect_area(rect_b)
            if (is_overlap[i] == False and (
                    (area_i <= area_j) and intersect_a > intersect_thres * min(area_j, area_i) and min(area_i, area_j) > min_area)):
                is_overlap[i] = True
                break

    if return_indices:
        return is_overlap

    return [boxes[i] for i in range(len(boxes)) if not is_overlap[i]]


def rect_area(rect):
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)


def intersect_area(box_a, box_b, min_thresh=2):
    x1, y1, x2, y2 = box_a
    x3, y3, x4, y4 = box_b

    left, right = max(x1, x3), min(x2, x4)
    top, bottom = max(y1, y3), min(y2, y4)

    # print(box_a, box_b, left, right, top, bottom)

    if left <= right - min_thresh and top <= bottom - min_thresh:
        # print(1.0 * (right - left) * (bottom - top))
        return 1.0 * (right - left + 1) * (bottom - top + 1)

    return 0.0


def check_intersect_boxes(boxes, scale):
    is_box_overlap = [False] * len(boxes)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if (i == j): continue
            if intersect_area(boxes[i], boxes[j]) > scale * 15:
                is_box_overlap[i] = True
                break

    return is_box_overlap


#### check if big rectangle box contain smaller one
def is_overlap(big_box, small_box, pad=2):
    x1, y1, x2, y2 = small_box
    x3, y3, x4, y4 = big_box
    x3 -= pad
    y3 -= pad
    x4 += pad
    y4 += pad
    return (x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4)

def IoU(rect_a, rect_b):
    area_a = rect_area(rect_a)
    area_b = rect_area(rect_b)
    intersect_a = intersect_area(rect_a, rect_b, min_thresh=0)
    return 1.0 * intersect_a / area_a #max(area_a, area_b)


def scale_rect(rect, scale_factor):
    return [int(i * scale_factor) for i in rect]


def scale_pts(pts, scale_factor):
    return [[int(i * scale_factor) for i in pt] for pt in pts]

def skelet(img, thres=150, expand=False, expand_horizontal=True, iter=1):
    img = img > thres

    img = skeletonize(img)
    # img = binary_erosion(img, iterations=1)
    img = binary_dilation(img, iterations=iter)

    if expand:
        logging.debug('Expanding mask')
        pad = 5
        kernel_shape = (1, pad) if expand_horizontal else (pad, 1)
        kernel = np.ones(kernel_shape, dtype='uint8')
        img = binary_dilation(img, kernel, iterations=1)

    return img

def threshold_and_upscale_map(img_shape, gt, skeletonize=False, threshold=150, expand=False):
    h, w = img_shape[:2]
    gt = cv2.resize(gt, (w, h))
    if skeletonize:
        gt = skelet(gt, expand=expand)
    else:
        gt = gt > threshold

    return gt
