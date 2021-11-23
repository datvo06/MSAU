from __future__ import print_function
from __future__ import division
'''This file contains original heuristic rules written for the graph-based
key-value
'''
from .bbox_operations import (
    check_intersect_horizontal_proj,
    check_intersect_vertical_proj,
    get_intersect_range_horizontal_proj,
    get_intersect_range_vertical_proj,
    check_bbox_contains_each_other,
    check_bbox_almost_contains_each_other
)
import cv2
import numpy as np

__author__ = "Marc"


def get_threshhold_really_horizontal():
    return 2.0


def get_threshold_really_vertical():
    return 0.2


def is_direct_vertical_with_a_condition(cell1, cell2, other_cond):
    if cell1.y > cell2.y:
        cell1, cell2 = cell2, cell1
    cond2 = ((cell2.y - (cell1.y+cell1.h)) < min(cell1.h, cell2.h))
    return cond2 and other_cond(cell1, cell2)


def is_align_horizontal_with_almost_max_width(cell1, cell2, thresh_ratio=0.9):
    return get_intersect_range_horizontal_proj(
        cell1.get_bbox(), cell2.get_bbox()) > max(
        cell1.h, cell2.h)*thresh_ratio


def is_direct_align_horizontal(cell1, cell2, thresh_ratio=0.9):
    cond1 = is_align_horizontal_with_almost_max_width(
        cell1, cell2, thresh_ratio)
    if cell1.x > cell2.x:
        cell1, cell2 = cell2, cell1
    if (not cell1.is_sub) or (not cell2.is_sub):
        min_width = min(cell1.w, cell2.w)
    elif cell1.is_sub and cell2.is_sub:
        min_width = min(cell1.w, cell2.w)*3
    cond2 = ((cell2.x - (cell1.w+cell1.x)) < min_width)
    return cond1 and cond2


def is_talign_horizontal(cell1, cell2):
    lintersect = abs(cell1.y - cell2.y)
    return lintersect < max(0.2*max(cell1.h, cell2.h), 5)


def is_balign_horizontal(cell1, cell2):
    lintersect = abs(cell1.y+cell1.h - cell2.y - cell2.h)
    lintersect = min(lintersect, abs(cell2.y+cell2.h - cell1.y - cell1.h))
    return lintersect < max(0.2*max(cell1.h, cell2.h), 20)


def is_lalign_vertical(cell1, cell2):
    lintersect = abs(cell1.x - cell2.x)
    return lintersect < max(0.2*max(cell1.w, cell2.w), 20)


def is_direct_lalign_vertical(cell1, cell2):
    return is_direct_vertical_with_a_condition(cell1, cell2, is_lalign_vertical)


def is_direct_ralign_vertical(cell1, cell2):
    return is_direct_vertical_with_a_condition(cell1, cell2, is_ralign_vertical)


def is_ralign_vertical(cell1, cell2):
    lintersect = abs(cell1.x+cell1.w - cell2.x - cell2.w)
    lintersect = min(lintersect, abs(cell2.x+cell2.w - cell1.x - cell1.w))
    return lintersect < max(0.2*max(cell1.w, cell2.w), 5)


def is_direct_align_vertical(cell1, cell2):
    cond1 = get_intersect_range_vertical_proj(
        cell1.get_bbox(), cell2.get_bbox()) > max(
        cell1.w, cell2.w)*0.9
    if cell1.y > cell2.y:
        cell1, cell2 = cell2, cell1
    cond2 = ((cell2.y - (cell1.y+cell1.h)) < min(cell1.h, cell2.h))
    return cond1 and cond2


def get_list_direct_vertical_with_cond(cell, cond, take_top=False):
    if take_top:
        cell_query_set = cell.tops
    else:
        cell_query_set = cell.bottoms
    return [rcell for rcell in cell_query_set
            if cond(cell, rcell)]


def get_list_direct_align_vertical(cell, take_top=False):
    return get_list_direct_vertical_with_cond(
        cell, is_direct_align_vertical, take_top)


def get_list_direct_lalign_vertical(cell, take_top=False):
    return get_list_direct_vertical_with_cond(cell, is_direct_lalign_vertical,
                                              take_top)


def get_list_lalign_vertical(cell, take_top=False):
    return get_list_direct_vertical_with_cond(cell, is_lalign_vertical, take_top)


def get_list_ralign_vertical(cell, take_top=False):
    return get_list_direct_vertical_with_cond(cell, is_ralign_vertical, take_top)


def get_list_direct_ralign_vertical(cell, take_top=False):
    return get_list_direct_vertical_with_cond(cell, is_direct_ralign_vertical,
                                              take_top)


def get_list_direct_align_horizontal(cell, take_left=False, thresh_ratio=0.9):
    if take_left:
        cell_query_set = cell.lefts
    else:
        cell_query_set = cell.rights
    return [rcell for rcell in cell_query_set
            if is_direct_align_horizontal(cell, rcell, thresh_ratio)]


def build_left_right_edges(cell_list_top_down):
    i = 0
    for cell in cell_list_top_down:
        i += 1
        # if cell.name not in last_keys:
        #    continue
        cell_collide = [other_cell
                        for other_cell in cell_list_top_down
                        if other_cell.x >= cell.x and
                        check_intersect_horizontal_proj(
                            cell.get_bbox(), other_cell.get_bbox()
                        ) and cell != other_cell]
        cell_collide = [other_cell
                        for other_cell in cell_collide
                        if get_intersect_range_horizontal_proj(
                            cell.get_bbox(), other_cell.get_bbox()
                        ) > min(cell.h, other_cell.h)*0.3]

        for other_cell in cell_collide:
            if cell.is_left_of(other_cell, cell_collide
                               ) and other_cell not in cell.rights:
                cell.rights.append(other_cell)
                other_cell.lefts.append(cell)


def build_top_down_edges(cell_list_left_right):
    i = 0
    for cell in cell_list_left_right:
        i += 1
        # if cell.name not in last_keys:
        #    continue
        cell_collide = [other_cell
                        for other_cell in cell_list_left_right
                        if (other_cell.y > cell.y + cell.h)*0.8 and
                        check_intersect_vertical_proj(
                            cell.get_bbox(), other_cell.get_bbox()
                        ) and cell != other_cell]
        for other_cell in cell_collide:
            if cell.is_top_of(other_cell, cell_collide
                              ) and other_cell not in cell.bottoms:
                cell.bottoms.append(other_cell)
                other_cell.tops.append(cell)


def build_containing_edges(cell_list_area):
    '''
    args:
        :param cell_list_area: cell_list sorted by area, ascending
    '''
    for i, cell in enumerate(cell_list_area):
        cell_list_containing_this = [
            ocell for ocell in cell_list_area[i+1:]
            if check_bbox_contains_each_other(cell, ocell) or
            check_bbox_almost_contains_each_other(cell, ocell)
        ]
        cell.parents = cell_list_containing_this
        for ocell in cell_list_containing_this:
            ocell.children.append(cell)


def build_cell_relation(cell_list):
    # 1. left-right
    cell_list_top_down = sorted(cell_list, key=lambda cell: cell.y)
    cell_list_left_right = sorted(cell_list, key=lambda cell: cell.x)
    cell_list_area = sorted(cell_list, key=lambda cell: cell.w*cell.h)
    # 1.1 Check this cell with every cell to the right of it
    # TODO: More effective iteration algo e.g: cached collisions matrix
    build_left_right_edges(cell_list_top_down)
    # 2. top-down
    build_top_down_edges(cell_list_left_right)
    build_containing_edges(cell_list_area)


class CellNode(object):
    """ Class representing cell node """
    current_created_cells = 0

    def __init__(self, x, y, w, h, ocr_value="", is_sub=False):
        """ init cell
        input:
            x, y, w, h: coords & dimmension of cell
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_dotted = False
        self.is_sub = is_sub
        self.name = "node_" + str(CellNode.current_created_cells)
        CellNode.current_created_cells += 1

        # Get cell background

        # Init adjacent
        self.lefts = []
        self.rights = []
        self.tops = []
        self.parents = []
        self.children = []
        self.bottoms = []
        self.ocr_value = ocr_value

    def remove(self):
        # Remove only, adding relation can wait
        for lcell in self.lefts:
            lcell.rights = [ocell for ocell in lcell.rights if ocell != self]
        for rcell in self.rights:
            rcell.lefts = [ocell for ocell in rcell.lefts if ocell != self]
        for tcell in self.tops:
            tcell.bottoms = [ocell for ocell in tcell.bottoms if ocell != self]
        for bcell in self.rights:
            bcell.tops = [ocell for ocell in bcell.tops if ocell != self]

    def get_aspect_ratio(self):
        return self.w/self.h

    def is_really_horizontal_cell(self):
        """Threshhold based determination
        To do: consider text orientation
        """
        return self.get_aspect_ratio() > get_threshhold_really_horizontal()

    def is_really_vertical_cell(self):
        """Threshold based determination
        To do: consider text orientation"""
        return self.get_aspect_ratio() < get_threshold_really_vertical()

    def get_bbox(self):
        return (self.x, self.y, self.w, self.h)

    def get_real_sub_lines(self):
        if len(self.sub_lines) == 0:
            if self.is_sub:
                return [self]
            else:
                return []
        else:
            return_list = []
            if self.is_sub:
                return_list.append(self)
            for child in self.sub_lines:
                return_list.extend(child.get_real_sub_lines())
            return return_list

    def is_left_of(self, other_cell, ref_cells):
        """Check if this cell is directly left of
        other cell given a set of full cells sorted"""
        # 0. Check if other_cell in self.rights
        if other_cell in self.rights:
            return True

        if other_cell.x < self.x or not check_intersect_horizontal_proj(
                self.get_bbox(), other_cell.get_bbox()
        ):
            return False
        if get_intersect_range_horizontal_proj(
            self.get_bbox(), other_cell.get_bbox()
        ) > 0.9*min(self.h, other_cell.h):
            if other_cell.x - self.x < 0.1*min(self.w, other_cell.w):
                return True

        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [cell for cell in ref_cells
                     if check_intersect_horizontal_proj(
                         self.get_bbox(), cell.get_bbox()) and
                     (cell.x + cell.w) < other_cell.x + other_cell.w*0.1 and
                     cell.x >= (self.x+self.w*0.8) and
                     check_intersect_horizontal_proj(
                         self.get_bbox(), cell.get_bbox())
                     ]
        # 2. filters all the small overlapping cells
        ref_cells = [cell for cell in ref_cells
                     if get_intersect_range_horizontal_proj(
                         self.get_bbox(),
                         cell.get_bbox()
                     ) > min(self.h, cell.h) / 5]
        ref_cells = [cell for cell in ref_cells
                     if
                     get_intersect_range_horizontal_proj(
                         cell.get_bbox(),
                         other_cell.get_bbox()
                     ) > other_cell.h / 2 or
                     get_intersect_range_horizontal_proj(
                         self.get_bbox(),
                         cell.get_bbox()
                     ) > min(cell.h, self.h)*0.8
                     ]

        # 3. Check if there are any cells lies between this and other_cell
        if len(ref_cells) > 0:
            return False

        # 4. return results
        return True

    def is_right_of(self, other_cell, ref_cells):
        """Check if this cell is directly left of
        other cell given a set of full cells sorted"""
        return other_cell.is_left_of(self, ref_cells)

    def is_top_of(self, other_cell, ref_cells):
        """Check if this cell is directly left of
        other cell given a set of full cells sorted"""
        # 0. Check if other_cell in self.rights
        if other_cell in self.bottoms:
            return True

        if other_cell.y < self.y or not check_intersect_vertical_proj(
                self.get_bbox(), other_cell.get_bbox()):
            return False

        if get_intersect_range_vertical_proj(
            self.get_bbox(), other_cell.get_bbox()) < min(
                self.w, other_cell.w)/5:
            return False
        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [cell for cell in ref_cells
                     if check_intersect_vertical_proj(
                         self.get_bbox(), cell.get_bbox()) and
                     (cell.y + cell.h) < other_cell.y + other_cell.h*0.1 and
                     cell.y >= (self.y+self.h*0.8) and
                     check_intersect_vertical_proj(
                         self.get_bbox(), cell.get_bbox())
                     ]
        # 2. filters all the small overlapping cells
        ref_cells = [cell for cell in ref_cells
                     if
                     get_intersect_range_vertical_proj(
                         self.get_bbox(),
                         cell.get_bbox()
                     ) > min(self.w, cell.w) / 5]
        ref_cells = [cell for cell in ref_cells
                     if get_intersect_range_vertical_proj(
                         cell.get_bbox(),
                         other_cell.get_bbox()
                     ) > other_cell.w / 2 or
                     get_intersect_range_vertical_proj(self.get_bbox(),
                                                       cell.get_bbox()
                                                       ) >
                     min(self.w, cell.w)*0.8
                     ]

        # 3. Check if there are any cells lies between this and other_cell
        if len(ref_cells) > 0:
            return False

        # 4. return result
        return True

    def get_center(self):
        return (int(self.x+self.w/2), int(self.y+self.h/2))

    def set_text(self, text):
        self.ocr_value = text

    def __getitem__(self, key):
        return self.get_bbox()[key]


def get_list_cells(list_bboxs, ocr_values):
    '''
    Args:
        :param list_bboxs: list of (x, y, w, h)
        :param ocr values: list of texts
    '''
    return [CellNode(bbox[0], bbox[1], bbox[2], bbox[3], ocr_values[i])
            for i, bbox in enumerate(list_bboxs)]


def get_cell_from_cell_dict(cell_dict, img):
    import re
    list_bboxs = []
    list_ocr_values = []
    for key in cell_dict:
        # Initialize new cell
        list_bboxs.append(cell_dict[key]['location'])
        list_ocr_values.append(cell_dict[key].get('value', ''))
    return get_list_cells(list_bboxs, list_ocr_values)


def get_adj_mat(cell_list):
    build_cell_relation(cell_list)
    N = len(cell_list)
    adj = np.zeros((N, N, 6))
    for i, cell in enumerate(cell_list):
        cell.index = i
    for i, cell in enumerate(cell_list):
        adj[i, [ocell.index for ocell in cell.lefts], 0] = 1
        adj[i, [ocell.index for ocell in cell.rights], 1] = 1
        adj[i, [ocell.index for ocell in cell.tops], 2] = 1
        adj[i, [ocell.index for ocell in cell.bottoms], 3] = 1
        adj[i, [ocell.index for ocell in cell.parents], 4] = 1
        adj[i, [ocell.index for ocell in cell.children], 5] = 1
    return adj


def get_dict_from_cell_list(cell_list):
    '''This function output a dictionary in normalized form from cell_list
    '''
    the_dict = {}
    for cell in cell_list:
        the_dict[cell.name] = {"value": cell.get_text(),
                               "location": [cell.y, cell.x, cell.w, cell.h],
                               "label": cell.label}
    return the_dict
