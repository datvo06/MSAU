"""
IMPORTANCE: bbox format is (x, y, w, h)
"""
from __future__ import print_function, unicode_literals
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os
import cv2


currdir = os.path.dirname(os.path.abspath(__file__))

fonts = []
for i in range(1, 24, 2):
    fonts.append(ImageFont.truetype(
        os.path.join(currdir, "arial-unicode-ms.ttf"), i))


def get_used_font(top_height):
    i = 0
    while(top_height > fonts[i].font.height and i < len(fonts) - 1):
        i += 1
    print(top_height, fonts[i-1].font.height)

    return fonts[i]


def get_colors_list_boxs():
    """
    B, G, R for openCV
    """
    return [(255, 0, 0), (0, 255, 0), (127, 127, 0)]


def get_colors_list_edges():
    return [(127, 0, 127), (0, 0, 0), (0, 127, 127)]


def draw_bbox(img, x, y, w, h, color, thickness=5):
    """
    This return the drawn image
    """
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    # print(x, y, w, h)
    return cv2.rectangle(img, (x, y), (x+w, y+h), color, int(thickness))


def draw_node_bbox(img, bbox, label, importance=1.0):
    """
    For each node, denote significant by bbox
    """
    # # print("\'draw_node_bbox\' inputs:",
    #       img.shape, bbox, label, importance, sep="\n")
    colors = get_colors_list_boxs()
    # print("colors", colors)
    # TODO: choose colors.

    return draw_bbox(img, bbox[0], bbox[1], bbox[2], bbox[3],
                     colors[label % len(colors)],
                     min(max(int(5*importance), 1), 5))


def draw_nodes_bboxs(img, list_bboxs, labels, importances):
    for i, bbox in enumerate(list_bboxs):
        img = draw_node_bbox(img, bbox, int(labels[i]), importances[i])
    return img


def draw_text(img, x, y, w, h, text, color=(255, 255, 255),
              importances=None):
    """
    This will never be called here, it will be call in the outer folder instead
    TODO: Testing
    """
    font = get_used_font(h)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    charwidth = int(font.getsize(text)[0]/max(len(text), 1))
    if importances is None:
        draw.text(
            (x, y), text, font=font,
            # fill=(color))
            fill=(0, 0, 0))
    else:
        for i, character in enumerate(text):
            draw.text(
                (x+i*charwidth+1, y),
                character, font=font
                ,
                fill=(
                    tuple(
                        # [int(0)
                        [int(127-0.5*color_elem*(importances[i]))
                         for color_elem in color])
                ),
                )
    return np.array(pil_img)


def draw_node_text_importances(img, list_bboxs,
                               list_texts,
                               bow_dict,
                               bow_importances):

    get_mapping = (lambda c: c)\
        if bow_dict is None else (lambda c: bow_dict[c])
    for i, bbox in enumerate(list_bboxs):
        if bow_dict is None:
            img = draw_text(
                img, bbox[0], bbox[1], bbox[2], bbox[3],
                list_texts[i],
                importances=None)

        else:
        # print(list_texts[i])
            img = draw_text(
                img, bbox[0], bbox[1], bbox[2], bbox[3],
                list_texts[i],
                importances=[bow_importances[i][get_mapping(c)]
                                   for c in list_texts[i]])
    return img


def get_pseudo_text_bow_repr(bows, word_list):
    concated_string = ''.join([word_list[i] for i, val in enumerate(bows) if val != 0])
    return concated_string


def draw_arrow(img, x1, y1, x2, y2, color, thickness=1):
    print(x1, y1, x2, y2, color)
    return cv2.arrowedLine(img, (int(x1), int(y1)),
                           (int(x2), int(y2)), color, max(int(thickness), 1))


def draw_node_position_feature_importances(img, bboxes, node_pos_importances):
    # feature order: topleft, top right, lower right, lowerleft
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        for j, pt in enumerate(pts):
            cv2.circle(img, pt, 5*node_pos_importances[i][j], (0, 255, 0), -1)
    return img


def draw_position_feature_importances(img, list_bboxs, position_importances):
    """
    Denote significant of features by radius of filled circles
    """
    img = img.copy()
    for i, bbox in enumerate(list_bboxs):
        draw_node_position_feature_importances(img,
                                               bbox,
                                               position_importances[i])
    return img


def draw_nodes(img, list_texts, list_bboxs,
               node_labels,
               node_importances,
               position_importances,
               bow_importances,
               bow_dict,
               cur_node_idx
               ):
    """
    Args:

        list_bboxs:

        cur_node_idx: Which node to fill color.

        Node importances: N
        position_importances: N
        bow importances: NxBoW
        bow_dict: dict mapping from char to feat index
    """

    print(list_bboxs)
    # Fill the current Node explained.
    if cur_node_idx is not None:
        x, y, w, h = list_bboxs[cur_node_idx]
        img = draw_bbox(img, x, y, w, h,
                        color=(255, 125, 125),
                        thickness=cv2.FILLED)

    # First, draw the nodes based on the importances
    img = draw_nodes_bboxs(img, list_bboxs, node_labels, node_importances)

    # Then, draw each nodes's positional importances
    img = draw_node_position_feature_importances(img, list_bboxs,
                                                 position_importances)
    # Finally, draw each node's text with importances

    img = draw_node_text_importances(img, list_bboxs, list_texts,
                                     bow_dict, bow_importances)

    return img


def draw_edges(img, list_bboxs, adj_mats, adj_importances_mask):
    """
    Args:
        img: input opencv image to be drawn on
        list_bboxs: list of (x, y, w, h)
        adj_mats: (N, N, D)
    """
    img = img.copy()
    list_bboxs = np.array(list_bboxs)
    # print("\n\n\nMin in draw edges:", np.min(list_bboxs[:, 0]), np.min(list_bboxs[:, 1]), sep="\n")
    # print("\n\n\nMax in draw edges:", np.max(list_bboxs[:, 0]), np.max(list_bboxs[:, 1]), sep="\n")
    bbox_centers = np.vstack(
        ((list_bboxs[:, 0] + list_bboxs[:, 2]/2),
         (list_bboxs[:, 1] + list_bboxs[:, 3]/2))).transpose().astype('uint')

    # print("\n\n\nMin in bbox_centers:", np.min(bbox_centers), sep="\n")
    # print("\n\n\nMax in bbox_centers:", np.max(bbox_centers), sep="\n")

    # Loop through every node
    for i, bbox in enumerate(list_bboxs):
        for k in range(int(adj_mats.shape[-1]/2)):
            list_j = np.argwhere(adj_mats[i, :, 2*k] != 0)
            # print(list_j)
            for j in list_j:
                j = int(j)
                img = draw_arrow(
                    img,
                    bbox_centers[i][0], bbox_centers[i][1],
                    bbox_centers[j][0], bbox_centers[j][1],
                    get_colors_list_edges()[k % len(get_colors_list_edges())],
                    thickness=7*(adj_importances_mask[i, j, 2*k] + adj_importances_mask[i, j, 2*k+1]))
    return img


def visualize_graph(list_bows, list_positions,
                    adj_mats, node_labels,
                    cur_node_idx=None,
                    node_importances=None,
                    position_importances=None,
                    bow_importances=None,
                    adj_importances=None,
                    orig_img=None,
                    word_list=None,
                    is_text=False):
    """
    Args:
        list_bows: the bag of words features
        list_positions: list of (x, y, w, h)
        adj_mats: matrix of NxNxE (E is number of edge types)
        word_list: the list of word (if we were to visualize bows), aka corpus
        is_text: if the bows is texts

        Node importances: N
        position_importances: Nx4
        bow importances: NxBoW
        adj_importances: NxNxE

    """
    N = list_bows.shape[0]

    if node_importances is None:
        node_importances = np.ones(N) * 0.3
    if position_importances is None:
        position_importances = np.ones((N, 4, 1)) * 0.3
    if bow_importances is None:
        bow_importances = np.ones(list_bows.shape) * 0.3
    if adj_importances is None:
        adj_importances = np.ones(adj_mats.shape) * 0.3

    # First, get the texts for all bows
    list_bows = list_bows.astype(int)
    list_texts = [str(i for i in range(len(bow)) if bow[i] != 0)
                  for bow in list_bows]
    bow_dict = dict([(word, i) for i, word in enumerate(word_list)]
                    ) if word_list is not None else None

    if not is_text and word_list is not None:
        list_texts = [get_pseudo_text_bow_repr(bow, word_list)
                      for bow in list_bows]
    # print(list_texts)
    max_x = np.max(list_positions[:, 0] + list_positions[:, 2])
    max_y = np.max(list_positions[:, 1] + list_positions[:, 3])
    if orig_img is None:
        img = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        img.fill(255)  # Fill with grey
    else:
        img = orig_img.copy()
    # First, draw the nodes
    img = draw_nodes(
                img, list_texts, list_positions,
                node_labels, node_importances, position_importances,
                bow_importances, bow_dict, cur_node_idx)
    # then, draw the edges
    # TODO:
    img = draw_edges(img, list_positions, adj_mats,
                     adj_importances)
    return img


def visualize_graph_text(
        list_texts, list_positions,
        adj_mats, node_labels,
        cur_node_idx=None,
        node_importances=None,
        position_importances=None,
        bow_importances=None,
        adj_importances=None,
        orig_img=None,
        word_list=None,
        ):
    """
    Args:
        list_bows: the bag of words features
        list_positions: list of (x, y, w, h)
        adj_mats: matrix of NxNxE (E is number of edge types)
        word_list: the list of word (if we were to visualize bows), aka corpus
        is_text: if the bows is texts

        Node importances: N
        position_importances: Nx4
        bow importances: NxBoW
        adj_importances: NxNxE

    """
    N = len(list_texts)

    if node_importances is None:
        node_importances = np.ones(N) * 0.3
    if position_importances is None:
        position_importances = np.ones((N, 4, 1)) * 0.3
    if bow_importances is None and word_list is not None:
        bow_importances = np.ones(list_bows.shape) * 0.3
    if adj_importances is None:
        adj_importances = np.ones(adj_mats.shape) * 0.3

    # First, get the texts for all bows
    bow_dict = dict([(word, i) for i, word in enumerate(word_list)]
                    ) if word_list is not None else None

    # print(list_texts)
    max_x = np.max(list_positions[:, 0] + list_positions[:, 2])
    max_y = np.max(list_positions[:, 1] + list_positions[:, 3])
    if orig_img is None:
        img = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        img.fill(255)  # Fill with grey
    else:
        img = orig_img.copy()
    # First, draw the nodes
    img = draw_nodes(
                img, list_texts, list_positions,
                node_labels, node_importances, position_importances,
                bow_importances, bow_dict, cur_node_idx)
    # then, draw the edges
    # TODO:
    img = draw_edges(img, list_positions, adj_mats,
                     adj_importances)
    return img
