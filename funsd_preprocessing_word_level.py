from __future__ import print_function, unicode_literals
import json
import glob
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import utils.graph_building_utils as gbutils
import pickle
from sklearn.feature_extraction.text import CountVectorizer

transformer_model = SentenceTransformer('bert-base-nli-mean-tokens')

__author__ = "Marc"


def normalize_pos_feats(list_bboxs, clamp_min=0.1):
    list_bboxs = np.array(list_bboxs).astype('float')  # expected to be Nx4
    min_x = np.min(list_bboxs[:, 0])
    min_y = np.min(list_bboxs[:, 1])
    max_x = np.max(list_bboxs[:, 0] + list_bboxs[:, 2])
    max_y = np.max(list_bboxs[:, 1] + list_bboxs[:, 3])
    list_bboxs[:, 0] = (list_bboxs[:, 0] - min_x)/(
        max_x - min_x)

    list_bboxs[:, 1] = (list_bboxs[:, 1] - min_y)/(
        max_x - min_x)

    list_bboxs[:, 2] = (list_bboxs[:, 2])/(
        max_x - min_x)
    list_bboxs[:, 3] = (list_bboxs[:, 3])/(max_y - min_y)
    list_bboxs = (list_bboxs + clamp_min)/(clamp_min+1.0)
    return list_bboxs


def get_inv_dict_charset(charset):
    inv_dict_charset = dict([(c, i) for i, c in enumerate(charset)])
    for i, c in enumerate(charset):
        onehot_char = np.zeros(len(charset))
        onehot_char[i] = 1.0
        inv_dict_charset[c] = onehot_char
    return inv_dict_charset


def get_charset(corpus):
    corpus = ''.join(corpus.split())
    charset = sorted(list(set(corpus)))
    return charset, get_inv_dict_charset(charset)


def transform_from_charset(text, inv_dict_charset):
    mat_repr = np.zeros((len(text), len(inv_dict_charset.keys())))
    for i, c in enumerate(text):
        try:
            mat_repr[i, :] = inv_dict_charset[c]
        except KeyError:
            continue
    return mat_repr


def get_preprocessed_list_word_msau(dirpath,
                                    inv_dict_charset=None):
    data_preprocessed_list = []
    list_ocrs_total = []
    for json_filename in glob.glob(os.path.join(dirpath, "*.json")):
        list_bboxs = []
        list_ocrs = []
        list_labels = []
        list_linking = []
        list_ids = []
        json_dict = json.load(open(json_filename))['form']
        list_bbox_word = []
        list_ocr_word = []
        list_word_textline = []
        for textline in json_dict:
            list_bboxs.append(
                [textline['box'][0], textline['box'][1],
                 textline['box'][2]-textline['box'][0]+1,
                 textline['box'][3]-textline['box'][1]+1])
            list_ocrs.append(textline['text'])
            for word in textline['words']:
                list_bbox_word.append(
                    [word['box'][0], word['box'][1],
                     word['box'][2]-word['box'][0]+1,
                     word['box'][3]-word['box'][1]+1])
                list_ocr_word.append(word['text'])
                list_word_textline.append(len(list_bboxs)-1)
            list_labels.append(textline['label'])
            list_linking.append(textline['linking'])
            list_ids.append(textline['id'])
        list_ocrs_total.extend(list_ocrs)
        cell_list = gbutils.get_list_cells(list_bboxs, list_ocrs)
        cell_list_word = gbutils.get_list_cells(list_bbox_word, list_ocr_word)

        data_dict = {'file_path': json_filename}
        data_dict['word_to_textline'] = list_word_textline
        data_dict['cells_word'] = cell_list_word
        data_dict['cells'] = cell_list
        data_dict['labels'] = list_labels
        data_dict['ids'] = list_ids
        data_dict['link'] = list_linking
        data_preprocessed_list.append(data_dict)

    if inv_dict_charset is None:
        charset, inv_dict_charset = get_charset(
            ' '.join(list_ocrs_total))
        print("Charset len: ", len(charset))

    for data_dict in data_preprocessed_list:
        list_word_feats = []
        for cell in data_dict['cells_word']:
            list_word_feats.append(transform_from_charset(cell.ocr_value,
                                                          inv_dict_charset))
        data_dict['charset_feature'] = list_word_feats
    return data_preprocessed_list, inv_dict_charset


if __name__ == '__main__':
    train_data_list, inv_dict_charset = get_preprocessed_list_word_msau(
        "dataset/training_data/annotations/")
    test_data_list, inv_dict_charset = get_preprocessed_list_word_msau(
        "dataset/testing_data/annotations/", inv_dict_charset=inv_dict_charset)
    pickle.dump(inv_dict_charset, open('inv_dict_charset.pkl', 'wb'))
    pickle.dump(train_data_list,
        open('funsd_preprocess_train_word.pkl', 'wb'))
    pickle.dump(test_data_list,
        open('funsd_preprocess_test_word.pkl', 'wb'))
