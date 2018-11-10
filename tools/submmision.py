#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Get the submission.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2, glob, json
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__', 'd00', 'd01', 'd10', 'd11', 'd20', 'd40', 'd43', 'd44')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_70000.ckpt',),
        'res152': ('res152_faster_rcnn_iter_70000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),
           'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def test(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the test image
    im_file = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007', 'VOC2007', 'JPEGImages', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    result_data = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        result_data_part = get_results(cls, dets, thresh=CONF_THRESH)
        if len(result_data_part) != 0:
            result_data.append(result_data_part)
    return result_data

def get_results(class_name, dets, thresh=0.5):
    """Get detection results for this class."""

    result_data_objects = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return {}
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        result_bbox = {
            'xmin': float(bbox[0]),
            'ymin': float(bbox[1]),
            'xmax': float(bbox[2]),
            'ymax': float(bbox[3])
        }
        result_data_object = {
            'bbox': result_bbox,
            'score': float(score)
        }
        result_data_objects.append(result_data_object)
    result_data = {
        'category': class_name,
        'objects': result_data_objects
    }
    return result_data

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101 res152]',
                        choices=NETS.keys(), default='res152')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join(cfg.ROOT_DIR, 'output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True  
    
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 9, tag='default', 
                              anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2, 4, 8])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    
    print('Loaded network {:s}'.format(tfmodel))

    im_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007', 'VOC2007', 'JPEGImages')
    im_names = glob.glob(r'%s/test*.jpg' %(im_path))
    im_names.sort()

    csv = os.path.join(cfg.ROOT_DIR, os.path.splitext(NETS[demonet][0])[0] + '_submissions_aug.csv')
    with open(csv, 'w') as submissions:
        for im_name in im_names:
            im_name = im_name.split('/')[-1]
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Detection for {}'.format(im_name))
            result = test(sess, net, im_name)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('result:' + json.dumps(result, indent=4, separators=(',', ':')))
            object_score_dict = {}
            for category_result_dict in result:
                for object_dict in category_result_dict['objects']:
                    predition_string = str(CLASSES.index(category_result_dict['category'])) + \
                                       ' ' + str(int(object_dict['bbox']['xmin'])) + ' ' + str(int(object_dict['bbox']['ymin'])) + \
                                       ' ' + str(int(object_dict['bbox']['xmax'])) + ' ' + str(int(object_dict['bbox']['ymax']))
                    object_score_dict[predition_string] = object_dict['score']
            predition_strings = ''
            top_5 = sorted(object_score_dict.items(), key=lambda item:item[1], reverse=True)[:5]
            for object_score in top_5:
                predition_strings += object_score[0] + ' '
            submissions.write(im_name + ',' + predition_strings + '\n')
