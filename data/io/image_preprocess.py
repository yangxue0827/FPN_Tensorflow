# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
# import


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, target_shortside_len * w//h),
                           false_fn=lambda: (target_shortside_len * h//w,  target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)

    xmin, xmax = xmin * new_w//w, xmax * new_w//w
    ymin, ymax = ymin * new_h//h, ymax * new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0) # ensure imgtensor rank is 3
    return img_tensor, tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                               false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]


def flip_left_right(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_left_right(img_tensor)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_xmin = w - xmax
    new_xmax = w - xmin
    # return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))
    return img_tensor, tf.transpose(tf.stack([ymin, new_xmin, ymax, new_xmax, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label


