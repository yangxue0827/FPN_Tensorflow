# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def roi_visualize(img, img_h, img_w, roi_box, rois):
    with tf.variable_scope('roi_visualize'):
        ymin, xmin, ymax, xmax = tf.unstack(roi_box, axis=1)

        normalize_ymin = ymin / img_h
        normalize_xmin = xmin / img_w
        normalize_ymax = ymax / img_h
        normalize_xmax = xmax / img_w

        tmp_img = tf.squeeze(img) + tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        tmp_img = tf.cast(tmp_img * 225 / tf.reduce_max(tmp_img), dtype=tf.uint8)
        tmp_img = tf.expand_dims(tmp_img, axis=0)
        target = tf.image.crop_and_resize(tmp_img,
                                          boxes=tf.transpose(tf.stack([normalize_ymin, normalize_xmin,
                                                                       normalize_ymax, normalize_xmax])),
                                          box_ind=tf.zeros(shape=[tf.shape(roi_box)[0], ],
                                                           dtype=tf.int32),
                                          crop_size=[56, 56],
                                          name='crop_img_object'
                                          )

        rois = tf.image.resize_bilinear(rois, size=[56, 56])
        rois_mean = tf.reduce_mean(rois, axis=3)
        tf.summary.image('target', target[:, :, :, ::-1])
        tf.summary.image('rois', tf.expand_dims(rois_mean, axis=3))
