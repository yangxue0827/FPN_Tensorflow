# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
all of these functions are written by Google, 
you can find it in the Google Detection API
'''
import tensorflow as tf
import numpy as np

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  NOTE: ****
        if use tf.gather will cost lots of  gpu memory ---- added by yjr
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])


def gather_with_axis_1(tensor, indices):
  '''

  :param tensor:
  :param indices:
  :return:
  '''
  def py_gather(arry, indices):

    return arry[:, 1, :]