import pkg_resources
import importlib
importlib.reload(pkg_resources)
import os
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from tfx_bsl.public import tfxio

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = x - tft.mean(x)
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.compute_and_apply_vocabulary(s)
  x_centered_times_y_normalized = x_centered * y_normalized
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }

raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'y': tf.io.FixedLenFeature([], tf.float32),
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string),
    }))

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
  transformed_dataset, transform_fn = (
      (raw_data, raw_data_metadata) |
      tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

transformed_data, transformed_metadata = transformed_dataset