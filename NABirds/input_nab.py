"""Prepare input batches.
"""

from __future__ import absolute_import, division, print_function

import os

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import control_flow_ops



_SHUFFLE_BUFFER = 10000
NUM_CHANNELS = 3
TRAIN_SHARDS_NUM = 16
VAL_SHARDS_NUM = 16

def get_filenames(dataset_type, data_dir):
    """Return filenames for dataset.
    Args:
        dataset_type: string; type of dataset.
        data_dir: string; directory containing the input data.
    Returns:
        data_filemames: list of strings; it contains paths to TFRecords.
    """
    
    # Data are assumed to be stored in TFRecords
    if (dataset_type == 'train'):
        data_filemames = [os.path.join(data_dir, 'train-%04d-of-%04d' % (i+1, TRAIN_SHARDS_NUM)) for i in range(TRAIN_SHARDS_NUM)]
    elif (dataset_type == 'validation'):
        data_filemames = [os.path.join(data_dir, 'validation-%04d-of-%04d' % (i+1, VAL_SHARDS_NUM)) for i in range(VAL_SHARDS_NUM)]
  
    return data_filemames

def parse_example_proto(example_serialized, adv_eval_data=False):
    """Parse an Example proto that corresponds to an image.
    Args:
        example_serialized: string; serialized Example protocol buffer.
        adv_eval_data: boolean; whether to include information for advanced
            evaluation in the input batches.
    Returns:
        to_batch: tuple; it contains the following entries:
            encoded_img: string; encoded JPEG file.
            label: int; numeric image label.
            img_filename (optional): string; the filename of an image.
            img_label_text (optional): string; the human-readable label of an image.
    """

    # Extract dense features in Example proto
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    
    features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
    encoded_img = features['image/encoded']
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    label = tf.cast(tf.reshape(label, shape=[1]), dtype=tf.float32)

    if (not adv_eval_data):
        to_batch = (encoded_img, label)
    else:
        img_filename = features['image/filename']
        img_label_text = features['image/class/text']
        to_batch = (encoded_img, label, img_filename, img_label_text)

    return to_batch

def apply_with_random_selector(x, func, cases):
    """Compute func(x, cases[sel]), with sel sampled from cases.
    Args:
        x: Tensor; input Tensor to process.
        func: function; python function to apply.
        num_cases: list; cases to sample from.
    Returns:
        The result of func(x, cases[sel]), sel is sampled dynamically.
    """

    sel = tf.random.uniform([], maxval=len(cases), dtype=tf.int32)
    # Pass the input only to one of the func calls
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, i))[1], cases[i])
            for i in range(len(cases))])[0]

def distort_image(image_buffer, output_height, output_width, num_channels, bbox):
    """Distort an image for data augmentation.
    Args:
        image_buffer: string; raw JPEG image buffer.
        output_height: int; height of the image after preprocessing.
        output_width: int; width of the image after preprocessing.
        num_channels: int; depth of the image buffer for decoding.
        bbox: 3-D float Tensor; it contains the bounding boxes related to
            an image. Bounding box coordinates are in range [0, 1],
            arranged in order [ymin, xmin, ymax, xmax]. The Tensor is of
            shape [1, num_boxes, 4], where num_boxes is the number of
            bounding boxes related to the image.
    Returns:
        distorted_image: 3-D float Tensor; it contains an image. It is of
            size [H, W, C], where H is the image height, W is the image
            width, and C is the number of channels.
    """

    # Create a bounding box by distorting an existing one (if it is provided).
    # The new bounding box should respect specific constraints, e.g., be within
    # a range of aspect ratios. If no bounding box is provided, the entire
    # image is considered the initial bounding box to be distorted.
    sampled_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                                        tf.io.extract_jpeg_shape(image_buffer),
                                        bounding_boxes=bbox,
                                        min_object_covered=0.1,
                                        aspect_ratio_range=[0.5, 2.0],
                                        area_range=[0.85, 1.0],
                                        max_attempts=50,
                                        use_image_if_no_bounding_boxes=True,
                                        seed=0)
    bbox_begin, bbox_size, _ = sampled_distorted_bounding_box

    # Reassemble and crop the bounding box
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    distorted_image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)
    distorted_image = tf.image.convert_image_dtype(distorted_image, dtype=tf.float32)

    # Resize the image. Select a resize method randomly. The image aspect ratio may change.
    resize_methods = [tf.image.ResizeMethod.BILINEAR,
                      tf.image.ResizeMethod.LANCZOS3,
                      tf.image.ResizeMethod.LANCZOS5,
                      tf.image.ResizeMethod.BICUBIC,
                      tf.image.ResizeMethod.GAUSSIAN,
                      tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                      tf.image.ResizeMethod.AREA,
                      tf.image.ResizeMethod.MITCHELLCUBIC]
    distorted_image = apply_with_random_selector(distorted_image,
                                                 lambda x, resize_method: tf.image.resize(distorted_image,
                                                 [output_height, output_width],
                                                 method=resize_method, antialias=False),
                                                 cases=resize_methods)

    # Restore image shape
    distorted_image.set_shape([output_height, output_width, num_channels])

    # Perform a random horizontal flip of the image
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Perform a random translation of the image
    distorted_image = tf.expand_dims(distorted_image, 0)
    s = 0.1
    vy = s * tf.cast(tf.shape(distorted_image)[1], tf.float32)
    vx = s * tf.cast(tf.shape(distorted_image)[2], tf.float32)
    dy = tf.random.uniform(shape=[tf.shape(distorted_image)[0], 1], minval=-vy, maxval=vy)
    dx = tf.random.uniform(shape=[tf.shape(distorted_image)[0], 1], minval=-vx, maxval=vx)
    d = tf.concat([dx, dy], axis=-1)
    distorted_image = tfa.image.translate(distorted_image, translations=d)
    
    # Perform a random rotation of the image
    r_limit = 20.0 * np.pi / 180.0
    r = tf.random.uniform(shape=[tf.shape(distorted_image)[0]], minval=-r_limit, maxval=r_limit)
    distorted_image = tfa.image.rotate(distorted_image, angles=r)

    distorted_image = tf.squeeze(distorted_image)

    return distorted_image

def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels, dataset_type, is_training):
    """Preprocess an image.
    Args:
        image_buffer: string; encoded JPEG file.
        bbox: 3-D float Tensor; it contains the bounding boxes related to an
            image. Bounding box coordinates are in range [0, 1], arranged in
            order [ymin, xmin, ymax, xmax]. The Tensor is of shape
            [1, num_boxes, 4], where num_boxes is the number of bounding
            boxes related to the image.
        output_height: int; height of the image after preprocessing.
        output_width: int; width of the image after preprocessing.
        num_channels: int; depth of the image buffer for decoding.
        dataset_type: string; type of dataset.
        is_training: boolean; whether the input will be used for training.
    Returns:
        image: 3-D float Tensor; it contains an image. It is of
            size [H, W, C], where H is the image height, W is
            the image width, and C is the number of channels.
    """
    
    if ((dataset_type == 'train') and (is_training)):
        # For training data during training, apply random distortions for data augmentation
        image = distort_image(image_buffer, output_height, output_width, num_channels, bbox)
    else:
        # Decode and resize the input image
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR, antialias=False)
        image = tf.squeeze(image, [0])
    
    # Transform image values from range [0, 1], to [-1, 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image

def parse_record(raw_record, dataset_type, is_training,
                 img_size_y, img_size_x, dtype, adv_eval_data):
    """Parse a record containing a training example that corresponds to an image.
    Args:
        raw_record: string; serialized Example protocol buffer.
        dataset_type: string; type of dataset.
        is_training: boolean; whether the input will be used for training.
        img_size_y: int; image height in pixels.
        img_size_x: int; image width in pixels.
        dtype: string; data type to use for images/features.
        adv_eval_data: boolean; whether to include information for advanced
            evaluation in the input batches.
    Returns:
        batch: tuple; it contains the following entries:
            image: 3-D float Tensor; it contains an image. It is of
                size [H, W, C], where H is the image height, W is
                the image width, and C is the number of channels.
            label: int; numeric image label.
            img_filename (optional): string; the filename of an image.
            img_label_text (optional): string; the human-readable label of an image.
    """

    # Parse Example protocol buffer
    if (not adv_eval_data):
        image_buffer, label = parse_example_proto(raw_record, adv_eval_data)
    else:
        (image_buffer, label,
        img_filename, img_label_text) = parse_example_proto(raw_record, adv_eval_data)

    # Pre-process image
    bbox = tf.constant([[[0., 0., 1., 1.]]], dtype=tf.float32)
    image = preprocess_image(image_buffer=image_buffer,
                             bbox=bbox,
                             output_height=img_size_y,
                             output_width=img_size_x,
                             num_channels=NUM_CHANNELS,
                             dataset_type=dataset_type,
                             is_training=is_training)

    # Return batch
    if (not adv_eval_data):
        batch  = (image, label)
    else:
        batch  = (image, label, img_filename, img_label_text)
        
    return batch

def process_record_dataset(dataset,
                           dataset_type,
                           is_training,
                           batch_size,
                           img_size_y,
                           img_size_x,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=-1,
                           dtype=tf.float32,
                           drop_remainder=False,
                           adv_eval_data=False):
    """Create input dataset from raw records.
    Args:
        dataset: tf dataset; dataset with raw records.
        dataset_type: string; type of dataset.
        is_training: boolean; whether the input will be used for training.
        batch_size: int; number of samples per batch (global, not per replica).
        img_size_y: int; image height in pixels.
        img_size_x: int; image width in pixels.
        shuffle_buffer: int; buffer size to use when shuffling records. A larger
            value results in higher randomness, but a smaller one reduces startup
            time and uses less memory.
        parse_record_fn: function; function that processes raw records.
        num_epochs: int; number of times to repeat the dataset.
        dtype: string; data type to use for images/features.
        drop_remainder: boolean; whether to drop the remainder of the
            batches. If True, the batch dimension will be static.
        adv_eval_data: boolean; whether to include information for advanced
            evaluation in the input batches.
    Returns:
        dataset: tf dataset; iterable input dataset.
    """

    # Shuffle records before repeating, to respect epoch boundaries
    if (is_training):
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    
    # Repeat dataset for the number of epochs to train
    if (num_epochs < 1):
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)

    # Parse raw records
    dataset = dataset.map(lambda value: parse_record_fn(value, dataset_type, is_training,
                                                        img_size_y, img_size_x, dtype, adv_eval_data),
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. Prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def input_fn(dataset_type,
             is_training,
             data_dir,
             batch_size,
             img_size_y,
             img_size_x,
             num_epochs=-1,
             dtype=tf.float32,
             parse_record_fn=parse_record,
             drop_remainder=False,
             filenames=None,
             adv_eval_data=False):
    """Prepare input batches.
    Args:
        dataset_type: string; type of dataset.
        is_training: boolean; whether the input will be used for training.
        data_dir: string; directory containing the input data.
        batch_size: int; number of samples per batch (global, not per replica).
        img_size_y: int; image height in pixels.
        img_size_x: int; image width in pixels.
        num_epochs: int; number of times to repeat the dataset.
        dtype: string; data type to use for images/features.
        parse_record_fn: function; function that processes raw records.
        drop_remainder: boolean; indicates whether to drop the remainder of the
            batches. If True, the batch dimension will be static.
        filenames: list of strings; it contains paths to TFRecords.
        adv_eval_data: boolean; whether to include information for advanced
            evaluation in the input batches.
    Returns:
        input_dataset: tf dataset; iterable input dataset.
    """

    # Get TFRecords paths
    if (filenames is None):
        filenames = get_filenames(dataset_type, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Shuffle input files
    if (is_training):
        if (dataset_type == 'train'):
            dataset = dataset.shuffle(buffer_size=TRAIN_SHARDS_NUM)
        elif (dataset_type == 'validation'):
            dataset = dataset.shuffle(buffer_size=VAL_SHARDS_NUM)

    # Process input files concurrently
    dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Process TFRecords
    input_dataset = process_record_dataset(dataset=dataset,
                                           dataset_type=dataset_type,
                                           is_training=is_training,
                                           batch_size=batch_size,
                                           img_size_y=img_size_y,
                                           img_size_x=img_size_x,
                                           shuffle_buffer=_SHUFFLE_BUFFER,
                                           parse_record_fn=parse_record_fn,
                                           num_epochs=num_epochs,
                                           dtype=dtype,
                                           drop_remainder=drop_remainder,
                                           adv_eval_data=adv_eval_data)

    return input_dataset
