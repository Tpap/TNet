"""Convert Caltech-UCSD Birds-200-2011 images to TFRecords. Information about the training and
validation splits of the data reside in csv files, which are created by using create_csv_cub.py.
Raw data can be downloaded here http://www.vision.caltech.edu/visipedia/CUB-200-2011.html, and
are assumed to reside in the following directory structure:
    images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
    images/002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg
    ...
"""

from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
import os
import random
import sys
import threading
import scipy.io
import pandas as pd
import six

import numpy as np
import tensorflow as tf



# python create_tfrecords_cub.py --img_dir '/scratch/ap4094/datasets/caltech_UCSD_Birds_200_2011/data/CUB_200_2011/images/' --train_csv_path '/scratch/ap4094/CUB/train_anno.csv' --dev_csv_path '/scratch/ap4094/CUB/validation_anno.csv' --output_dir '/scratch/ap4094/CUB/TFRecords/'

parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', type=str, default='/CUB_200_2011/images/', help='Directory with raw image data.')
parser.add_argument('--train_csv_path', type=str, default='/CUB_200_2011/train_anno.csv', help='Path to csv file with information about the images in the training split.')
parser.add_argument('--dev_csv_path', type=str, default='/CUB_200_2011/validation_anno.csv', help='Path to csv file with information about the images in the validation split.')
parser.add_argument('--output_dir', type=str, default='/TFRecords/', help='Output data directory.')

parser.add_argument('--train_shards_num', type=int, default=16, help='Number of shards in training TFRecord files.')
parser.add_argument('--dev_shards_num', type=int, default=16, help='Number of shards in validation TFRecord files.')
parser.add_argument('--num_threads', type=int, default=16, help='Number of threads to parallelize processing.')

FLAGS = parser.parse_args()

def _int64_feature(value):
    """Insert int features into Example proto.
    Args:
        value: int or list of ints; features to insert
            in Example proto.
    Returns:
        feature: example proto; it contains a list of ints.
    """

    if ((not isinstance(value, list)) and (not isinstance(value, np.ndarray))):
        value = [value]
    
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    return feature

def _float_feature(value):
    """Insert float features into Example proto.
    Args:
        value: float or list of floats; features to insert
            in Example proto.
    Returns:
        feature: example proto; it contains a list of floats.
    """

    if ((not isinstance(value, list)) and (not isinstance(value, np.ndarray))):
        value = [value]
    
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=value))

    return feature

def _bytes_feature(value):
    """Insert byte features into Example proto.
    Args:
        value: string or list of strings; features to
            insert in Example proto.
    Returns:
        feature: example proto; it contains a byte list.
    """
    
    if (isinstance(value, type(tf.constant(0)))):
        value = value.numpy()
    if (six.PY3 and isinstance(value, six.text_type)):
        value = six.binary_type(value, encoding='utf-8') 
    
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    return feature

def _convert_to_example(filename, image_buffer, label_value, label_name, height, width):
    """Build an Example proto for an image.
    Args:
        filename: string; path to image file.
        image_buffer: string; JPEG encoded image.
        label_value: int; numeric ground truth label.
        label_name: string; human-readable label.
        height: int; image height in pixels.
        width: int; image width in pixels.
    Returns:
        example: example proto; it contains the following fields:
            image/height: int; image height in pixels.
            image/width: int; image width in pixels.
            image/colorspace: string; colorspace, always 'RGB'.
            image/channels: int; number of channels, always 3.
            image/class/label: int; index of a classification label in range [1, 200].
            image/class/text: string; human-readable label.
            image/format: string; image format, always 'JPEG'.
            image/filename: string; image file basename.
            image/encoded: string; JPEG encoded image.
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label_value),
        'image/class/text': _bytes_feature(label_name),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)
    }))

    return example

def _process_image(filename):
    """Process a single image file.
    Args:
        filename: string; path to an image file.
    Returns:
        image_buffer: string; JPEG encoded image.
        height: int; image height in pixels.
        width: int; image width in pixels.
    """

    # Read image file
    image_data = tf.io.read_file(filename)

    # Decode image
    try:
        image = tf.io.decode_jpeg(image_data, channels=3)
    except:
        print("Oops! %s." %filename)
        return -1
    
    # Assert that the image has the appropriate dimensions
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _process_image_files_batch(thread_index, ranges, name, filenames,
                               labels_values, labels_names, num_shards):
    """Execute 1 thread that processes images and saves them as TFRecords
       of Example protos.
    Args:
        thread_index: int; unique thread identifier.
        ranges: list of ints; it contains the range of images to
            process.
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        labels_values: list of ints; it contains numeric labels.
        labels_names: list of strings; it contains human-readable labels.
        num_shards: int; number of shards.
    Returns:
        -
    """

    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64)
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    # Generate each shard
    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.4d-of-%.4d' % (name, (shard+1), num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        # Process each file for a shard
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label_value = labels_values[i]
            label_name = labels_names[i]

            # Process an image
            image_buffer, height, width = _process_image(filename)

            # Create an Example proto
            example = _convert_to_example(filename, image_buffer, label_value,
                                          label_name, height, width)
            
            # Write to TFRecord
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if (not (counter % 1000)):
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.'
          %(datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, labels_values, labels_names, num_shards):
    """Process images and save them as TFRecords of Example protos.
    Args:
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        labels_values: list of ints; it contains numeric labels.
        labels_names: list of strings; it contains human-readable labels.
        num_shards: int; number of shards.
    Returns:
        -
    """

    assert len(filenames) == len(labels_values) == len(labels_names)

    # Break images into batches
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring threads' execution
    coord = tf.train.Coordinator()

    # Run threads
    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames,
                labels_values, labels_names, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(name, data_dir, csv_file):
    """Build lists of images file paths, numeric labels, and
       human-readable labels.
    Args:
        name: string; unique identifier specifying the data set.
        directory: string; path to data set.
        csv_file: string; path to csv file with information about
            the data.
    Returns:
        filenames: list of strings; it contains paths to image files.
        labels_values: list of ints; it contains numeric labels.
        labels_names: list of strings; it contains human-readable labels.
    """

    df = pd.read_csv(csv_file)
    filenames = df['fname'].tolist()
    filenames = [os.path.join(data_dir, f.lstrip('/')) for f in filenames]

    labels_values = df.to_numpy()[:, 1].astype(np.int).tolist()
    labels_names = df.to_numpy()[:, 2].tolist()

    print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(np.unique(labels_values)), data_dir))

    return filenames, labels_values, labels_names

def _process_dataset(name, directory, num_shards, csv_file):
    """Process a complete data set and save it in TFRecords.
    Args:
        name: string; unique identifier specifying the data set.
        directory: string; path to data set.
        num_shards: int; number of shards.
        csv_file: string; path to csv file with information about
            the data.
    Returns:
        -
    """

    filenames, labels_values, labels_names = _find_image_files(name, directory, csv_file)
    _process_image_files(name, filenames, labels_values, labels_names, num_shards)

def main(argv=None):
    """Convert Caltech-UCSD Birds-200-2011 training and validation
       images to TFRecords.
    Args:
        -
    Returns:
        -
    """

    assert not FLAGS.train_shards_num % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.train_shards_num')
    assert not FLAGS.dev_shards_num % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.dev_shards_num')

    if (not os.path.isdir(FLAGS.output_dir)):
        os.makedirs(FLAGS.output_dir)
    print('Saving results to %s' % FLAGS.output_dir)
    sys.stdout.flush()
    
    # Create TFRecords
    _process_dataset('validation', FLAGS.img_dir, FLAGS.dev_shards_num, FLAGS.dev_csv_path)
    _process_dataset('train', FLAGS.img_dir, FLAGS.train_shards_num, FLAGS.train_csv_path)

if __name__ == '__main__':
    main()
