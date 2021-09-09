"""Convert NABirds images to TFRecords. Raw data can be downloaded here
https://dl.allaboutbirds.org/nabirds, and are assumed to reside in the following
directory structure:
    images/0295/01f53d6bf5e449438d2bb79e0854bca4.jpg
    images/0296/069519c379574fb285d7bb920443ea89.jpg
    ...
Metadata files that can be downloaded with the raw data, are utilized as well.
In particular, the following files are used: images.txt, train_test_split.txt,
sizes.txt, classes.txt, and image_class_labels.txt.

images.txt contains the list of image file names, with each line corresponding to one image.
The content of the file is expected to be as follows:
<image_id> <image_name>
where image_id is a numeric identifier for each image in the dataset, and image_name is the path
to the corresponding image file.
An example line is the following:
0000139e-21dc-4d0c-bfe1-4cae3c85c829 0817/0000139e21dc4d0cbfe14cae3c85c829.jpg

train_test_split.txt contains the suggested training/validation split, with each line corresponding
to one image. The content of the file is expected to be as follows:
<image_id> <is_training_image>
where image_id is a unique identifier for each image in the dataset (same as in images.txt), and
is_training_image takes either value 1 or 0, denoting that the file is in the training or the validation
set, respectively.
An example line is the following:
0000139e-21dc-4d0c-bfe1-4cae3c85c829 0

sizes.txt contains the spatial dimensions of each image, with each line corresponding to one image.
The content of the file is expected to be as follows:
<image_id> <width> <height>
where image_id is a unique identifier for each image in the dataset (same as in images.txt), width
is the width of the corresponding image in pixels, and height is the height of the image in pixels.
An example line is the following:
0000139e-21dc-4d0c-bfe1-4cae3c85c829 296 341

classes.txt contains the list of human-readable labels (not all of them are represented in the image data),
with each line corresponding to a different label.
The content of the file is expected to be as follows:
<class_id> <class_name>
where class_id is a unique numeric identifier for each class, and class_name is the corresponding human-readable label.
An example line is the following:
37 Barn Owl

image_class_labels.txt contains the mapping between images and ground truth labels.
The content of the file is expected to be as follows:
<image_id> <class_id>
where mage_id is a unique identifier for each image in the dataset (same as in images.txt), and class_id is a unique
numeric identifier for each class.
An example line is the following:
0000139e-21dc-4d0c-bfe1-4cae3c85c829 817
"""

from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
import os
import random
import sys
import threading
import scipy.io
import pickle
import six

import numpy as np
import tensorflow as tf



parser = argparse.ArgumentParser()

parser.add_argument('--data_directory', type=str, default='/images/', help='Directory with raw image data.')
parser.add_argument('--root_directory', type=str, default='/NABirds/data/', help='Directory with metadata files.')
parser.add_argument('--output_directory', type=str, default='/TFRecords/', help='Output data directory.')
parser.add_argument('--image_ids_struct_path', type=str, default=None, help='Path to txt file with pyhton dictionary that contains metadata needed for the creation of TFRecord files.')

parser.add_argument('--train_shards', type=int, default=16, help='Number of shards in training TFRecord files.')
parser.add_argument('--validation_shards', type=int, default=16, help='Number of shards in validation TFRecord files.')
parser.add_argument('--num_threads', type=int, default=16, help='Number of threads to parallelize processing.')

FLAGS = parser.parse_args()

IMAGE_IDS_STRUCT_FNAME = 'image_ids_struct.txt'

def create_image_ids_struct():
    """Create dictionary with information about the NAbirds dataset.
       It includes image filenames, ground truth numeric labels,
       human-readable labels, image spatial dimensions, and indicators
       that distinguish between the training and validation splits.
    Args:
        -
    Returns:
        -
    """

    image_ids = {}

    fname = 'images.txt'
    with open(os.path.join(FLAGS.root_directory, fname)) as f:
        for line in f:
            tokens = line.strip().split()
            image_ids[tokens[0]] = {}
            image_ids[tokens[0]]['image_name'] = tokens[1]

    fname = 'train_test_split.txt'
    with open(os.path.join(FLAGS.root_directory, fname)) as f:
        for line in f:
            tokens = line.strip().split()
            image_ids[tokens[0]]['is_training_image'] = int(tokens[1])

    fname = 'sizes.txt'
    with open(os.path.join(FLAGS.root_directory, fname)) as f:
        for line in f:
            tokens = line.strip().split()
            image_ids[tokens[0]]['height'] = int(tokens[2])
            image_ids[tokens[0]]['width'] = int(tokens[1])

    class_ids = {}
    fname = 'classes.txt'
    with open(os.path.join(FLAGS.root_directory, fname)) as f:
        for line in f:
            tokens = line.strip().split()
            class_ids[tokens[0]] = tokens[1]

    fname = 'image_class_labels.txt'
    with open(os.path.join(FLAGS.root_directory, fname)) as f:
        for line in f:
            tokens = line.strip().split()
            image_ids[tokens[0]]['class_id'] = tokens[1]
            image_ids[tokens[0]]['class_name'] = class_ids[tokens[1]]

    labels = {}
    label_id = 0
    for e in image_ids:
        if (image_ids[e]['class_id'] not in labels):
            labels[image_ids[e]['class_id']] = label_id
            image_ids[e]['label'] = label_id
            label_id += 1
        else:
            image_ids[e]['label'] = labels[image_ids[e]['class_id']]

    with open(os.path.join(FLAGS.root_directory, IMAGE_IDS_STRUCT_FNAME), "wb") as fp:
        pickle.dump(image_ids, fp)

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

def _convert_to_example(filename, image_buffer, label, human_label, height, width):
    """Build an Example proto for an image.
    Args:
        filename: string; path to image file.
        image_buffer: string; JPEG encoded image.
        label: int; numeric ground truth label.
        human_label: string; human-readable label.
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
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(human_label),
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
        image = tf.io.decode_image(image_data, channels=3)
    except:
        print("Oops! %s." %filename)

    # Assert that the image has the appropriate dimensions
    assert (image.shape[2] == 3)
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    image_data = tf.io.encode_jpeg(image, format='rgb', quality=100)

    return image_data, height, width

def _process_image_files_batch(thread_index, ranges, name, filenames,
                               labels, human_labels, num_shards):
    """Execute 1 thread that processes images and saves them as TFRecords
       of Example protos.
    Args:
        thread_index: int; unique thread identifier.
        ranges: list of ints; it contains the range of images to
            process.
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        labels: list of ints; it contains numeric labels.
        human_labels: list of strings; it contains human-readable labels.
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
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        # Process each file for a shard
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            human_label = human_labels[i]

            # Process an image
            image_buffer, height, width = _process_image(filename)

            # Create an Example proto
            example = _convert_to_example(filename, image_buffer, label,
                                          human_label, height, width)
            
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

def _process_image_files(name, filenames, labels, human_labels, num_shards):
    """Process images and save them as TFRecords of Example protos.
    Args:
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        labels: list of ints; it contains numeric labels.
        human_labels: list of strings; it contains human-readable labels.
        num_shards: int; number of shards.
    Returns:
        -
    """

    assert len(filenames) == len(labels) == len(human_labels)

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
                labels, human_labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %(datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(name, data_dir):
    """Build lists of images file paths, numeric labels, and
       human-readable labels.
    Args:
        name: string; unique identifier specifying the data set.
        data_dir: string; path to data set.
    Returns:
        filenames: list of strings; it contains paths to image files.
        labels: list of ints; it contains numeric labels.
        human_labels: list of strings; it contains human-readable labels.
    """

    data_type_bool = int(name == 'train')

    with open(os.path.join(FLAGS.root_directory, IMAGE_IDS_STRUCT_FNAME), "rb") as fp:
        image_ids = pickle.load(fp)
    
    # Iterate over the image files
    filenames = []
    labels = []
    human_labels = []
    label_num = 0
    for e in image_ids:
        im_struct = image_ids[e]
        if (im_struct['is_training_image'] == data_type_bool):
            filenames.append(os.path.join(data_dir, im_struct['image_name']))
            
            if (im_struct['label'] not in labels):
                label_num += 1
            labels.append(im_struct['label'])
            human_labels.append(im_struct['class_name'])

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to labels in the
    # saved TFRecord files. Make the randomization repeatable
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    human_labels = [human_labels[i] for i in shuffled_index]

    print('Found %d .jpg files across %d labels inside %s.' %(len(filenames), label_num, data_dir))
    sys.stdout.flush()

    return filenames, labels, human_labels

def _process_dataset(name, directory, num_shards):
    """Process a complete data set and save it in TFRecords.
    Args:
        name: string; unique identifier specifying the data set.
        directory: string; path to data set.
        num_shards: int; number of shards.
    Returns:
        -
    """

    filenames, labels, human_labels = _find_image_files(name, directory)
    _process_image_files(name, filenames, labels, human_labels, num_shards)

def main(argv=None):
    """Convert NABirds training and validation images to TFRecords.
    Args:
        -
    Returns:
        -
    """

    assert not FLAGS.train_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.train_shards_num')
    assert not FLAGS.validation_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.dev_shards_num')

    if (not os.path.isdir(FLAGS.output_directory)):
        os.makedirs(FLAGS.output_directory)
    print('Saving results to %s' % FLAGS.output_directory)
    sys.stdout.flush()

    # Create dictionary with metadata information
    if (not FLAGS.image_ids_struct_path):
        create_image_ids_struct()

    # Create TFRecords
    _process_dataset('validation', FLAGS.data_directory, FLAGS.validation_shards)
    _process_dataset('train', FLAGS.data_directory, FLAGS.train_shards)

if __name__ == '__main__':
    main()
