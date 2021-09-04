"""Convert ImageNet images to TFRecords.
Raw ImageNet data can be downloaded and converted to TFRecords by following the commands here
https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_imagenet.sh.
"download_and_convert_imagenet.sh" converts raw images to TFRecords by using the following script
https://github.com/tensorflow/models/blob/master/research/slim/datasets/build_imagenet_data.py.
The current script converts raw images to TFRecords, similar to "build_imagenet_data.py", but
it is implemented in TF 2.0. So, you can use "download_and_convert_imagenet.sh" to download raw data,
and then use the current script to convert them to TFRecords file format with Example protos.

After downlading raw ImageNet data according to download_and_convert_imagenet.sh,
they are expected to reside in JPEG files located in the following directory structure:
    data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
    data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
    ...
where 'n01440764' is the unique synset label associated with the corresponding images.
The training data set consists of 1000 sub-directories (i.e. labels)
for a total of 1,281,167M JPEG images.
The evaluation data set consists of 1000 sub-directories (i.e. labels)
for a total of 50,000 JPEG images.
You should also expect the bounding box annotations to be converted from XML files
into a single CSV, "imagenet_2012_bounding_boxes.csv"

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.
Each record within a TFRecord file is a serialized Example proto.
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


# python create_tfrecords_imagenet.py --output_directory /scratch/ap4094/TFRecords_2/ --labels_file '/scratch/ap4094/datasets/imagenet/data/imagenet_lsvrc_2015_synsets.txt' --imagenet_metadata_file '/scratch/ap4094/datasets/imagenet/data/imagenet_metadata.txt' --bounding_box_file '/scratch/ap4094/datasets/imagenet/data/imagenet_2012_bounding_boxes.csv'

parser = argparse.ArgumentParser()

parser.add_argument('--train_directory', type=str, default='/train/', help='Training data directory.')
parser.add_argument('--validation_directory', type=str, default='/validation/', help='Validation data directory.')
parser.add_argument('--output_directory', type=str, default='/TFRecords/', help='Output data directory.')

parser.add_argument('--train_shards', type=int, default=1024, help='Number of shards in training TFRecord files.')
parser.add_argument('--validation_shards', type=int, default=128, help='Number of shards in validation TFRecord files.')
parser.add_argument('--num_threads', type=int, default=32, help='Number of threads to parallelize processing.')

# The labels file contains a list of valid labels. It can be downloaded
# here https://github.com/tensorflow/models/blob/master/research/slim/datasets/
# The content of the file is expected to be as follows:
#   n01440764
#   n01443537
#   n01484850
#   ...
#
# where each line corresponds to a label expressed as a synset.
parser.add_argument('--labels_file', type=str, default='/imagenet_lsvrc_2015_synsets.txt', help='File with list of synsets.')

# This file contains a map from synsets to human-readable labels. It can be downloaded
# here https://github.com/tensorflow/models/blob/master/research/slim/datasets/
# The content of the file is expected to be as follows:
#   n02119247    black fox
#   n02119359    silver fox
#   n02119477    red fox, Vulpes fulva
#   ...
# 
parser.add_argument('--imagenet_metadata_file', type=str, default='/imagenet_metadata.txt', help='File with mapping from synsets to human-readable labels.')

# This file is the output of process_bounding_box.py, which can be found
# here https://github.com/tensorflow/models/blob/master/research/slim/datasets/
# "imagenet_2012_bounding_boxes.csv" is expected to be created by following commands
# provided in "download_and_convert_imagenet.sh".
# The content of the file is expected to be as follows:
#   n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
#   ...
#
# where each line corresponds to one bounding box annotation associated
# with an image. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an image file.
parser.add_argument('--bounding_box_file', type=str, default='/imagenet_2012_bounding_boxes.csv', help='File with list of bounding boxes.')

FLAGS = parser.parse_args()


def _int64_feature(value):
    """Insert int features into Example proto.
    Args:
        value: int or list of ints; features to insert
            in Example proto.
    Returns:
        feature: example proto; it contains a list of ints.
    """
    
    if (not isinstance(value, list)):
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

    if (not isinstance(value, list)):
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

def _convert_to_example(filename, image_buffer, label, synset, human, bbox,
                        height, width):
    """Build an Example proto for an image.
    Args:
        filename: string; path to image file.
        image_buffer: string; JPEG encoded image.
        label: int; numeric ground truth label.
        synset: string; WordNet ID (synset).
        human: string; human-readable label.
        bbox: list; it contains coordinates of bounding boxes.
        height: int; image height in pixels.
        width: int; image width in pixels.
    Returns:
        example: example proto; it contains the following fields:
            image/height: int; image height in pixels.
            image/width: int; image width in pixels.
            image/colorspace: string; colorspace, always 'RGB'.
            image/channels: int; number of channels, always 3.
            image/class/label: int; index of a classification label in range [1, 1000].
            image/class/synset: string; unique label ID, e.g., 'n01440764'.
            image/class/text: string; human-readable label, e.g., 'red fox, Vulpes vulpes'.
            image/object/bbox/xmin: list of ints; denotes the minimum horizontal pixel
                value of a bounding box, in proportion to the image width. It takes values
                in [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/xmax: list of ints; denotes the maximum horizontal pixel
                value of a bounding box, in proportion to the image width. It takes values
                in [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/ymin: list of ints; denotes the minimum vertical pixel value
                of a bounding box, in proportion to the image height. It takes values in
                [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/ymax: list of ints; denotes the maximum vertical pixel value
                of a bounding box, in proportion to the image height. It takes values in
                [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/label: int; index of a classification label. It is always
                identical to the corresponding image label.
            image/format: string; image format, always 'JPEG'.
            image/filename: string; image file basename, e.g., 'n01440764_10026.JPEG'
                or 'ILSVRC2012_val_00000293.JPEG'.
            image/encoded: string; JPEG encoded image.
    """

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bbox:
        assert len(b) == 4
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/class/text': _bytes_feature(human),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature([label] * len(xmin)),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)
    }))
    
    return example

def _is_png(filename):
    """Determine whether image file is in PNG format.
    Args:
        filename: string; path to an image file.
    Returns:
        is_png: boolean; indicates whether image file is
            in PNG format.
    """

    # File list from:
    # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU

    is_png = 'n02105855_2933.JPEG' in filename

    return is_png

def _is_cmyk(filename):
    """Determine whether image file is in CMYK color space.
    Args:
        filename: string; path to an image file.
    Returns:
        is_cmyk: boolean; indicates whether image file is in
            CMYK color space.
    """
    # File list from: https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                'n07583066_647.JPEG', 'n13037406_4650.JPEG']
    
    is_cmyk = filename.split('/')[-1] in blacklist

    return is_cmyk

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

    # Encode all images to JPEG
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image = tf.io.decode_png(image_data, channels=3)
        image_data = tf.io.encode_jpeg(image, format='rgb', quality=100)
    elif _is_cmyk(filename):
        print('Converting CMYK to RGB for %s' % filename)
        image = tf.io.decode_jpeg(image_data, channels=3)
        image_data = tf.io.encode_jpeg(image, format='rgb', quality=100)

    # Decode image
    try:
        image = tf.io.decode_jpeg(image_data, channels=3)
    except:
        print("Oops! %s." %filename)

    # Assert that the image has the appropriate dimensions
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _process_image_files_batch(thread_index, ranges, name, filenames,
                               synsets, labels, humans, bboxes, num_shards):
    """Execute 1 thread that processes images and saves them as TFRecords
       of Example protos.
    Args:
        thread_index: int; unique thread identifier.
        ranges: list of ints; it contains the range of images to
            process.
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        synsets: list of strings; it contains WordNet IDs (synsets).
        labels: list of ints; it contains numeric ground truth labels.
        humans: list of strings; it contains human-readable labels.
        bboxes: list; it contains bounding boxes for each image.
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
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        # Process each file for a shard
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            synset = synsets[i]
            human = humans[i]
            bbox = bboxes[i]

            # Process an image
            image_buffer, height, width = _process_image(filename)

            # Create an Example proto
            example = _convert_to_example(filename, image_buffer, label,
                                          synset, human, bbox, height, width)
            
            # Write to TFRecord
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if (not (counter % 1000)):
                print('%s [thread %d]: Processed %d of %d images in thread batch.'
                      %(datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s'
              %(datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.'
          %(datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, synsets, labels, humans,
                         bboxes, num_shards):
    """Process images and save them as TFRecords of Example protos.
    Args:
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        synsets: list of strings; it contains WordNet IDs (synsets).
        labels: list of ints; it contains numeric ground truth labels.
        humans: list of strings; it contains human-readable labels.
        bboxes: list; it contains bounding boxes for each image.
        num_shards: int; number of shards.
    Returns:
        -
    """

    assert len(filenames) == len(synsets) == len(labels) == len(humans) == len(bboxes)

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
                synsets, labels, humans, bboxes, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all threads to terminate
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %(datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir, labels_file):
    """Build lists of all images file paths, synsets, and labels in
       a data set.
    Args:
        data_dir: string; path to data set.
        labels_file: string; path to file that contains image labels
            expressed as synsets.
    Returns:
        filenames: list of strings; it contains paths to image files.
        synsets: list of strings; it contains WordNet IDs (synsets).
        labels: list of ints; it contains numeric ground truth labels.
    """

    print('Determining list of input files and labels from %s.' % data_dir)
    sys.stdout.flush()
    challenge_synsets = [l.strip() for l in tf.io.gfile.GFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    synsets = []

    # Leave label index 0 empty as a background class
    label_index = 1

    # Construct the list of JPEG files and labels
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
        matching_files = tf.io.gfile.glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        if (not (label_index % 100)):
            print('Finished finding files in %d of %d classes.' % (label_index, len(challenge_synsets)))
            sys.stdout.flush()
        
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to labels in the
    # saved TFRecord files. Make the randomization repeatable
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.'
          %(len(filenames), len(challenge_synsets), data_dir))
    sys.stdout.flush()
    
    return filenames, synsets, labels

def _find_human_readable_labels(synsets, synset_to_human):
    """Build a list of human-readable labels.
    Args:
        synsets: list of strings; it contains WordNet IDs (synsets).
        synset_to_human: dictionary; it maps synsets to human-readable
            labels.
    Returns:
        humans: list of strings; it contains human-readable labels.
    """

    humans = []
    for s in synsets:
        assert s in synset_to_human, ('Failed to find: %s' % s)
        humans.append(synset_to_human[s])
    
    return humans

def _find_image_bounding_boxes(filenames, image_to_bboxes):
    """Find the bounding boxes of image files.
    Args:
        filenames: list of strings; it contains paths to image files.
        images_to_bboxes: dictionary; it maps image file names to
            bounding boxes.
    Returns:
        bboxes: list; it contains bounding boxes for each image.
    """
    
    num_image_bbox = 0
    bboxes = []
    for f in filenames:
        basename = os.path.basename(f)
        if (basename in image_to_bboxes):
            bboxes.append(image_to_bboxes[basename])
            num_image_bbox += 1
        else:
            bboxes.append([])
    
    print('Found %d images with bboxes out of %d images' % (num_image_bbox, len(filenames)))
    sys.stdout.flush()
    
    return bboxes

def _process_dataset(name, directory, num_shards, synset_to_human, image_to_bboxes):
    """Process a complete data set and save it in TFRecords.
    Args:
        name: string; unique identifier specifying the data set.
        directory: string; path to data set.
        num_shards: int; number of shards.
        synset_to_human: dictionary; it maps synsets to human-readable
            labels.
        images_to_bboxes: dictionary; it maps image file names to
            bounding boxes.
    Returns:
        -
    """

    filenames, synsets, labels = _find_image_files(directory, FLAGS.labels_file)
    humans = _find_human_readable_labels(synsets, synset_to_human)
    bboxes = _find_image_bounding_boxes(filenames, image_to_bboxes)
    _process_image_files(name, filenames, synsets, labels, humans, bboxes, num_shards)

def _build_synset_lookup(imagenet_metadata_file):
    """Build map from synsets to human-readable labels.
    Args:
        imagenet_metadata_file: string; path to file containing mapping from
            synsets to human-readable labels.
    Returns:
        synset_to_human: dictionary; it maps synsets to human-readable labels.
    """

    lines = tf.io.gfile.GFile(imagenet_metadata_file, 'r').readlines()
    synset_to_human = {}
    for l in lines:
        if (l):
            parts = l.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human
    
    return synset_to_human

def _build_bounding_box_lookup(bounding_box_file):
    """Build map from image file names to bounding boxes.
    Args:
        bounding_box_file: string; path to file with bounding boxes
            annotations.
    Returns:
        images_to_bboxes: dictionary; it maps image file names to
            bounding boxes.
    """

    lines = tf.io.gfile.GFile(bounding_box_file, 'r').readlines()
    images_to_bboxes = {}
    num_bbox = 0
    num_image = 0
    for l in lines:
        if (l):
            parts = l.split(',')
            assert len(parts) == 5, ('Failed to parse: %s' % l)
            filename = parts[0]
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])
            box = [xmin, ymin, xmax, ymax]

            if (filename not in images_to_bboxes):
                images_to_bboxes[filename] = []
                num_image += 1
            images_to_bboxes[filename].append(box)
            num_bbox += 1

    print('Successfully read %d bounding boxes across %d images.' % (num_bbox, num_image))
    sys.stdout.flush()

    return images_to_bboxes

def main(argv=None):
    """Convert ImageNet training and validation images to TFRecords.
    Args:
        -
    Returns:
        -
    """

    assert not FLAGS.train_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards')
  
    if (not os.path.isdir(FLAGS.output_directory)):
        os.makedirs(FLAGS.output_directory)
    print('Saving results to %s' % FLAGS.output_directory)
    sys.stdout.flush()

    # Build a map from synsets to human-readable labels
    synset_to_human = _build_synset_lookup(FLAGS.imagenet_metadata_file)
    # Build a map from image file names to bounding boxes
    image_to_bboxes = _build_bounding_box_lookup(FLAGS.bounding_box_file)

    # Create TFRecords
    _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards, synset_to_human, image_to_bboxes)
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards, synset_to_human, image_to_bboxes)

if __name__ == '__main__':
    main()