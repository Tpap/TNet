"""Convert fMoW images to TFRecords.
Raw fMoW data can be downloaded here https://github.com/fMoW/dataset.
The current script utilizes the rgb version of fMoW, and not the full version.
fMoW data are split in training, validation and test sets. After download, for
the training and validations sets, jpeg and json files are expected to reside
in the following directory structure:
    /train/airport/airport_0/airport_0_0_rgb.jpg
    /train/airport/airport_0/airport_0_0_rgb.json
    ...

    /val/airport/airport_0/airport_0_0_rgb.jpg
    /val/airport/airport_0/airport_0_0_rgb.json
    ...

For the test set, jpeg and json files are expected to reside
in the following directory structure:
    /test/0011978/0011978_0_rgb.jpg
    /test/0011978/0011978_0_rgb.json
    ...

Test set directory structure doesn't reveal the labels of the images, because
it was initially realeased in the context of an IARPA challenge (https://www.iarpa.gov/challenges/fmow.html).
However, given that the challenge is over, test set annotations are available
for download with the rest of the data here https://github.com/fMoW/dataset.
After downloding the ground truth test data, they consist of json files that
reside in the following directory structure:
    /test_gt/airport/airport_0/airport_0_0_rgb.json
    /test_gt/airport/airport_0/airport_0_1_rgb.json
    ...

The additional test_gt_mapping.json file is provided to establish a correspondance
between the annotations under folder test_gt, and the images under folder test. To
this end, we provide match_test_gt.py script, which organizes jpeg and json files
for the test set, in the following directory structure:
    /test_matched_with_gt/airport/airport_0/airport_0_0_rgb.jpeg
    /test_matched_with_gt/airport/airport_0/airport_0_0_rgb.json
    ...

Given the desired uniformity in the directory organization of the training,
validation, and test sets is established, the current script converts image
data to TFRecord files. Each record within a TFRecord file is a serialized
Example proto.
"""

from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
import os
import random
import sys
import threading
import json

import numpy as np
import six
import tensorflow as tf



parser = argparse.ArgumentParser()

parser.add_argument('--train_directory', type=str, default='/train/', help='Training data directory.')
parser.add_argument('--validation_directory', type=str, default='/val/', help='Validation data directory.')
parser.add_argument('--test_directory', type=str, default='/test_matched_with_gt/', help='Test data directory.')
parser.add_argument('--output_directory', type=str, default='/TFRecords/', help='Output data directory.')

parser.add_argument('--train_shards', type=int, default=512, help='Number of shards in training TFRecord files.')
parser.add_argument('--validation_shards', type=int, default=128, help='Number of shards in validation TFRecord files.')
parser.add_argument('--test_shards', type=int, default=128, help='Number of shards in test TFRecord files.')
parser.add_argument('--num_threads', type=int, default=32, help='Number of threads to parallelize processing.')
parser.add_argument('--maximum_min_dim', type=int, default=1000, help='Maximum size allowed for the smallest image spatial dimension.')
parser.add_argument('--cropped_data', action='store_true', help='Whether the provided data are cropped acoording to bounding boxes annotations.')

FLAGS = parser.parse_args()

category_names = ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site',
                  'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course',
                  'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building',
                  'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement',
                  'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool',
                  'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

def clip_0_1(x):
    """Clip given float number within [0, 1] range.
    Args:
        x: float; value to clip.
    Returns:
        x: float; value within [0, 1] range.
    """

    if (x < 0.):
        x = 0.
    elif (x > 1.0):
        x = 1.0

    return x

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

def _convert_to_example(filename, image_buffer, label,
                        category, bbox, height, width):
    """Build an Example proto for an image.
    Args:
        filename: string; path to image file.
        image_buffer: string; JPEG encoded image.
        label: int; numeric ground truth label.
        category: string; human-readable label.
        bbox: list; it contains coordinates of bounding boxes.
        height: int; image height in pixels.
        width: int; image width in pixels.
    Returns:
        example: example proto; it contains the following fields:
            image/height: int; image height in pixels.
            image/width: int; image width in pixels.
            image/colorspace: string; colorspace, always 'RGB'.
            image/channels: int; number of channels, always 3.
            image/class/label: int; index of a classification label in range [0, 61].
            image/class/text: string; human-readable label.
            image/object/bbox/ymin: list of ints; denotes the minimum vertical pixel value
                of a bounding box, in proportion to the image height. It takes values in
                [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/xmin: list of ints; denotes the minimum horizontal pixel
                value of a bounding box, in proportion to the image width. It takes values
                in [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/ymax: list of ints; denotes the maximum vertical pixel value
                of a bounding box, in proportion to the image height. It takes values in
                [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/xmax: list of ints; denotes the maximum horizontal pixel
                value of a bounding box, in proportion to the image width. It takes values
                in [0, 1]. Each entry in the list corresponds to a different bounding box.
            image/object/bbox/label: int; index of a classification label. It is always
                identical to the corresponding image label.
            image/format: string; image format, always 'JPEG'.
            image/filename: string; image file basename.
            image/encoded: string; JPEG encoded image.
    """

    b_ymin = []
    b_xmin = []
    b_ymax = []
    b_xmax = []
    for b in bbox:
        assert len(b) == 4
        [l.append(point) for l, point in zip([b_ymin, b_xmin, b_ymax, b_xmax], b)]

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(category),
        'image/object/bbox/ymin': _float_feature(b_ymin),
        'image/object/bbox/xmin': _float_feature(b_xmin),
        'image/object/bbox/ymax': _float_feature(b_ymax),
        'image/object/bbox/xmax': _float_feature(b_xmax),
        'image/object/bbox/label': _int64_feature([label] * len(b_xmin)),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)
    }))
    
    return example

def _process_image(filename, img_size):
    """Process a single image file.
    Args:
        filename: string; path to an image file.
        img_sizes: tuple of ints; it contains the spatial
            dimensions of an image.
    Returns:
        image_buffer: string; JPEG encoded image.
        height: int; image height in pixels.
        width: int; image width in pixels.
    """

    # Read image file
    image_data = tf.io.read_file(filename)

    # Calculate decoding ratio to avoid overflow due to huge images
    min_dim = min(img_size)
    if (min_dim > 8 * FLAGS.maximum_min_dim):
        ratio = 8
    elif (min_dim > 4 * FLAGS.maximum_min_dim):
        ratio = 4
    elif (min_dim > 2 * FLAGS.maximum_min_dim):
        ratio = 2
    else:
        ratio = 1
    image = tf.io.decode_jpeg(image_data, ratio=ratio, channels=3)

    # Ensure smallest image dimension does not exceed FLAGS.maximum_min_dim
    height = image.shape[0]
    width = image.shape[1]
    min_dim = min([height, width])
    if (min_dim > FLAGS.maximum_min_dim):
        if (height == min_dim):
            new_height = FLAGS.maximum_min_dim
            new_width = np.ceil(float(new_height) * (float(width)/float(height)))
        else:
            new_width = FLAGS.maximum_min_dim
            new_height = np.ceil(float(new_width) * (float(height)/float(width)))

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(tf.expand_dims(image, axis=0), size=[int(new_height), int(new_width)],
                                preserve_aspect_ratio=True, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.squeeze(image)
  
    # Assert that the image has the appropriate dimensions
    assert (len(image.shape) == 3)
    assert (image.shape[2] == 3)
    height = image.shape[0]
    width = image.shape[1]
    assert ((height <= FLAGS.maximum_min_dim) or (width <= FLAGS.maximum_min_dim))
  
    # Encode the image, if it was processed
    if ((min_dim > FLAGS.maximum_min_dim) or (ratio != 1)):
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image_data = tf.image.encode_jpeg(image, format='rgb', quality=100)

    return image_data, height, width

def _process_image_files_batch(thread_index, ranges, name, filenames,
                               labels, categories, bboxes, img_sizes, num_shards):
    """Execute 1 thread that processes images and saves them as TFRecords
       of Example protos.
    Args:
        thread_index: int; unique thread identifier.
        ranges: list of ints; it contains the range of images to
            process.
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        labels: list of ints; it contains numeric ground truth labels.
        categories: list of strings; it contains human-readable ground
            truth labels.
        bboxes: list; it contains bounding boxes for each image.
        img_sizes: list of tuples; each tuple contains the spatial
            dimensions of an image.
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
            category = categories[i]
            bbox = bboxes[i]
            img_size = img_sizes[i]

            # Process an image
            image_buffer, height, width = _process_image(filename, img_size)

            # Create an Example proto
            example = _convert_to_example(filename, image_buffer, label,
                                          category, bbox, height, width)
            
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

def _process_image_files(name, filenames, labels, categories, bboxes, img_sizes, num_shards):
    """Process images and save them as TFRecords of Example protos.
    Args:
        name: string; unique identifier specifying the data set.
        filenames: list of strings; it contains paths to image files.
        labels: list of ints; it contains numeric ground truth labels.
        categories: list of strings; it contains human-readable ground
            truth labels.
        bboxes: list; it contains bounding boxes for each image.
        img_sizes: list of tuples; each tuple contains the spatial
            dimensions of an image.
        num_shards: int; number of shards.
    Returns:
        -
    """

    assert len(filenames) == len(labels) == len(categories) == len(bboxes) == len(img_sizes)

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
                labels, categories, bboxes, img_sizes, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %(datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir):
    """Build lists of all images file paths, numeric labels, and
       human-readable labels in a data set.
    Args:
        data_dir: string; path to data set.
    Returns:
        filenames: list of strings; it contains paths to image files.
        labels: list of ints; it contains numeric ground truth labels.
        categories: list of strings; it contains human-readable ground
            truth labels.
    """

    print('Determining list of input files and labels from %s.' % data_dir)
    sys.stdout.flush()
    filenames = []
    labels = []
    categories = []

    # Construct the list of JPEG files and labels
    label_index = 0
    for category in category_names:
        if (not FLAGS.cropped_data):
            jpeg_file_path = os.path.join(data_dir, category, '*', category + '_*_rgb.jpg')
        else:
            jpeg_file_path = os.path.join(data_dir, category, '*', '*', category + '_*_rgb.jpg')
        matching_files = tf.io.gfile.glob(jpeg_file_path)

        filenames.extend(matching_files)
        labels.extend([label_index] * len(matching_files))
        categories.extend([category] * len(matching_files))

        if (not (label_index % 10)):
            print('Finished finding files in %d of %d classes.' % (label_index, len(category_names)))
            sys.stdout.flush()
        
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to labels in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    categories = [categories[i] for i in shuffled_index]

    print('Found %d .jpg files across %d labels inside %s.'
          %(len(filenames), len(category_names), data_dir))
    sys.stdout.flush()

    return filenames, labels, categories

def _find_image_bounding_boxes(filenames, categories):
    """Find the bounding boxes for a given image file.
    Args:
        filenames: list of strings; it contains paths to image files.
        categories: list of strings; it contains human-readable ground
            truth labels.
    Returns:
        bboxes: list; it contains bounding boxes for each image.
        img_sizes: list of tuples; each tuple contains the spatial
            dimensions of an image.
    """

    num_image_bbox = 0
    bbox_num = 0
    bboxes = []
    img_sizes = []
    # Iterate over image files
    for i in range(len(filenames)):
        f = filenames[i]
        category = categories[i]

        f_json = f.replace('.jpg', '.json')
        jsonData = json.load(open(f_json))

        json_bboxes = jsonData['bounding_boxes']
        if not isinstance(json_bboxes, list):
            json_bboxes = [json_bboxes]
        
        h = float(jsonData['img_height'])
        w = float(jsonData['img_width'])
        # Iterate over available bounding boxes for an image file
        bb_lst = []
        if (not FLAGS.cropped_data):
            for bb in json_bboxes:
                if ((bb['category'] != category) or (bb['ID'] == -1)):
                    continue
                # Change box format from [xmin, ymin, width, height] to
                # [ymin, xmin, ymax, xmax] with values as image size percentages
                bb['box'] = [float(e) for e in bb['box']]
                ymin = bb['box'][1] / h
                ymin = clip_0_1(ymin)
                xmin = bb['box'][0] / w
                xmin = clip_0_1(xmin)
                ymax = (bb['box'][1] + bb['box'][3]) / h
                ymax = clip_0_1(ymax)
                xmax = (bb['box'][0] + bb['box'][2]) / w
                xmax = clip_0_1(xmax)
                bb_lst.append([ymin, xmin, ymax, xmax])
            
            if (len(bb_lst) > 0): 
                num_image_bbox += 1
            bbox_num += len(bb_lst)
        else:
            # Cropped images result from crop_fMoW.py,
            # and bounding boxes are standarized
            assert len(jsonData['bounding_boxes']) == 1
            
            bb = jsonData['bounding_boxes'][0]
            assert bb['category'] == category
            
            box = bb['box']
            assert ((box[0] == 0.) and (box[1] == 0.) and (box[2] == 1.0) and (box[3] == 1.0))
            
            bb_lst.append(box)
            num_image_bbox += 1
            bbox_num += 1
        
        bboxes.append(bb_lst)
        img_sizes.append([h, w])

    print('Found %d images with %d bboxes out of %d images'
          %(num_image_bbox, bbox_num, len(filenames)))
    sys.stdout.flush()

    return bboxes, img_sizes

def _process_dataset(name, directory, num_shards):
    """Process a complete data set and save it in TFRecords.
    Args:
        name: string; unique identifier specifying the data set.
        directory: string; path to data set.
        num_shards: int; number of shards.
    Returns:
        -
    """

    filenames, labels, categories = _find_image_files(directory)
    bboxes, img_sizes = _find_image_bounding_boxes(filenames, categories)
    _process_image_files(name, filenames, labels, categories, bboxes, img_sizes, num_shards)

def main(argv=None):
    """Convert fMoW training, validation, and test images to TFRecords.
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

    # Create TFRecords
    _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards)
    _process_dataset('test', FLAGS.test_directory, FLAGS.test_shards)
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)

if __name__ == '__main__':
    main()