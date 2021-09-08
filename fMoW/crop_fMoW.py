"""Crop fMoW images based on bounding box annotations.
"""

from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
import os
import random
import sys
import threading
import json
from multiprocessing import cpu_count
import cv2
import copy

import numpy as np
import six
import tensorflow as tf



parser = argparse.ArgumentParser()

parser.add_argument('--train_directory', type=str, default='/train/', help='Training data directory.')
parser.add_argument('--validation_directory', type=str, default='/val/', help='Validation data directory.')
parser.add_argument('--test_directory', type=str, default='/test_matched_with_gt/', help='Test data directory.')
parser.add_argument('--output_directory', type=str, default='/data_cropped/', help='Output data directory.')
parser.add_argument('--num_threads', type=int, default=16, help='Number of threads to parallelize processing.')

FLAGS = parser.parse_args()

category_names = ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site',
                  'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course',
                  'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building',
                  'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement',
                  'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool',
                  'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

def _process_image_files_batch(thread_index, ranges, file_paths, categories, outDir):
    """Execute 1 thread that processes images and saves crops according
       to bounding box annotations.
    Args:
        thread_index: int; unique thread identifier.
        ranges: list of ints; it contains the range of images to process.
        file_paths: list of strings; it contains paths to image files.
        categories: list of strings; it contains human-readable labels.
        outDir: string; directory to save output data.
    Returns:
        -
    """

    # Process each file
    files_in_thread = np.arange(ranges[thread_index][0], ranges[thread_index][1], dtype=int)
    img_num = 0
    bbox_num = 0
    for i in files_in_thread:
        img_num += 1
        f_src_img = file_paths[i]
        f_src_json = f_src_img.replace('.jpg', '.json')

        # Load image
        img = cv2.imread(f_src_img).astype(np.float32)

        # Load json file with image information
        jsonData = json.load(open(f_src_json))
        if not isinstance(jsonData['bounding_boxes'], list):
            jsonData['bounding_boxes'] = [jsonData['bounding_boxes']]

        label = categories[i]
        for bb in jsonData['bounding_boxes']:
            category = bb['category']
            if ((category != label) or (bb['ID'] == -1)):
                continue
            bbox_num += 1
            # Each bounding box is a list of 4 ints. The first two entries (box[0] and box[1])
            # are the coordinates in pixels of top left corner of the box (first the horizontal
            # and then the vertical coordinate), and the last two entries (box[2] and box[3])
            # are the width and the height of the box
            box = bb['box']

            # Ignore tiny boxes
            if box[2] <= 2 or box[3] <= 2:
                continue

            # Add margin around a bounding box for more contextual information.
            # The followed strategy is based on _process_file function from
            # https://github.com/fMoW/baseline/blob/master/code/data_ml_functions/dataFunctions.py
            contextMultWidth = 0.15
            contextMultHeight = 0.15

            wRatio = float(box[2]) / img.shape[1]
            hRatio = float(box[3]) / img.shape[0]
            
            if ((wRatio < 0.5) and (wRatio >= 0.4)):
                contextMultWidth = 0.2
            if ((wRatio < 0.4) and (wRatio >= 0.3)):
                contextMultWidth = 0.3
            if ((wRatio < 0.3) and (wRatio >= 0.2)):
                contextMultWidth = 0.5
            if ((wRatio < 0.2) and (wRatio >= 0.1)):
                contextMultWidth = 1
            if (wRatio < 0.1):
                contextMultWidth = 2
                
            if ((hRatio < 0.5) and (hRatio >= 0.4)):
                contextMultHeight = 0.2
            if ((hRatio < 0.4) and (hRatio >= 0.3)):
                contextMultHeight = 0.3
            if ((hRatio < 0.3) and (hRatio >= 0.2)):
                contextMultHeight = 0.5
            if ((hRatio < 0.2) and (hRatio >= 0.1)):
                contextMultHeight = 1
            if (hRatio < 0.1):
                contextMultHeight = 2
            
            widthBuffer = int((box[2] * contextMultWidth) / 2.0)
            heightBuffer = int((box[3] * contextMultHeight) / 2.0)

            r1 = box[1] - heightBuffer
            r2 = box[1] + box[3] + heightBuffer
            c1 = box[0] - widthBuffer
            c2 = box[0] + box[2] + widthBuffer

            if (r1 < 0):
                r1 = 0
            if (r2 > img.shape[0]):
                r2 = img.shape[0]
            if (c1 < 0):
                c1 = 0
            if (c2 > img.shape[1]):
                c2 = img.shape[1]

            if ((r1 >= r2) or (c1 >= c2)):
                continue

            subImg = img[r1:r2, c1:c2, :]

            jsonData_dst = copy.deepcopy(jsonData)
            bb['box'] = [0., 0., 1.0, 1.0]
            jsonData_dst['bounding_boxes'] = [bb]
            jsonData['img_height'] = r2 - r1 
            jsonData['img_width'] = c2 - c1

            # Determine output directory and save files
            slashes = [k for k, ltr in enumerate(f_src_img) if ltr == '/']
            outBaseName = '%s_%s' %(category, bb['ID'])
            currOut = os.path.join(outDir, f_src_img[(slashes[-3] + 1):slashes[-1]], outBaseName)

            if (not os.path.isdir(currOut)):
                try:
                    os.makedirs(currOut)
                except:
                    print("Directory already created.")

            f_name = os.path.basename(f_src_img)
            f_dst_img = os.path.join(currOut, f_name)
            f_dst_json = f_dst_img.replace('.jpg', '.json')

            cv2.imwrite(f_dst_img, subImg)
            json.dump(jsonData_dst, open(f_dst_json, 'w'))

    print('%s [thread %d]: Wrote %d images with %d bboxes.' %(datetime.now(), thread_index, img_num, bbox_num))
    sys.stdout.flush()

def _process_image_files(file_paths, categories, outDir):
    """Process images and save crops according to bounding box annotations.
    Args:
        file_paths: list of strings; it contains paths to image files.
        categories: list of strings; it contains human-readable labels.
        outDir: string; directory to save output data.
    Returns:
        -
    """

    # Break images into batches
    num_threads = FLAGS.num_threads
    spacing = np.linspace(0, len(file_paths), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Run threads
    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, file_paths, categories, outDir)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %(datetime.now(), len(file_paths)))
    sys.stdout.flush()

def _find_image_files(data_dir):
    """Build lists of all images file paths, synsets, and labels in
       a data set.
    Args:
        data_dir: string; path to data set.
    Returns:
        file_paths: list of strings; it contains paths to image files.
        categories: list of strings; it contains human-readable labels.
    """

    # Construct the lists of image files and categories
    print('Determining list of input files and categories from %s.' % data_dir)
    file_paths = []
    categories = []
    label_index = 1
    for category in category_names:
        jpeg_file_path = os.path.join(data_dir, category, '*', category + '_*_rgb.jpg')
        matching_files = tf.io.gfile.glob(jpeg_file_path)

        file_paths.extend(matching_files)
        categories.extend([category] * len(matching_files))

        if (not (label_index % 10)):
            print('Finished finding files in %d of %d classes.' %(label_index, len(category_names)))
        label_index += 1

    # Shuffle images to distribute large images to different threads
    # and avoid bottlenecks, since image size seems to be class specific
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]
    categories = [categories[i] for i in shuffled_index]

    print('Found %d .jpg files across %d labels inside %s.' %(len(file_paths), len(category_names), data_dir))
  
    return file_paths, categories

def _process_dataset(directory, outDir):
    """Process a complete data set (training, validation or test).
    Args:
        directory: string; path to data set.
        outDir: string; directory to save output data.
    Returns:
        -
    """

    file_paths, categories = _find_image_files(directory)
    _process_image_files(file_paths, categories, outDir)

def main(argv=None):
    """Crop fMoW training, validation and testing images
       based on bounding box annotations.
    Args:
        -
    Returns:
        -
    """

    _process_dataset(FLAGS.validation_directory, os.path.join(FLAGS.output_directory, 'val'))
    _process_dataset(FLAGS.test_directory, os.path.join(FLAGS.output_directory, 'test'))
    _process_dataset(FLAGS.train_directory, os.path.join(FLAGS.output_directory, 'train'))

if __name__ == '__main__':
  main()