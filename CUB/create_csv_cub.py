"""Create csv files for the training and validation splits of the Caltech-UCSD Birds-200-2011 dataset.
Each entry in the csv files containes the path to an image, its numeric label, and its human-readable
label. Raw data can be downloaded here http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
"""

from __future__ import absolute_import, division, print_function

import os
import random
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd



parser = argparse.ArgumentParser()

# This file (images.txt) contains the list of image file names, with each line corresponding to one image.
# The content of the file is expected to be as follows:
# <image_id> <image_name>
# 
# where image_id is a unique numeric identifier for each image in the dataset, and image_name is the path
# to the corresponding image file. An example line is the following:
# 16 001.Black_footed_Albatross/Black_Footed_Albatross_0016_796067.jpg
parser.add_argument('--imgs_list_txt', type=str, default='/images.txt', help='File with list of image paths.')

# This file (train_test_split.txt) contains the suggested training/validation split, with each line corresponding
# to one image. The content of the file is expected to be as follows:
# <image_id> <is_training_image>
#
# where image_id is a unique numeric identifier for each image in the dataset (same as in images.txt), and
# is_training_image takes either value 1 or 0, denoting that the file is in the training or the validation
# set, respectively. An example line is the following:
# 16 0
parser.add_argument('--split_list_txt', type=str, default='/train_test_split.txt', help='File with information about train/validation split of the data.')

parser.add_argument('--save_dir', type=str, default='/CUB_200_2011/', help='Output data directory')

FLAGS = parser.parse_args()

def find_image_files(imgs_list_txt):
    """Build lists of all images file paths, numeric labels, and
       human-readable labels.
    Args:
        imgs_list_txt: string; path to file with list of image paths.
    Returns:
        filenames: list of strings; it contains paths to image files.
        labels_values: list of ints; it contains numeric labels.
        labels_names: list of strings; it contains human-readable labels.
    """

    lines = tf.io.gfile.GFile(imgs_list_txt, 'r').readlines()

    filenames = []
    labels_values = []
    labels_names = []
    # Iterate over file lines
    for l in lines:
        if l:
            parts = l.strip().split(' ')
            assert len(parts) == 2
            filenames.append('/' + parts[1])
            
            p = parts[1].split('.', 1)
            labels_values.append(int(p[0]))
            labels_names.append(p[1].split('/', 1)[0])

    print('Found %d JPEG files across %d labels.' %(len(filenames), len(labels_names)))

    return filenames, labels_values, labels_names

def split_data(split_list_txt, filenames, labels_values, labels_names):
    """Create entries for csv files about the training and validation
       splits of the Caltech-UCSD Birds-200-2011 dataset. Each entry
       includes the path to an image, its numeric label, and its
       human-readable label.
    Args:
        split_list_txt: string; path to file with information about
            train/validation split of the data.
        filenames: list of strings; it contains paths to image files.
        labels_values: list of ints; it contains numeric labels.
        labels_names: list of strings; it contains human-readable labels.
    Returns:
        train_csv_entries: np array; it contains paths to the image files
            of the training split. It also contains the numeric label and
            the  human-readable label of each iamge. It is of size
            [num_imgs_train, 3], where num_imgs_train is the number of
            images in the training split.
        validation_csv_entries: np array; it contains paths to the image
            files of the validation split. It also contains the numeric
            label and the human-readable label of each iamge. It is of
            size [num_imgs_val, 3], where num_imgs_val is the number of
            images in the validation split.
    """

    lines = tf.io.gfile.GFile(split_list_txt, 'r').readlines()
    
    split_val = []
    for l in lines:
        if l:
            split_val.append(l.strip().split(' ')[1])

    # Shuffle the ordering of image files to guarantee
    # random ordering of the images with respect to labels
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    labels_values = [labels_values[i] for i in shuffled_index]
    labels_names = [labels_names[i] for i in shuffled_index]
    split_val = [split_val[i] for i in shuffled_index]

    df_array = np.concatenate((np.expand_dims(np.asarray(filenames), 1), np.expand_dims(np.asarray(labels_values), 1),
                                np.expand_dims(np.asarray(labels_names), 1)), axis=1)
    mask = np.asarray(split_val).astype(bool)
    inv_mask = (1 - mask).astype(bool)

    # Create entries for csv files about the training split
    train_csv_entries = df_array[mask, :]
    # Create entries for csv files about the validation split
    validation_csv_entries = df_array[inv_mask, :]

    print('Added %d entries to train split, and %d entries to validation split.' %(train_csv_entries.shape[0], validation_csv_entries.shape[0]))

    return train_csv_entries, validation_csv_entries

def save_to_csv(csv_entries, save_dir, tag):
    """Save csv entries.
    Args:
        csv_entries: np array; it contains paths to image files with
            their numeric labels, and human-readable labels. It is of size
            [num_imgs, 3], where num_imgs is the number of images files.
        save_dir: string; directory to save the csv file.
        tag: string; name of the csv file to save.
    Returns:
        -
    """

    cols = ['fname', 'class_number', 'class_name']
    df = pd.DataFrame(csv_entries, columns=cols)
    fp = os.path.join(save_dir, tag + '.csv')

    if (not os.path.isdir(FLAGS.save_dir)):
        os.makedirs(FLAGS.save_dir)
    df.to_csv(fp, encoding='utf-8', index=False)
    print('CSV saved at %s.' %fp)

def main(argv=None):
    """Create csv files for the training and validation splits
       of the Caltech-UCSD Birds-200-2011 dataset.
    Args:
        -
    Returns:
        -
    """

    # Build lists with image file paths, numeric labels, and human-readable labels
    filenames, labels_values, labels_names = find_image_files(FLAGS.imgs_list_txt)

    # Create csv entries for training and validation splits
    train_csv_entries, validation_csv_entries = split_data(FLAGS.split_list_txt, filenames, labels_values, labels_names)

    # Save csv files for training and validation splits
    save_to_csv(train_csv_entries, FLAGS.save_dir, 'train_anno')
    save_to_csv(validation_csv_entries, FLAGS.save_dir, 'validation_anno')

if __name__ == '__main__':
    main()
