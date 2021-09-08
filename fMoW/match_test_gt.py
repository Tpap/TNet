"""Match ground truth information with test images.
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
between the annotations under folder test_gt, and the images under folder test.
The current script organizes jpeg and json files for the test set, in the following
directory structure:
    /test_matched_with_gt/airport/airport_0/airport_0_0_rgb.jpeg
    /test_matched_with_gt/airport/airport_0/airport_0_0_rgb.json
    ...
"""

import argparse
import os
import json
from tqdm import tqdm
import errno
import shutil



parser = argparse.ArgumentParser()

parser.add_argument('--root_test_dir', type=str, default='/fMoW-rgb/', help='Root directory of the original test data.')
parser.add_argument('--test_output_dir', type=str, default='/test_matched_with_gt/', help='Directory to output the matched data.')
parser.add_argument('--match_gt_json_path', type=str, default='/test_gt_mapping.json', help='Path to test_gt_mapping.json.')

FLAGS = parser.parse_args()

def try_mkdir(input_dir):
    """Try to make directory.
    Args:
        input_dir: string; directory to create.
    Returns:
        -
    """

    if (not os.path.isdir(input_dir)):
        try:
            os.makedirs(input_dir)
        except OSError as e:
            if (e.errno == errno.EEXIST):
                pass

def main(argv=None):
    """Match data with ground truth information,
       and save to new directory structure.
    Args:
        -
    Returns:
        -
    """
    
    # Load test_gt_mapping.json, and iterate over its entries
    jsonData = json.load(open(FLAGS.match_gt_json_path))
    for entry in tqdm(jsonData):
        src_test_dir = os.path.join(FLAGS.root_test_dir, entry['output'])
        src_test_gt_dir = os.path.join(FLAGS.root_test_dir, entry['input'])
        save_dir_suffix = entry['input'].split('/', 1)[1]
        save_dir = os.path.join(FLAGS.test_output_dir, save_dir_suffix)
        try_mkdir(save_dir)

        f_name_prefix_test_gt = entry['input'].split('/')[-1]
        f_name_prefix_test = entry['output'].split('/')[-1]
        
        for _, _, files in os.walk(src_test_dir):
            for f_src in files:
                # Ignore msrgb images
                if f_src.endswith('_rgb.jpg'):
                    f_src_test_img = f_src

                    f_scr_test_gt_json = f_src.replace('.jpg', '.json')
                    f_scr_test_gt_json = f_scr_test_gt_json.replace(f_name_prefix_test, f_name_prefix_test_gt)
                    
                    f_dst_json = f_scr_test_gt_json
                    f_dst_img = f_src_test_img.replace(f_name_prefix_test, f_name_prefix_test_gt)

                    jsonData_src_test_gt = json.load(open(os.path.join(src_test_gt_dir, f_scr_test_gt_json)))
                    # Ignore bounding boxes with unknown ids
                    if not isinstance(jsonData_src_test_gt['bounding_boxes'], list):
                        jsonData_src_test_gt['bounding_boxes'] = [jsonData_src_test_gt['bounding_boxes']]
                    bb_lst = []
                    for bb in jsonData_src_test_gt['bounding_boxes']:
                        if (bb['ID'] != -1):
                            bb_lst.append(bb)
                    
                    jsonData_dst = jsonData_src_test_gt
                    jsonData_dst['bounding_boxes'] = bb_lst
                    
                    # Save updated json file
                    json.dump(jsonData_dst, open(os.path.join(save_dir, f_dst_json), 'w'))
                    # Copy test image under the new directory
                    shutil.copy(os.path.join(src_test_dir, f_src_test_img), os.path.join(save_dir, f_dst_img))

if __name__ == '__main__':
    main()
