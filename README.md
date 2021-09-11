# Traversal Network (TNet)

We provide the TensorFlow implementation of the Traversal Network (TNet) architecture, presented in "Hard-Attention for Scalable Image Classification" (https://arxiv.org/pdf/2102.10212.pdf). The code is organized according to the datasets used for the experimental evaluation of TNet. Each folder contains code to convert raw data to TFRecords, to stream input batches, to build TNet and baseline models, and to train and evaluate the models. Learned weights, along with instructions to replicate the results presented in the paper, are provided as well.

## ImageNet ILSVRC 2012

All related files can be found under the `/ImageNet/` folder.

### Data preparation

Detailed intstructions to download the raw data, and to create related metadata files, are provided in `create_tfrecords_imagenet.py`. Given the necessary files are created, and the data directories are organized appropriately, the following command can be used to convert raw data to TFRecords:

```
python create_tfrecords_imagenet.py --output_directory '/path/to/output/dir/'
                                    --labels_file '/path/to/imagenet_lsvrc_2015_synsets.txt'
                                    --imagenet_metadata_file '/path/to/imagenet_metadata.txt'
                                    --bounding_box_file '/path/to/imagenet_2012_bounding_boxes.csv'
```

### Training

There are many different flags that can be used to customize the training of TNet and BagNet-77 baseline. An example command for training TNet is the following:

```
python train.py --to_train
                --batch_size 64
                --num_epochs 200
                --initial_lr 0.0001
                --lr_scedule_1step
                --keep_prob 0.5
                --loc_per_grid 3.0
                --reinfornce_reg_w 0.1
                --perFReg_ce_weight 0.3
                --perFReg_reinf_weight 0.3
                --overlap 0.34375
                --num_patches_y 5
                --num_patches_x 5
                --base_res_y 77
                --base_res_x 77
                --num_res_levels 2
                --num_do_layers 1
                --descr_tag 'BagNet_77_TNet'
                --save_tag 'TNet_imagenet'
                --num_gpus 2
                --data_dir '/path/to/TFRecords/dir/'
                --ckpt_dir '/path/to/ckpts/dir/'
                --summaries_dir '/path/to/summaries/dir/'
                --keep_weights_summary
```

An example command for traing BagNet-77 baseline is the following:

```
python train_bl.py --to_train
                   --batch_size 64
                   --num_epochs 200
                   --initial_lr 0.0001
                   --lr_scedule_1step
                   --keep_prob 0.375
                   --num_do_layers 1
                   --descr_tag 'BagNet_77'
                   --save_tag 'BagNet_77_imagenet'
                   --num_gpus 2
                   --data_dir '/path/to/TFRecords/dir/'
                   --ckpt_dir '/path/to/ckpts/dir/'
                   --summaries_dir '/path/to/summaries/dir/'
                   --keep_weights_summary
```

Commands to replicate the training of the networks presented in the paper, can be found in `results_replication.txt`.

The weights of the TNet model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=1JmKOP6aN2tYUsD4-zWMfRCbXlId6gcko&export=download). <br />
The weights of the BagNet-77 baseline reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=1jB3ouvcVhxYnTrIlagUawBkVqhL8Wfro&export=download).

### Evaluation

A trained TNet model can be evaluated on the training and validation sets, by using a command similar to the following example:

```
python train.py --to_evaluate_train
                --to_evaluate_val
                --batch_size 64
                --loc_per_grid 3.0
                --overlap 0.34375
                --num_patches_y 5
                --num_patches_x 5
                --base_res_y 77
                --base_res_x 77
                --num_res_levels 2
                --descr_tag 'BagNet_77_TNet'
                --save_tag 'BagNet_77_imagenet'
                --num_gpus 1
                --data_dir '/path/to/TFRecords/dir/'
                --ckpt_dir '/path/to/ckpts/dir/'
                --summaries_dir '/path/to/summaries/dir/'
                --restore_dir '/path/to/dir/with/ckpt/to/restore/'
```

An example command for evaluating a trained BagNet-77 baseline network, is the following:

```
python train_bl.py --to_evaluate_train
                   --to_evaluate_val
                   --batch_size 64
                   --descr_tag 'BagNet_77'
                   --save_tag 'BagNet_77_imagenet'
                   --num_gpus 1
                   --data_dir '/path/to/TFRecords/dir/'
                   --ckpt_dir '/path/to/ckpts/dir/'
                   --summaries_dir '/path/to/summaries/dir/'
                   --restore_dir '/path/to/dir/with/ckpt/to/restore/'
```

Commands to evaluate the networks presented in the paper, can be found in `results_replication.txt`.

## Functional Map of the World (fMoW)

All related files can be found under the `/fMoW/` folder.

### Data preparation

Details about how to download raw data are provided in `create_TFRecords_fMoW.py`. As explained in `create_TFRecords_fMoW.py`, test set data should be manually matched to ground truth labels. This can be done with the following command:

```
python match_test_gt.py --root_test_dir '/path/to/original/test/data/root/dir/'
                        --test_output_dir '/path/to/output/dir/'
                        --match_gt_json_path '/path/to/test_gt_mapping.json'
```

Given the desired uniformity in the directory organization of the training, validation, and test sets is established, the following command can be used to convert raw data to TFRecords:

```
python create_TFRecords_fMoW.py --train_directory '/path/to/training/set/dir/'
                                --validation_directory '/path/to/validation/set/dir/'
                                --test_directory '/path/to/test/set/dir/'
                                --output_directory '/path/to/output/dir/'
```

In order to crop images according to the provided bounding boxes, the following command can be used:

```
python crop_fMoW.py --train_directory '/path/to/training/set/dir/'
                    --validation_directory '/path/to/validation/set/dir/'
                    --test_directory '/path/to/test/set/dir/'
                    --output_directory '/path/to/output/dir/'
```

TFRecords for cropped images can be created with the following command:

```
python create_TFRecords_fMoW.py --cropped_data
                                --train_directory '/path/to/training/set/dir/'
                                --validation_directory '/path/to/validation/set/dir/'
                                --test_directory '/path/to/test/set/dir/'
                                --output_directory '/path/to/output/dir/'
                                --maximum_min_dim 224
```

### Training and evaluation

Training and evaluation commands are similar to the ones provided for ImageNet. The commands used to train and evaluate the networks presented in the paper, can be found in `results_replication.txt`.

The weights of the TNet model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=12WJCIZ0nBICEf4X1C8qNPt3kCYSNHmxa&export=download). <br />
The weights of the EfficientNet-B0 model trained on cropped images, can be downloaded [here](https://drive.google.com/u/1/uc?id=12RzU8hNHbOi3NCzoTWrceYkpclBy9IJN&export=download). <br />
The weights of the EfficientNet-B0 model trained on images of size 224x224 px, can be downloaded [here](https://drive.google.com/u/1/uc?id=12MxuuwyZ-WtQ9kBlX8u23uc9MeYQDNnR&export=download). <br />
The weights of the EfficientNet-B0 model trained on images of size 448x448 px, can be downloaded [here](https://drive.google.com/u/1/uc?id=12Ob1oKW8LZePUZt2dI1Jsz4oWwNZeTFi&export=download). <br />
The weights of the EfficientNet-B0 model trained on images of size 896x896 px, can be downloaded [here](https://drive.google.com/u/1/uc?id=12Q-gnz5ve1UvJBxaKp5mlP3LHk6ERPTn&export=download).

## CUB-200-2011

All related files can be found under the `/CUB/` folder.

### Data preparation

The link to download raw data is provided in `create_tfrecords_cub.py`. Before the creation of TFRecords, data can be split into training and validation sets through the following command (a csv file for each split is created):

```
python create_csv_cub.py --imgs_list_txt '/path/to/images.txt'
                         --split_list_txt '/path/to/train_test_split.txt'
                         --save_dir '/path/to/output/dir/'
```

Given the csv files for each data split are created, the following command can be used to convert raw data to TFRecords:

```
python create_tfrecords_cub.py --img_dir '/path/to/images/dir/'
                               --train_csv_path '/path/to/train_anno.csv'
                               --dev_csv_path '/path/to/validation_anno.csv'
                               --output_dir '/path/to/output/dir/'
```

### Training and evaluation

Training and evaluation commands are similar to the ones provided for ImageNet. As noted in the paper, the pre-trained weights for EfficientNet models that are used for fine-tuning, can be downloaded <a href="https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet">here</a>. They correspond to the weights of models trained with NoisyStudent and RandAugment, with the extra JFT-300M unlabeled data. Under the folder `/restore_dicts/` are provided dictionaries that are used to load the pre-trained weights to TNet and the baselines. The commands used to train and evaluate the networks presented in the paper, can be found in `results_replication.txt`.

The weights of the TNet-B0 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=13vG6HS7GvV5I9HVsje67wSTUSdVqZ1c2&export=download). <br />
The weights of the TNet-B1 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=144S8dBKi8NSUwfeejNi9ZhFpZiV34DDc&export=download). <br />
The weights of the TNet-B2 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=146LIt0YAZBr0WAlYXyBymE2Ml3HlZION&export=download). <br />
The weights of the TNet-B3 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=147xTMXINAEXyMcvqFkiajG7b3Kr-zmpn&export=download). <br />
The weights of the TNet-B4 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=148P9fcajdNu_qvdz2Vz0makZE5z0TX5A&export=download). <br />
The weights of the EfficientNet-B0 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=13p5fCpEHp1T6-jYjpwNrnTUm5PCYK3HF&export=download). <br />
The weights of the EfficientNet-B1 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=13pvYgEskLkvGg1RLkkBZPLvCRoVV_0Jf&export=download). <br />
The weights of the EfficientNet-B2 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=13vASt7Cidwx6YytTF0lUxNz-HtWawZpy&export=download). <br />
The weights of the EfficientNet-B3 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14IG_vz8BbBzVRweytb-j3BQkbJW5fU3A&export=download). <br />
The weights of the EfficientNet-B4 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14L8pQA7_C0As5Pj5oZz2n5ZHAvMJyPYS&export=download). <br />

## NABirds

All related files can be found under the `/NABirds/` folder.

### Data preparation

Details about the NABirds data are provided in `create_tfrecords_nab.py`. The following command can be used to convert raw data to TFRecords:

```
python create_tfrecords_nab.py --root_directory '/path/to/dir/with/all/downloaded/data/'
                               --data_directory '/path/to/images/dir/'
                               --output_directory '/path/to/output/dir/'
```

### Training and evaluation

Training and evaluation commands are similar to the ones provided for ImageNet. As noted in the paper, the pre-trained weights for EfficientNet models that are used for fine-tuning, can be downloaded <a href="https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet">here</a>. They correspond to the weights of models trained with NoisyStudent and RandAugment, with the extra JFT-300M unlabeled data. Under the folder `/restore_dicts/` are provided dictionaries that are used to load the pre-trained weights to TNet and the baselines. The commands used to train and evaluate the networks presented in the paper, can be found in `results_replication.txt`.

The weights of the TNet-B0 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14PNE7AlStP6ID1QJzft-6dguM3c7kBz_&export=download). <br />
The weights of the TNet-B1 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14UD3OSowZb3z97bcJlAm-VCjQAAQfroh&export=download). <br />
The weights of the TNet-B2 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14bb7v6h_FaFnIRDt7Ibm2O8D1UBvABN2&export=download). <br />
The weights of the TNet-B3 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14YDJUlB-DlpfompGXn6Lp_QQyU16e8IP&export=download). <br />
The weights of the TNet-B4 model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14YAnGLCoKlODvCnU9TgQHkz6F1pFJMj2&export=download). <br />
The weights of the EfficientNet-B0 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14bpOMBfeeSugkjfcQuU5Edr8c_3HeCsI&export=download). <br />
The weights of the EfficientNet-B1 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14cF2yJlY4CbSHkrb2bcsRzQx0AlScZip&export=download). <br />
The weights of the EfficientNet-B2 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14efrtrX8gvQR9a3LIhVuTtI-k_s5nhAd&export=download). <br />
The weights of the EfficientNet-B3 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14h1pJWkEKrV7uufYC4eEnMmCruADZDSA&export=download). <br />
The weights of the EfficientNet-B4 baseline model reported in the paper, can be downloaded [here](https://drive.google.com/u/1/uc?id=14nDTt88vBQHz4VbxAs0WzEHO0uI4snsE&export=download). <br />

