# Traversal Network (TNet)

We provide the TensorFlow implementation of the Traversal Network (TNet) architecture, presented in "Hard-Attention for Scalable Image Classification" (https://arxiv.org/pdf/2102.10212.pdf). The code is organized according to the datasets used for the experimental evaluation of TNet. Each folder contains code to convert raw data to TFRecords, to stream input batches, to build TNet and baseline models, and to train and evaluate the models. Trained weights, along with instructions to replicate the results presented in the paper, are provided as well.

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

There are many different flags that can be used for a customized training of TNet and the BagNet-77 baseline. An example command for traing TNet is the following:

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
                --descr_tag 'BagNet_77_TNet'
                --num_gpus 2
                --base_res_y 77
                --base_res_x 77
                --num_res_levels 2
                --num_do_layers 1
                --save_tag 'TNet_imagenet'
                --data_dir '/path/to/TFRecords/dir'
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
                   --num_gpus 2
                   --descr_tag 'BagNet_77'
                   --save_tag 'BagNet_77_imagenet'
                   --data_dir '/path/to/TFRecords/dir'
                   --ckpt_dir '/path/to/ckpts/dir/'
                   --summaries_dir '/path/to/summaries/dir/'
                   --keep_weights_summary
```

Commands to replicate the training of the networks presented in the paper, can be found in `results_replication.txt`.

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
                --descr_tag 'BagNet_77_TNet'
                --num_gpus 1
                --base_res_y 77
                --base_res_x 77
                --num_res_levels 2
                --save_tag 'BagNet_77_imagenet'
                --data_dir '/path/to/TFRecords/dir'
                --ckpt_dir '/path/to/ckpts/dir/'
                --summaries_dir '/path/to/summaries/dir/'
                --restore_dir '/path/to/dir/with/ckpt/to/restore/'
```

An example command for evaluating a trained BagNet-77 baseline network, is the following:

```
python train_bl.py --to_evaluate_train
                   --to_evaluate_val
                   --batch_size 64
                   --num_gpus 1
                   --descr_tag 'BagNet_77'
                   --save_tag 'BagNet_77_imagenet'
                   --data_dir '/path/to/TFRecords/dir'
                   --ckpt_dir '/path/to/ckpts/dir/'
                   --summaries_dir '/path/to/summaries/dir/'
                   --restore_dir '/path/to/dir/with/ckpt/to/restore/'
```

Commands to evaluate the networks presented in the paper, can be found in `results_replication.txt`.

## Functional Map of the World (fMoW)

All related files can be found under the /fMoW/ folder.

### Data preparation

Follow instructions

### Training and evaluation

## CUB-200-2011

All related files can be found under the /CUB/ folder.

### Data preparation

Follow instructions

### Training and evaluation

## NABirds

All related files can be found under the /NABirds/ folder.

### Data preparation

Follow instructions

### Training and evaluation

