"""Train and evaluate TNet.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import time
import pickle
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import pandas as pd

import input_nab as tf_input
import models_nab as tf_models
from models_nab import adjust_var_name



parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, help='Global batch size (recommended to be divisible by the number of GPUs).')
parser.add_argument('--num_samples', type=int, default=1, help='Number of samples for the Monte Carlo estimator in the learning rule.')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--num_classes', type=int, default=555, help='Number of classes.')
parser.add_argument('--initial_lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--two_oprimizers', action='store_true', help='Whether to use differet optimizers for different sets of variables; used in fine-tuning.')
parser.add_argument('--initial_lr2', type=float, help='Initial learning rate for a subset of trainable variables.')
parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Learning rate decay factor.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout drop probability.')
parser.add_argument('--l2_reg', type=float, default=0., help='Importance weight for l2 regularization.')
parser.add_argument('--loc_per_grid', nargs='+', type=float, default=[3.0], help='Number of locations to attend per attention grid; l-1 entries for a total number of l processing levels.')
parser.add_argument('--reinfornce_reg_w', type=float, default=0.0, help='Importance weight for reinforce loss term.')
parser.add_argument('--perFReg_ce_weight', type=float, default=0.5, help='Importance weight for cross entropy loss when per feature regularization is applied.')
parser.add_argument('--perFReg_reinf_weight', type=float, default=0.5, help='Importance weight for reinforce loss term when per feature regularization is applied.')
parser.add_argument('--perFReg_cap', type=int, default=-1, help='The maximum processing level that per-feature regularization is applied. If negative, all processing levels are considered.')
parser.add_argument('--overlap', type=float, default=0.5, help='The fraction of each spatial image dimension, occupied by the correspending dimension of a grid patch (same for both spatial dimensions).')
parser.add_argument('--descr_tag', type=str, help='Description of the model to build')
parser.add_argument('--width_coefficient', type=float, default=1.0, help='Coefficient for scaling the width of the feature extraction netwrok.')
parser.add_argument('--depth_coefficient', type=float, default=1.0, help='Coefficient for scaling the depth of the feature extraction netwrok.')
parser.add_argument('--block_drop_rate', type=float, default=0.2, help='The maximum probability for dropping model blocks during feature extraction (stochastic depth parameter).')
parser.add_argument('--depth_divisor', type=int, default=8, help='Determines quantization during depth scaling.')
parser.add_argument('--activation', type=str, default='swish', help='Type of activation to be used.')
parser.add_argument('--batch_norm', action='store_true', help='Whether to use Batch Normalization.')
parser.add_argument('--gpus_type', type=str, default='NotReported', help='Type of GPUs in use')
parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use.')
parser.add_argument('--num_patches_y', type=int, default=3, help='Number of patches on the vertical dimension of the grid considered by the location module.')
parser.add_argument('--num_patches_x', type=int, default=3, help='Number of patches on the horizontal dimension of the grid considered by the location module.')
parser.add_argument('--ls_dim', type=int, default=1280, help='Latent space dimensionality.')
parser.add_argument('--pos_dim_divisor', type=int, default=1, help='Determines the dimensionality of positional embeddings, by dividing the latent space dimensionality.')
parser.add_argument('--base_res_y', type=int, default=32, help='Base resolution of the feature extraction module in the vertical dimension.')
parser.add_argument('--base_res_x', type=int, default=32, help='Base resolution of the feature extraction module in the horizontal dimension.')
parser.add_argument('--num_res_levels', type=int, default=1, help='Total number of processing levels that TNet goes through.')
parser.add_argument('--img_size_y', type=int, default=224, help='Resolution of the input image in the vertical dimension.')
parser.add_argument('--img_size_x', type=int, default=224, help='Resolution of the input image in the horizontal dimension.')
parser.add_argument('--adv_eval_data', action='store_true', help='Whether to return advanced evaluation data.')
parser.add_argument('--vars_to_exclude', nargs='+', type=str, default=[], help='Variables to not restore.')
parser.add_argument('--vars_to_update', nargs='+', type=str, default=[], help='Variables that will be updated during training; by default, all trainable variables are updated.')
parser.add_argument('--lr_scedule_2step', action='store_true', help='Whether to drop learning rate 2 times.')
parser.add_argument('--lr_scedule_1step', action='store_true', help='Whether to drop learning rate once.')
parser.add_argument('--feat_weighting', action='store_true', help='Whether to use the feature weighting module.')

parser.add_argument('--image_ids_struct_path', type=str, default='/image_ids_struct.txt', help='Path to txt file with pyhton dictionary that contains metadata needed for the creation of TFRecord files.')

parser.add_argument('--contrastive_loss', action='store_true', help='Whether to use contrastive loss with cosine similarity.')
parser.add_argument('--contrastive_margin', type=float, default=None, help='Maximum allowed cosine similarity between features from different classes in the contrastive loss. It takes values in range (0., 1.).')
parser.add_argument('--l_contrastive', type=float, default=0., help='Importance weight for contrastive loss.')

parser.add_argument('--batches_to_profile_range', nargs='+', type=float, default=[-1, -1], help='Range of batches to profile.')
parser.add_argument('--profile_evaluation', action='store_true', help='Whether to profile final evaluation by using batches in batches_to_profile_range range.')
parser.add_argument('--profile_step', type=float, default=0., help='Step for profiling; used to calculate statistics, mean and std.')
parser.add_argument('--to_train', action='store_true', help='Whether to train the model.')
parser.add_argument('--to_evaluate_val', action='store_true', help='Whether to evaluate the model with validation data.')
parser.add_argument('--to_evaluate_train', action='store_true', help='Whether to evaluate the model with training data.')
parser.add_argument('--batches_to_time_range', nargs='+', type=float, default=[-1, -1], help='Range of batches to time.')
parser.add_argument('--eval_epochs_num', type=int, default=1, help='Number of epochs for evaluation; values greater that 1 are mainly used for the random sampling of batches during profiling.')
parser.add_argument('--dont_save_eval_txt', action='store_true', help='Whether to save txt file with the evaluation results.')

parser.add_argument('--save_tag', type=str, help='Tag that describes the model. It is added as an extension to the directories used for checkpoints and tf summaries.')
parser.add_argument('--data_dir', type=str, help='Path to the data (TFRecords) directory.')
parser.add_argument('--ckpt_dir', type=str, help='Directory where to write checkpoints (tag will be added).')
parser.add_argument('--summaries_dir', type=str, help='Directory where to write tf summaries (tag will be added).')
parser.add_argument('--restore_dir', type=str, default='No_ckpt', help='Directory from where to read model checkpoints.')
parser.add_argument('--ckpt_to_restore', type=int, default=-1, help='Which checkpoint file to use (-1 restores the last saved ckpt).')
parser.add_argument('--resume_training', action='store_true', help='Whether to resume training; global step is restored and not reset.')
parser.add_argument('--dictionary_to_restore_from', type=str, default=None, help='Path to dictionary that maps model variables to stored ones.')

parser.add_argument('--log_frequency', type=int, default=10, help='How often to log results to the console.')
parser.add_argument('--summary_frequency', type=int, default=2000, help='How often to log results to the summary file.')
parser.add_argument('--ckpt_frequency', type=int, default=10000, help='How often to make a checkpoint.')
parser.add_argument('--ckpt_sparse_num', type=int, default=10, help='How many checkpoints to make sparsily throughout training.')

parser.add_argument('--keep_grads_summary', action='store_true', help='Keep tf summary for the trainable weights gradients.')
parser.add_argument('--keep_weights_summary', action='store_true', help='Keep tf summary for the model variables.')
parser.add_argument('--keep_activations_summary', action='store_true', help='Keep tf summary for model activations.')

FLAGS = parser.parse_args()

BL_DECAY = 0.9

NUM_IMAGES = {
    'train': 23929,
    'validation': 24633
}

def set_up_dirs():
    """Set up directories for checkpoints and summaries.
    Args:
        -
    Returns:
        -
    """

    if (not FLAGS.ckpt_dir.endswith('/')): FLAGS.ckpt_dir = FLAGS.ckpt_dir + '/'
    FLAGS.ckpt_dir = FLAGS.ckpt_dir + FLAGS.save_tag + '/'
    FLAGS.ckpt_dir_latest = FLAGS.ckpt_dir + 'latest/'
    if (not tf.io.gfile.exists(FLAGS.ckpt_dir_latest)): tf.io.gfile.makedirs(FLAGS.ckpt_dir_latest)
    FLAGS.ckpt_dir_sparse = FLAGS.ckpt_dir + 'sparse/'
    if (not tf.io.gfile.exists(FLAGS.ckpt_dir_sparse)): tf.io.gfile.makedirs(FLAGS.ckpt_dir_sparse)

    if (not FLAGS.summaries_dir.endswith('/')): FLAGS.summaries_dir = FLAGS.summaries_dir + '/'
    FLAGS.summaries_dir = FLAGS.summaries_dir + FLAGS.save_tag + '/'
    if (not tf.io.gfile.exists(FLAGS.summaries_dir)): tf.io.gfile.makedirs(FLAGS.summaries_dir)

    if (FLAGS.restore_dir != 'No_ckpt'):
        if (not FLAGS.restore_dir.endswith('/')): FLAGS.restore_dir = FLAGS.restore_dir + '/'

def build_labels_to_human():
    """Build map from numerical labels to human-readable labels.
    Args:
        -
    Returns:
        labels_to_human: dictionary; it maps numeric labels
            to human-readable labels.
    """

    with open(FLAGS.image_ids_struct_path, "rb") as fp:
        image_ids = pickle.load(fp)

    labels_to_human = {}

    for e in image_ids:
        if (image_ids[e]['label'] not in labels_to_human):
            labels_to_human[image_ids[e]['label']] = image_ids[e]['class_name']
    
    return labels_to_human

def write_hyperparameters(total_train_steps, lr_boundaries=None, lr_values=None):
    """Save hyperparameter values.
    Args:
        total_train_steps: int; total number of training steps.
        lr_boundaries: list of ints; it contains the optimization
            steps where learning rates changes.
        lr_values: list of floats; it contains all learning rate
            values during optimization.
    Returns:
        -
    """

    if (FLAGS.to_train):
        print("Training for %d steps" %total_train_steps)

    print('batch_size = %s' %str(FLAGS.batch_size))
    print('num_samples = %s' %str(FLAGS.num_samples))
    print('num_epochs = %s' %str(FLAGS.num_epochs))
    print('num_classes = %s' %str(FLAGS.num_classes))
    print('initial_lr = %s' %str(FLAGS.initial_lr))
    print('two_oprimizers = %s' %str(FLAGS.two_oprimizers))
    print('initial_lr2 = %s' %str(FLAGS.initial_lr2))
    print('lr_decay_factor = %s' %str(FLAGS.lr_decay_factor))
    print('dropout_rate = %s' %str(FLAGS.dropout_rate))
    print('l2_reg = %s' %str(FLAGS.l2_reg))
    print('loc_per_grid = %s' %(', '.join([str(l) for l in FLAGS.loc_per_grid])))
    print('reinfornce_reg_w = %s' %str(FLAGS.reinfornce_reg_w))
    print('perFReg_ce_weight = %s' %str(FLAGS.perFReg_ce_weight))
    print('perFReg_reinf_weight = %s' %str(FLAGS.perFReg_reinf_weight))
    print('perFReg_cap = %s' %str(FLAGS.perFReg_cap))
    print('overlap = %s' %str(FLAGS.overlap))
    print('descr_tag = %s' %str(FLAGS.descr_tag))
    print('width_coefficient = %s' %str(FLAGS.width_coefficient))
    print('depth_coefficient = %s' %str(FLAGS.depth_coefficient))
    print('block_drop_rate = %s' %str(FLAGS.block_drop_rate))
    print('depth_divisor = %s' %str(FLAGS.depth_divisor))
    print('activation = %s' %str(FLAGS.activation))
    print('batch_norm = %s' %str(FLAGS.batch_norm))
    print('gpus_type = %s' %str(FLAGS.gpus_type))
    print('num_gpus = %s' %str(FLAGS.num_gpus))
    print('num_patches_y = %s' %str(FLAGS.num_patches_y))
    print('num_patches_x = %s' %str(FLAGS.num_patches_x))
    print('ls_dim = %s' %str(FLAGS.ls_dim))
    print('pos_dim_divisor = %s' %str(FLAGS.pos_dim_divisor))
    print('base_res_y = %s' %str(FLAGS.base_res_y))
    print('base_res_x = %s' %str(FLAGS.base_res_x))
    print('num_res_levels = %s' %str(FLAGS.num_res_levels))
    print('img_size_y = %s' %str(FLAGS.img_size_y))
    print('img_size_x = %s' %str(FLAGS.img_size_x))
    print('adv_eval_data = %s' %FLAGS.adv_eval_data)
    print('vars_to_exclude = %s' %(', '.join([str(l) for l in FLAGS.vars_to_exclude])))
    print('vars_to_update = %s' %(', '.join([str(l) for l in FLAGS.vars_to_update])))
    print('lr_scedule_2step = %s' %FLAGS.lr_scedule_2step)
    print('lr_scedule_1step = %s' %FLAGS.lr_scedule_1step)
    print('feat_weighting = %s' %str(FLAGS.feat_weighting))
    print('lr_boundaries = %s' %str(lr_boundaries))
    print('lr_values = %s' %str(lr_values))
    
    print('contrastive_loss = %s' %str(FLAGS.contrastive_loss))
    print('contrastive_margin = %s' %str(FLAGS.contrastive_margin))
    print('l_contrastive = %s' %str(FLAGS.l_contrastive))

    print('batches_to_profile_range = %s' %(', '.join([str(l) for l in FLAGS.batches_to_profile_range])))
    print('profile_evaluation = %s' %FLAGS.profile_evaluation)
    print('profile_step = %s' %str(FLAGS.profile_step))
    print('to_train = %s' %FLAGS.to_train)
    print('to_evaluate_val = %s' %FLAGS.to_evaluate_val)
    print('to_evaluate_train = %s' %FLAGS.to_evaluate_train)
    print('batches_to_time_range = %s' %(', '.join([str(l) for l in FLAGS.batches_to_time_range])))
    print('eval_epochs_num = %s' %FLAGS.eval_epochs_num)
    print('dont_save_eval_txt = %s' %FLAGS.dont_save_eval_txt)

    print('save_tag = %s' %FLAGS.save_tag)
    print('data_dir = %s' %FLAGS.data_dir)
    print('ckpt_dir = %s' %FLAGS.ckpt_dir)
    print('summaries_dir = %s' %FLAGS.summaries_dir)
    print('restore_dir = %s' %FLAGS.restore_dir)
    print('ckpt_to_restore = %s' %FLAGS.ckpt_to_restore)
    print('resume_training = %s' %FLAGS.resume_training)
    print('dictionary_to_restore_from = %s' %FLAGS.dictionary_to_restore_from)

    print('log_frequency = %s' %FLAGS.log_frequency)
    print('summary_frequency = %s' %FLAGS.summary_frequency)
    print('ckpt_frequency = %s' %FLAGS.ckpt_frequency)
    print('ckpt_sparse_num = %s' %FLAGS.ckpt_sparse_num)

    print('keep_grads_summary = %s' %FLAGS.keep_grads_summary)
    print('keep_weights_summary = %s' %FLAGS.keep_weights_summary)
    print('keep_activations_summary = %s' %FLAGS.keep_activations_summary)

    fname = 'readme.txt'
    f = open((FLAGS.ckpt_dir + fname), 'w')
    f.write('batch_size: ' + str(FLAGS.batch_size))
    f.write('\nnum_samples: ' + str(FLAGS.num_samples))
    f.write('\nnum_epochs: ' + str(FLAGS.num_epochs))
    f.write('\nnum_classes: ' + str(FLAGS.num_classes))
    f.write('\ninitial_lr: ' + str(FLAGS.initial_lr))
    f.write('\ntwo_oprimizers: ' + str(FLAGS.two_oprimizers))
    f.write('\ninitial_lr2: ' + str(FLAGS.initial_lr2))
    f.write('\nlr_decay_factor: ' + str(FLAGS.lr_decay_factor))
    f.write('\ndropout_rate: ' + str(FLAGS.dropout_rate))
    f.write('\nl2_reg: ' + str(FLAGS.l2_reg))
    f.write('\nloc_per_grid: ' + ', '.join([str(l) for l in FLAGS.loc_per_grid]))
    f.write('\nreinfornce_reg_w: ' + str(FLAGS.reinfornce_reg_w))
    f.write('\nperFReg_ce_weight: ' + str(FLAGS.perFReg_ce_weight))
    f.write('\nperFReg_reinf_weight: ' + str(FLAGS.perFReg_reinf_weight))
    f.write('\nperFReg_cap: ' + str(FLAGS.perFReg_cap))
    f.write('\noverlap: ' + str(FLAGS.overlap))
    f.write('\ndescr_tag: ' + str(FLAGS.descr_tag))
    f.write('\nwidth_coefficient: ' + str(FLAGS.width_coefficient))
    f.write('\ndepth_coefficient: ' + str(FLAGS.depth_coefficient))
    f.write('\nblock_drop_rate: ' + str(FLAGS.block_drop_rate))
    f.write('\ndepth_divisor: ' + str(FLAGS.depth_divisor))
    f.write('\nactivation: ' + str(FLAGS.activation))
    f.write('\nbatch_norm: ' + str(FLAGS.batch_norm))
    f.write('\ngpus_type: ' + str(FLAGS.gpus_type))
    f.write('\nnum_gpus: ' + str(FLAGS.num_gpus))
    f.write('\nnum_patches_y: ' + str(FLAGS.num_patches_y))
    f.write('\nnum_patches_x: ' + str(FLAGS.num_patches_x))
    f.write('\nls_dim: ' + str(FLAGS.ls_dim))
    f.write('\npos_dim_divisor: ' + str(FLAGS.pos_dim_divisor))
    f.write('\nbase_res_y: ' + str(FLAGS.base_res_y))
    f.write('\nbase_res_x: ' + str(FLAGS.base_res_x))
    f.write('\nnum_res_levels: ' + str(FLAGS.num_res_levels))
    f.write('\nimg_size_y: ' + str(FLAGS.img_size_y))
    f.write('\nimg_size_x: ' + str(FLAGS.img_size_x))
    f.write('\nadv_eval_data: ' + str(FLAGS.adv_eval_data))
    f.write('\nvars_to_exclude: ' + ', '.join([str(l) for l in FLAGS.vars_to_exclude]))
    f.write('\nvars_to_update: ' + ', '.join([str(l) for l in FLAGS.vars_to_update]))
    f.write('\nlr_scedule_2step: ' + str(FLAGS.lr_scedule_2step))
    f.write('\nlr_scedule_1step: ' + str(FLAGS.lr_scedule_1step))
    f.write('\nfeat_weighting: ' + str(FLAGS.feat_weighting))
    f.write('\nlr_boundaries: ' + str(lr_boundaries))
    f.write('\nlr_values: ' + str(lr_values))
    
    f.write('\ncontrastive_loss: ' + str(FLAGS.contrastive_loss))
    f.write('\ncontrastive_margin: ' + str(FLAGS.contrastive_margin))
    f.write('\nl_contrastive: ' + str(FLAGS.l_contrastive))

    f.write('\nbatches_to_profile_range: ' + ', '.join([str(l) for l in FLAGS.batches_to_profile_range]))
    f.write('\nprofile_evaluation: ' + str(FLAGS.profile_evaluation))
    f.write('\nprofile_step: ' + str(FLAGS.profile_step))
    f.write('\nto_train: ' + str(FLAGS.to_train))
    f.write('\nto_evaluate_val: ' + str(FLAGS.to_evaluate_val))
    f.write('\nto_evaluate_train: ' + str(FLAGS.to_evaluate_train))
    f.write('\nbatches_to_time_range: ' + ', '.join([str(l) for l in FLAGS.batches_to_time_range]))
    f.write('\neval_epochs_num: ' + str(FLAGS.eval_epochs_num))
    f.write('\ndont_save_eval_txt: ' + str(FLAGS.dont_save_eval_txt))

    f.write('\nsave_tag: ' + FLAGS.save_tag)
    f.write('\ndata_dir: ' + FLAGS.data_dir)
    f.write('\nckpt_dir: ' + FLAGS.ckpt_dir)
    f.write('\nsummaries_dir: ' + FLAGS.summaries_dir)
    f.write('\nrestore_dir: ' + FLAGS.restore_dir)
    f.write('\nckpt_to_restore: ' + str(FLAGS.ckpt_to_restore))
    f.write('\nresume_training: ' + str(FLAGS.resume_training))
    f.write('\ndictionary_to_restore_from: ' + str(FLAGS.dictionary_to_restore_from))

    f.write('\nlog_frequency: ' + str(FLAGS.log_frequency))
    f.write('\nsummary_frequency: ' + str(FLAGS.summary_frequency))
    f.write('\nckpt_frequency: ' + str(FLAGS.ckpt_frequency))
    f.write('\nckpt_sparse_num: ' + str(FLAGS.ckpt_sparse_num))

    f.write('\nkeep_grads_summary: ' + str(FLAGS.keep_grads_summary))
    f.write('\nkeep_weights_summary: ' + str(FLAGS.keep_weights_summary))
    f.write('\nkeep_activations_summary: ' + str(FLAGS.keep_activations_summary))
    f.close()

class StepTimer(object):
    def __init__(self, step, batch_size):
        """Initialize utility class for measuring processing time
           per image and per processing step.
        Args:
            step: int; global optimization step.
            batch_size: int; number of images processed in
                every optimization step.
        Returns:
            -
        """

        self.step = step
        self.batch_size = batch_size

    def start(self):
        """Reset timer.
        Args:
            -
        Returns:
            -
        """
        
        self.last_iteration = self.step.numpy()
        self.last_time = time.time()

    def time_metrics(self):
        """Calculate processing time per image and per
           processing step.
        Args:
            -
        Returns:
            secs_per_step: float; processing time in seconds,
                per processing step.
            secs_per_img: float; processing time in seconds,
                per image.
        """

        t = time.time()
        secs_per_step = (t - self.last_time) / (self.step.numpy() - self.last_iteration)
        secs_per_img = (t - self.last_time) / ((self.step.numpy() - self.last_iteration) * self.batch_size)

        return secs_per_step, secs_per_img

class distributedTrainer():
    def __init__(self):
        """Initialize class used for training and evaluation of TNet.
        Args:
            -
        Returns:
            -
        """

        # Set up strategy for distributing
        # training to multiple GPUs
        self.strategy = tf.distribute.get_strategy()

        # Set up input streams for training and evaluation
        self.train_steps_per_epoch = NUM_IMAGES['train'] // FLAGS.batch_size
        self.total_train_steps = self.train_steps_per_epoch * FLAGS.num_epochs
        self.eval_steps_per_epoch = NUM_IMAGES['validation'] // FLAGS.batch_size

        self.train_dataset = self.build_distributed_dataset(dataset_type='train', is_training=True, num_epochs=-1)
        self.train_dataset_iter = tf.nest.map_structure(iter, self.train_dataset)
        self.validation_dataset = self.build_distributed_dataset(dataset_type='validation', is_training=True, num_epochs=-1)
        self.validation_dataset_iter = tf.nest.map_structure(iter, self.validation_dataset)

        # Set up learning rate schedule and optimizers
        lr = self.compute_lr_schedule(FLAGS.initial_lr, self.total_train_steps)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        if (FLAGS.two_oprimizers):
            lr2 = self.compute_lr_schedule(FLAGS.initial_lr2, self.total_train_steps)
            self.optimizer2 = tf.keras.optimizers.Adam(lr2)
        self.global_step = self.optimizer.iterations

        # Initialize TNet
        self.model = tf_models.TNet(FLAGS.descr_tag, FLAGS.ls_dim, FLAGS.num_patches_y, FLAGS.num_patches_x, FLAGS.overlap,
                                    FLAGS.num_res_levels, FLAGS.num_classes, FLAGS.base_res_y, FLAGS.base_res_x, FLAGS.dropout_rate,
                                    FLAGS.loc_per_grid, FLAGS.width_coefficient, FLAGS.depth_coefficient, FLAGS.block_drop_rate,
                                    FLAGS.depth_divisor, FLAGS.activation, FLAGS.batch_norm, FLAGS.pos_dim_divisor, FLAGS.feat_weighting)
        # Build TNet by processing a small random batch eagerly
        self.model(tf.random.uniform(shape=[2, FLAGS.img_size_y, FLAGS.img_size_x, tf_input.NUM_CHANNELS]),
                   is_training=True, adv_eval_data=False, step=None, keep_step_summary=False)
        
        # Restore variables if checkpoint is provided
        if (FLAGS.restore_dir != 'No_ckpt'):
            self.restore_ckpt()
        
        # Set up metrics
        self.set_up_metrics()

        # Set up summary writer
        self.summary_writer = tf.summary.create_file_writer(FLAGS.summaries_dir)

        # Set up checkpoint managers to periodically save variables during
        # training. Use a checkpoint manager to save variables at steps
        # close to the current training step, and one for sparse checkpoints,
        # across the whole training duration. They help to recover and
        # evaluate the model at different training steps
        vars_to_save = self.model.variables
        vars_to_save.append(self.global_step)
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, **{f'{adjust_var_name(v.name, self.model.name)}': v for v in vars_to_save})
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, FLAGS.ckpt_dir_latest, max_to_keep=10, checkpoint_name='ckpt')
        self.checkpoint_manager_sparse = tf.train.CheckpointManager(checkpoint, FLAGS.ckpt_dir_sparse, max_to_keep=10, checkpoint_name='ckpt')
        FLAGS.ckpt_frequency_sparse = self.total_train_steps // FLAGS.ckpt_sparse_num

        # Flag for keeping summary of weights and/or gradients during training steps
        self.keep_step_summary = False

        # Specify trainable variables
        if (FLAGS.two_oprimizers):
            # Allocate variables to the 1st optimizer
            self.trainable_variables = []
            for var_to_update in FLAGS.vars_to_update:
                self.trainable_variables.extend([v for v in self.model.trainable_variables if (var_to_update in v.name)])
            # Allocate variables to the 2nd optimizer
            self.trainable_variables2 = self.model.trainable_variables
            for var_to_exclude in self.trainable_variables:
                self.trainable_variables2 = [v for v in self.trainable_variables2 if (var_to_exclude.name not in v.name)]
        elif (FLAGS.vars_to_update):
            self.trainable_variables = []
            for var_to_update in FLAGS.vars_to_update:
                self.trainable_variables.extend([v for v in self.model.trainable_variables if (var_to_update in v.name)])
        else:
            self.trainable_variables = self.model.trainable_variables
        
        # Keep a record of the hyperparameters
        write_hyperparameters(self.total_train_steps, self.optimizer.lr.boundaries, self.optimizer.lr.values)

        # Print a summary of the model
        if ('EfficientNet' in FLAGS.descr_tag):
            print('blocks_num = %d' %self.model.feature_extraction_module.blocks_num)
            print('layers_num = %d' %(self.model.feature_extraction_module.blocks_num * 3 + 2))
        if (FLAGS.num_res_levels > 1):
            self.model.summary()

    def build_distributed_dataset(self, dataset_type, is_training, num_epochs, adv_eval_data=False):
        """Build input stream.
        Args:
            dataset_type: string; type of dataset.
            is_training: boolean; whether the input will be used for training.
            num_epochs: int; number of times to repeat the dataset.
            adv_eval_data: boolean; whether to include information for advanced
                evaluation in the input batches.
        Returns:
            dist_input_data: tf dataset; distributed dataset.
        """
        
        drop_remainder = is_training
        if (FLAGS.num_gpus > 1):
            drop_remainder = True
        input_data = tf_input.input_fn(dataset_type=dataset_type, is_training=is_training, data_dir=FLAGS.data_dir,
                                       batch_size=FLAGS.batch_size, num_epochs=num_epochs, drop_remainder=drop_remainder,
                                       img_size_y=FLAGS.img_size_y, img_size_x=FLAGS.img_size_x, adv_eval_data=adv_eval_data)
        dist_input_data = self.strategy.experimental_distribute_dataset(input_data)

        return dist_input_data

    def compute_lr_schedule(self, init_lr, train_steps):
        """Set learning rate schedule.
        Args:
            init_lr: float; initial learning rate.
            train_steps: int; total number of training steps.
        Returns:
            lr: tf LearningRateSchedule; learning rate schedule.
        """

        if (FLAGS.lr_scedule_2step):
            # Learning rate drops twice
            b1 = 0.7 * train_steps
            b2 = b1 + 0.15 * train_steps
            boundaries = [b1, b2]
            v1 = init_lr
            v2 = v1 * FLAGS.lr_decay_factor
            v3 = v2 * FLAGS.lr_decay_factor
            values = [v1, v2, v3]
        elif (FLAGS.lr_scedule_1step):
            # Learning rate drops once
            b1 = 0.9 * train_steps
            boundaries = [b1]
            v1 = init_lr
            v2 = v1 * FLAGS.lr_decay_factor
            values = [v1, v2]
        else:
            # Learning rate drops once
            b1 = 0.9 * train_steps
            boundaries = [b1]
            v1 = init_lr
            v2 = v1 * FLAGS.lr_decay_factor
            values = [v1, v2]
        
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

        return lr

    def restore_ckpt(self):
        """Restore model checkpoint.
        Args:
            -
        Returns:
            -
        """

        if (FLAGS.ckpt_to_restore == -1):
            # Load the latest checkpoint, if desired checkpoint is not specified
            ckpt_path = tf.train.latest_checkpoint(FLAGS.restore_dir)
        else:
            # Load the specified checkpoint
            ckpt_path = FLAGS.restore_dir + 'ckpt-' + str(FLAGS.ckpt_to_restore)
        if (ckpt_path is not None):
            if (FLAGS.dictionary_to_restore_from is not None):
                # Load checkpoint from dictionary. The dictionary maps model variables
                # to stored variables in the checkpoint, in case they are named differently
                restore_dict = pickle.load(open(FLAGS.dictionary_to_restore_from, "rb"))
                vars_to_restore = self.model.variables
                vars_to_restore.append(self.global_step)
                # Exclude variables that should be randomly initialized,
                # e.g., in case of fine tuning
                if (len(FLAGS.vars_to_exclude) != 0):
                    for var_to_exclude in FLAGS.vars_to_exclude:
                        vars_to_restore = [v for v in vars_to_restore if (var_to_exclude not in v.name)]
                for var in vars_to_restore:
                    var_name = adjust_var_name(var.name, self.model.name)
                    if (var_name in restore_dict):
                        var_value = tf.train.load_variable(ckpt_path, restore_dict[var_name])
                        var.assign(var_value)
                
                print('Pre-trained model restored from %s by using dictionary stored in %s; restarting with global step = %d.'
                    %(FLAGS.restore_dir, FLAGS.dictionary_to_restore_from, self.global_step.numpy()))
            else:
                if (FLAGS.resume_training):
                    # Load variables and global step to resume training;
                    # non-trainable variables, like the ones used in batch
                    # normalization, are restored as well
                    vars_to_restore = self.model.variables
                    vars_to_restore.append(self.global_step)
                    restorer = tf.train.Checkpoint(optimizer=self.optimizer, **{f'{adjust_var_name(v.name, self.model.name)}': v for v in vars_to_restore})
                else:
                    # Load variables from checkpoint. It may be desired some
                    # variables to be excluded and be randomly initialized,
                    # e.g., in case of fine tuning
                    vars_to_restore = self.model.variables
                    if (len(FLAGS.vars_to_exclude) != 0):
                        for var_to_exclude in FLAGS.vars_to_exclude:
                            vars_to_restore = [v for v in vars_to_restore if (var_to_exclude not in v.name)]
                    restorer = tf.train.Checkpoint(**{f'{adjust_var_name(v.name, self.model.name)}': v for v in vars_to_restore})
                
                restorer.restore(ckpt_path).expect_partial()
                
                init_step = int(ckpt_path.split('/')[-1].split('-')[-1])
                print('Pre-trained model restored from %s at step = %d; restarting with global step = %d.'
                      %(FLAGS.restore_dir, init_step, self.global_step.numpy()))
        else:
            print('No checkpoint file found')
            return -1

    def set_up_metrics(self):
        """Set up training and evaluation metrics.
        Args:
            -
        Returns:
            -
        """

        # Measure the mean training loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
        # Measure the top-1 and top-5 accuracy on the training set
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy', dtype=tf.float32)
        self.train_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='train_accuracy_top5', dtype=tf.float32)
        # Measure the average number of attended locations on the training set
        self.train_avg_locs = tf.keras.metrics.Mean(name='train_avg_locs', dtype=tf.float32)

        # Measure the top-1 and top-5 accuracy on the validation set
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy', dtype=tf.float32)
        # Measure the average number of attended locations on the validation set
        self.eval_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='eval_accuracy_top5', dtype=tf.float32)
        self.eval_avg_locs = tf.keras.metrics.Mean(name='eval_avg_locs', dtype=tf.float32)
        
        # Set up helper lists for advanced evaluation
        if (FLAGS.adv_eval_data):
            self.filenames_lst = []
            self.labels_lst = []
            self.labels_text_lst = []
            self.preds_lst = []
            self.preds_labels_text_lst = []
            self.attended_locations_lst = []
            self.location_probs_lst = []
            self.locations_num_lst = []
            self.feat_probs_lst = []
            self.top_1_lst = []
            self.top_5_lst = []

    def contrastive_loss(self, features_prime, labels):
        """Compute contrastive loss.
        Args:
            features_prime: 2-D float Tensor; it contains feature vectors
                from a batch of images. It is of size [batch_size, ls_dim],
                where batch_size is the number of provided images in the
                batch, and ls_dim is the dimensionality of the features'
                latent space.
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
        Returns:
            c_loss: float; contrastive loss.
        """

        # l2 normalize the feature vectors and compute their cosine similarity.
        # cos_sim matrix is of size [batch_size, batch_size]
        features = tf.reshape(features_prime, [-1, self.model.ls_dim])
        features, _ = tf.linalg.normalize(features + 1e-10, ord='euclidean', axis=-1)
        features_T = tf.transpose(features)
        cos_sim = tf.linalg.matmul(features, features_T)

        # Determine if the feature vectors used to calculate each cosine similarity
        # value in cos_sim matrix, belong to the same, or different classes
        labels = tf.expand_dims(tf.squeeze(labels), -1)
        labels = tf.tile(labels, [1, tf.shape(labels)[0]])
        labels_T = tf.transpose(labels)
        c_labels = tf.cast(tf.math.equal(labels, labels_T), tf.float32)

        # Compute contrastive loss
        c_loss = c_labels * (1. - cos_sim) + (1. - c_labels) * tf.math.maximum(cos_sim - FLAGS.contrastive_margin, 0.0)
        c_loss = FLAGS.l_contrastive * tf.reduce_mean(c_loss)

        return c_loss

    def compute_cross_entropy_loss(self, labels, logits):
        """Compute cross entropy loss.
        Args:
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
            logits: 2-D float Tensor; it contains classification logits
                for an image batch. It is of size [batch_size, num_cls],
                where batch_size is the number of images in the batch,
                and num_cls is the number of classes.
        Returns:
            cross_entropy_per_image: 2-D float Tensor; it contains the
                cross entropy loss for every image in a batch. It is of
                size [batch_size, 1], where batch_size is the number of
                images in the batch.
        """
        
        labels = tf.cast(labels, tf.int32)
        cross_entropy_per_image = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_per_image = tf.expand_dims(cross_entropy_per_image, axis=1)

        return cross_entropy_per_image

    def compute_total_loss(self, logits, labels, location_log_probs,
                           logits_per_feature, logprobs_per_feature):
        """Compute training loss.
        Args:
            logits: 2-D float Tensor; it contains classification logits
                for an image batch. It is of size [batch_size, num_cls],
                where batch_size is the number of images in the batch,
                and num_cls is the number of classes.
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
            location_log_probs: 2-D float Tensor; it contains the sum of
                log probabilities of attended locations for the images in
                a batch. It is of size [batch_size, 1], where batch_size
                is the number of images in the batch.
            logits_per_feature: list of 2-D float Tensors; each Tensor
                contains classification logits from individual feature
                vectors extracted at each TNet processing level for a batch
                of images. The list has num_res_levels entries, and each
                entry is of size [loc_per_lvl*batch_size, num_cls], where
                num_res_levels is the number of processing levels that
                TNet went through, loc_per_lvl is the number of locations
                attended for each image at each processing level (may be
                different for different levels), batch_size is the number
                of images in the batch, and num_cls is the number of classes.
            logprobs_per_feature: list of 2-D float Tensors; each Tensor
                contains the log probabilities of attended locations from
                a different TNet processing level for a batch of images.
                The list has num_res_levels-1 entries, and each entry is
                of size [loc_per_lvl*batch_size, 1], where num_res_levels
                is the number of processing levels that TNet went through,
                loc_per_lvl is the number of locations attended for each
                image at each processing level (may be different for different
                levels), and batch_size is the number of images in the batch.
        Returns:
            total_loss: float; training loss.
        """

        # Calculate the number of feature vectors extracted for each image,
        # until processing level perFReg_cap
        bsz = tf.cast((FLAGS.batch_size / FLAGS.num_gpus), tf.float32)
        loc_per_lvl = 1
        total_loc_per_img = 1
        if (FLAGS.perFReg_cap < 0):
            pfr_cap = FLAGS.num_res_levels
        else:
            pfr_cap = FLAGS.perFReg_cap
        for i in range(pfr_cap - 1):
            loc_per_lvl *= FLAGS.loc_per_grid[i]
            total_loc_per_img += loc_per_lvl
        total_loc_per_img = tf.cast(total_loc_per_img, tf.float32)

        # Calculate cross entropy loss
        total_loss = 0.0
        cross_entropy_per_image = self.compute_cross_entropy_loss(labels, logits)
        cross_entropy = FLAGS.perFReg_ce_weight * (1.0/bsz) * tf.reduce_sum(cross_entropy_per_image)
        total_loss += cross_entropy

        # Calculate REINFORCE loss
        if (FLAGS.num_res_levels > 1):
            Reward = tf.cast(tf.math.in_top_k(predictions=logits, targets=tf.squeeze(tf.cast(labels, tf.int32)), k=1), tf.float32)
            Reward = tf.expand_dims(Reward, axis=-1)
            reinforce_term = tf.reduce_sum(tf.stop_gradient(Reward-self.model.baseline_var) * (-location_log_probs))
            reinforce_term = FLAGS.reinfornce_reg_w * FLAGS.perFReg_reinf_weight * (1.0/bsz) * reinforce_term
            total_baseline_step = FLAGS.perFReg_reinf_weight * (1.0/bsz) * tf.reduce_sum(Reward)
            total_loss += reinforce_term
        
        # Calculate per-feature regularization loss terms
        if (FLAGS.num_res_levels > 1):
            cross_entropy_perF = 0.0
            reinforce_term_perF = 0.0
            baseline_step = 0.0
            loc_per_lvl = 1
            for i in range(pfr_cap):
                # Calculate cross entropy loss per feature vector
                labels_per_feature = tf.repeat(labels, repeats=tf.cast(loc_per_lvl, tf.int32), axis=0)
                cross_entropy_per_image = self.compute_cross_entropy_loss(labels_per_feature, logits_per_feature[i])
                cross_entropy_perF += tf.reduce_sum(cross_entropy_per_image)
                if (i > 0):
                    # Calculate REINFORCE loss per feature vector
                    Reward = tf.cast(tf.math.in_top_k(predictions=logits_per_feature[i], targets=tf.squeeze(tf.cast(labels_per_feature, tf.int32)), k=1), tf.float32)
                    Reward = tf.expand_dims(Reward, axis=-1)
                    reinforce_term_perF += tf.reduce_sum(tf.stop_gradient(Reward-self.model.baseline_var) * (-logprobs_per_feature[i-1]))
                    baseline_step += tf.reduce_sum(Reward)
                if (i < (pfr_cap - 1)):
                    loc_per_lvl *= int(FLAGS.loc_per_grid[i])

            cross_entropy_perF = (1.0-FLAGS.perFReg_ce_weight) * (1.0/(total_loc_per_img*bsz)) * cross_entropy_perF
            total_loss += cross_entropy_perF
            
            reinforce_term_perF = FLAGS.reinfornce_reg_w * (1.0-FLAGS.perFReg_reinf_weight) * (1.0/((total_loc_per_img-1)*bsz)) * reinforce_term_perF
            total_baseline_step += (1.0-FLAGS.perFReg_reinf_weight) * (1.0/((total_loc_per_img-1)*bsz)) * baseline_step
            self.model.baseline_var.assign(self.model.baseline_var * BL_DECAY + total_baseline_step * (1.0 - BL_DECAY))
            total_loss += reinforce_term_perF
        
        # Calculate l2 regularization loss
        if (FLAGS.l2_reg > 0.):
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'batch_norm' not in v.name])
            total_loss += FLAGS.l2_reg * 2 * l2_loss
        
        return total_loss

    def update_train_metrics(self, logits, labels, total_loss, location_num_per_img):
        """Update training metrics.
        Args:
            logits: 2-D float Tensor; it contains classification logits
                for an image batch. It is of size [batch_size, num_cls],
                where batch_size is the number of images in the batch,
                and num_cls is the number of classes.
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
            total_loss: float; total training loss.
            location_num_per_img: 2-D float Tensor; it contains the total
                number of attended locations per image (the processing of
                the downsampled version of each image in the 1st processing
                level is not counted). It is of size [batch_size, 1], where
                batch_size is the number of images in the batch.
        Returns:
            -
        """

        self.train_loss.update_state(total_loss * FLAGS.num_gpus)
        self.train_accuracy.update_state(labels, logits)
        self.train_accuracy_top5.update_state(labels, logits)
        self.train_avg_locs.update_state(location_num_per_img)

    @tf.function
    def train_step(self, input_batch, step, keep_step_summary):
        """Distribute training step among GPUs.
        Args:
            input_batch: tuple; it contains the following entries:
                images: 4-D float Tensor; it contains the image batch to be
                    processed. It is of size [batch_size, H, W, C], where
                    batch_size is the number of images in the batch, H is
                    their height, W is their width, and C is the number of
                    channels.
                labels: 2-D float Tensor; it contains the labels of the
                    images in the batch. It is of size [batch_size, 1],
                    where batch_size is the number of images in the batch.
            step: int; global optimization step.
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            summary_grads (optional): list; it contains Tensors with the
                gradients of the trainable variables.
        """

        def step_fn(step_input_batch, step, keep_step_summary):
            """Perform a distributed training step.
            Args:
                step_input_batch: tuple; it contains the following entries:
                    images: 4-D float Tensor; it contains an image batch
                        distributed among GPUs. It is of size
                        [batch_size, H, W, C], where batch_size is the number
                        of images in the distributed batch, H is their height,
                        W is their width, and C is the number of channels.
                    labels: 2-D float Tensor; it contains the labels of the
                        images in the batch. It is of size [batch_size, 1],
                        where batch_size is the number of images in the batch.
                step: int; global optimization step.
                keep_step_summary: boolean; whether to keep tf summaries.
            Returns:
                summary_grads (optional): list; it contains Tensors with the
                    gradients of the trainable variables.
            """

            images, labels = step_input_batch
            labels = tf.squeeze(labels)
            # Repeat the batch for multiple Monte Carlo samples
            if (FLAGS.num_samples > 1):
                images = tf.tile(images, [FLAGS.num_samples, 1, 1, 1])
                labels = tf.tile(labels, [FLAGS.num_samples])

            with tf.GradientTape(persistent=FLAGS.two_oprimizers) as tape:
                # Process the provided image batch
                with self.summary_writer.as_default():
                    ret_lst = self.model(images, is_training=True, adv_eval_data=False, step=step,
                                         keep_step_summary=(FLAGS.keep_activations_summary and keep_step_summary))
                logits = ret_lst[0]
                location_num_per_img = ret_lst[1]
                location_log_probs = ret_lst[2]
                logits_per_feature = ret_lst[3]
                logprobs_per_feature = ret_lst[4]
                features_prime = ret_lst[5]
                
                # Compute total loss
                total_loss = self.compute_total_loss(logits, labels, location_log_probs,
                                                     logits_per_feature, logprobs_per_feature)
                if (FLAGS.contrastive_loss):
                    c_loss = self.contrastive_loss(features_prime, labels)
                    total_loss += c_loss
                total_loss = total_loss * (1.0 / FLAGS.num_gpus)

            # Calculate gradients
            if (not FLAGS.two_oprimizers):
                gradients = tape.gradient(total_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            else:
                gradients = tape.gradient(total_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                gradients2 = tape.gradient(total_loss, self.trainable_variables2)
                self.optimizer2.apply_gradients(zip(gradients2, self.trainable_variables2))

                gradients.extend(gradients2)
            
            # Update train metrics
            self.update_train_metrics(logits, labels, total_loss, location_num_per_img)
            
            summary_grads = None
            if (FLAGS.keep_grads_summary and keep_step_summary):
                summary_grads = gradients
            
            return summary_grads
        
        summary_grads = self.strategy.run(step_fn, args=(input_batch, step, keep_step_summary))

        return summary_grads

    def update_eval_metrics(self, logits, labels, location_num_per_img):
        """Update evaluation metrics.
        Args:
            logits: 2-D float Tensor; it contains classification logits
                for an image batch. It is of size [batch_size, num_cls],
                where batch_size is the number of images in the batch,
                and num_cls is the number of classes.
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
            location_num_per_img: 2-D float Tensor; it contains the total
                number of attended locations per image (the processing of
                the downsampled version of each image in the 1st processing
                level is not counted). It is of size [batch_size, 1], where
                batch_size is the number of images in the batch.
        Returns:
            -
        """

        self.eval_accuracy.update_state(labels, logits)
        self.eval_accuracy_top5.update_state(labels, logits)
        self.eval_avg_locs.update_state(location_num_per_img)

    @tf.function
    def eval_step(self, input_batch, is_training=False, adv_eval_data=False):
        """Distribute evaluation step among GPUs.
        Args:
            input_batch: tuple; it contains the following entries:
                images: 4-D float Tensor; it contains the image batch to be
                    processed. It is of size [batch_size, H, W, C], where
                    batch_size is the number of images in the batch, H is
                    their height, W is their width, and C is the number of
                    channels.
                labels: 2-D float Tensor; it contains the labels of the
                    images in the batch. It is of size [batch_size, 1],
                    where batch_size is the number of images in the batch.
            is_training: boolean; whether the model is in training phase.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
        Returns:
            ret_obj (optional): tuple; it contains the following entries:
                logits: 2-D float Tensor; it contains classification logits
                    for an image batch. It is of size [batch_size, num_cls],
                    where batch_size is the number of images in the batch,
                    and num_cls is the number of classes.
                attended_locations: 2-D float Tensor; it indicates which
                    locations are selected each time the location module
                    is applied during the processing of each image in a
                    batch. Candidate locations within each attention grid
                    are flattened in a left to right and top to bottom fashion,
                    and the value 1 is assigned to selected locations, while 0
                    is assigned to the rest. It is used for advanced evaluation
                    of the model. The Tensor is of size
                    [batch_size, num_att_per_img*locs_num], where batch_size is
                    the number of images in the batch, num_att_per_img is the
                    total number of times the location module is applied
                    during the processing of each image, and locs_num is the
                    number of candidate locations within each attention grid.
                location_probs: 2-D float Tensor; it contains the attention
                    probabilities of all candidate locations considered during
                    the processing of each image in a batch. It is used for
                    advanced evaluation of the model. The Tensor is of size
                    [batch_size, num_att_per_img*locs_num], where batch_size is
                    the number of images in the batch, num_att_per_img is the
                    total number of times the location module is applied during
                    the processing of each image, and locs_num is the number of
                    candidate locations within each attention grid.
                feat_probs: 3-D float Tensor; it contains weighting probabilities
                    for the feature vectors extracted from each image in a batch.
                    It is of size [batch_size, N, 1], where batch_size is the
                    number of images in the batch, and N is the total number of
                    extracted features from each image (N-1 attended locations,
                    plus the feature vector from the 1st processing level).
        """

        def step_fn(step_input_batch, is_training, adv_eval_data):
            """Perform a distributed evaluation step.
            Args:
                step_input_batch: tuple; it contains the following entries:
                    images: 4-D float Tensor; it contains an image batch
                        distributed among GPUs. It is of size
                        [batch_size, H, W, C], where batch_size is the number
                        of images in the distributed batch, H is their height,
                        W is their width, and C is the number of channels.
                    labels: 2-D float Tensor; it contains the labels of the
                        images in the batch. It is of size [batch_size, 1],
                        where batch_size is the number of images in the batch.
                is_training: boolean; whether the model is in training phase.
                adv_eval_data: boolean; whether to return additional information
                    that is used for advanced evaluation of the model.
            Returns:
                ret_obj (optional): tuple; it contains the following entries:
                    logits: 2-D float Tensor; it contains classification logits
                        for an image batch. It is of size [batch_size, num_cls],
                        where batch_size is the number of images in the batch,
                        and num_cls is the number of classes.
                    attended_locations: 2-D float Tensor; it indicates which
                        locations are selected each time the location module
                        is applied during the processing of each image in a
                        batch. Candidate locations within each attention grid
                        are flattened in a left to right and top to bottom fashion,
                        and the value 1 is assigned to selected locations, while 0
                        is assigned to the rest. It is used for advanced evaluation
                        of the model. The Tensor is of size
                        [batch_size, num_att_per_img*locs_num], where batch_size is
                        the number of images in the batch, num_att_per_img is the
                        total number of times the location module is applied
                        during the processing of each image, and locs_num is the
                        number of candidate locations within each attention grid.
                    location_probs: 2-D float Tensor; it contains the attention
                        probabilities of all candidate locations considered during
                        the processing of each image in a batch. It is used for
                        advanced evaluation of the model. The Tensor is of size
                        [batch_size, num_att_per_img*locs_num], where batch_size is
                        the number of provided images to classify, num_att_per_img
                        is the total number of times the location module is applied
                        during the processing of each image, and locs_num is the
                        number of candidate locations within each attention grid.
                    feat_probs: 3-D float Tensor; it contains weighting probabilities
                        for the feature vectors extracted from each image in a batch.
                        It is of size [batch_size, N, 1], where batch_size is the
                        number of images in the batch, and N is the total number of
                        extracted features from each image (N-1 attended locations,
                        plus the feature vector from the 1st processing level).
            """

            if (not adv_eval_data):
                images, labels = step_input_batch
            else:
                (images, labels, _, _) = step_input_batch
            labels = tf.squeeze(labels)

            # Process the provided image batch
            ret_lst = self.model(images, is_training=is_training, adv_eval_data=adv_eval_data,
                                 step=None, keep_step_summary=False)
            logits = ret_lst[0]
            location_num_per_img = ret_lst[1]
            if (adv_eval_data):
                attended_locations = ret_lst[2]
                location_probs = ret_lst[3]
                feat_probs = ret_lst[4]
            
            # Update evaluation metrics
            self.update_eval_metrics(logits, labels, location_num_per_img)
        
            if (adv_eval_data):
                return logits, attended_locations, location_probs, feat_probs
            else:
                return None
        
        ret_obj = self.strategy.run(step_fn, args=(input_batch, is_training, adv_eval_data))

        return ret_obj

    def reset_train_metrics(self):
        """Reset training metrics.
        Args:
            -
        Returns:
            -
        """

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.train_accuracy_top5.reset_states()
        self.train_avg_locs.reset_states()
    
    def reset_eval_metrics(self):
        """Reset evaluation metrics.
        Args:
            -
        Returns:
            -
        """

        self.eval_accuracy.reset_states()
        self.eval_accuracy_top5.reset_states()
        self.eval_avg_locs.reset_states()

        if (FLAGS.adv_eval_data):
            self.filenames_lst = []
            self.labels_lst = []
            self.labels_text_lst = []
            self.preds_lst = []
            self.preds_labels_text_lst = []
            self.attended_locations_lst = []
            self.location_probs_lst = []
            self.locations_num_lst = []
            self.feat_probs_lst = []
            self.top_1_lst = []
            self.top_5_lst = []

    def reset_log(self):
        """Reset training metrics and timer.
        Args:
            -
        Returns:
            -
        """

        self.reset_train_metrics()
        self.timer.start()
    
    def write_log(self, step):
        """Print training metrics.
        Args:
            step: int; global optimization step.
        Returns:
            -
        """

        secs_per_step, secs_per_img = self.timer.time_metrics()
        format_str = (f'step %d, LVL_{FLAGS.num_res_levels}: total_loss = %.2f; acc1 = %.2f, acc5 = %.2f, '
                      f'avg_locs = %.2f; (%.3f sec/step, %.3f sec/img)')
        print (format_str %(step, self.train_loss.result(), self.train_accuracy.result(),
                            self.train_accuracy_top5.result(), self.train_avg_locs.result(),
                            secs_per_step, secs_per_img))
    
    def reset_summary(self):
        """Reset evaluation metrics and summary variable.
        Args:
            -
        Returns:
            -
        """

        self.reset_eval_metrics()
        self.keep_step_summary = False

    def grads_summary(self, gradients, step):
        """Keep tf summaries for gradients.
        Args:
            gradients: list; it contains Tensors with the gradients
                of the trainable variables.
            step: int; global optimization step.
        Returns:
            -
        """

        for grad, var in zip(gradients, self.trainable_variables):
            if grad is not None:
                reduced_grad = self.strategy.reduce(tf.distribute.ReduceOp.SUM, grad, axis=None)
                tf.summary.histogram(adjust_var_name(var.name) + '/gradients', reduced_grad, step=step)
    
    def vars_summary(self, step):
        """Keep tf summaries for variables.
        Args:
            step: int; global optimization step.
        Returns:
            -
        """

        for var in self.model.variables:
            tf.summary.histogram(adjust_var_name(var.name), var, step=step)

    def write_summary(self, gradients, step):
        """Keep tf summaries.
        Args:
            gradients: list; it contains Tensors with the gradients
                of the trainable variables.
            step: int; global optimization step.
        Returns:
            -
        """

        # Perform an evaluation step on the validation set
        self.eval_step(next(self.validation_dataset_iter))

        # Keep tf summaries for gradients and model variables
        with self.summary_writer.as_default():
            if (FLAGS.keep_grads_summary):
                self.grads_summary(gradients, step)
            if (FLAGS.keep_weights_summary):
                self.vars_summary(step)
            
            # Keep tf summaries for training and evaluation metrics
            tf.summary.scalar("train_step_accuracy_top1", self.train_accuracy.result(), step=step)
            tf.summary.scalar("train_step_accuracy_top5", self.train_accuracy_top5.result(), step=step)
            tf.summary.scalar("train_locations_avg_num", self.train_avg_locs.result(), step=step)

            tf.summary.scalar("dev_step_accuracy_top1", self.eval_accuracy.result(), step=step)
            tf.summary.scalar("dev_step_accuracy_top5", self.eval_accuracy_top5.result(), step=step)
            tf.summary.scalar("dev_locations_avg_num", self.eval_avg_locs.result(), step=step)

            tf.summary.scalar("train_loss", self.train_loss.result(), step=step)
            secs_per_step, secs_per_img = self.timer.time_metrics()
            tf.summary.scalar("seconds_per_step", secs_per_step, step=step)
            tf.summary.scalar("seconds_per_image", secs_per_img, step=step)
            tf.summary.scalar("learning_rate", self.optimizer.learning_rate(self.global_step).numpy(), step=step)

            self.summary_writer.flush()

    def train(self):
        """Conduct the training loop.
        Args:
            -
        Returns:
            -
        """

        # Set up a timer to time training steps
        self.timer = StepTimer(self.global_step, FLAGS.batch_size)
        # Reset training metrics that are periodically
        # printed on the screen
        self.reset_log()
        # Reset evaluation metrics that are periodically
        # written as tf summaries
        self.reset_summary()
        # Iterate over the training set
        while (self.global_step.numpy() < self.total_train_steps):
            step = self.global_step.numpy()
            if (step % FLAGS.summary_frequency == 0):
                self.keep_step_summary = True
            
            # Perform a training step
            summary_grads = self.train_step(next(self.train_dataset_iter), self.global_step, self.keep_step_summary)
            
            # Keep tf summaries periodically
            if (step % FLAGS.summary_frequency == 0):
                self.write_summary(summary_grads, step)
                self.reset_summary()
            
            # Print training metrics periodically
            if ((step % FLAGS.log_frequency) == 0):
                self.write_log(step)
                self.reset_log()
            
            # Keep checkpoints periodically
            if ((step % FLAGS.ckpt_frequency) == 0 or (step == self.total_train_steps - 1)):
                self.checkpoint_manager.save(step)
            if ((step % FLAGS.ckpt_frequency_sparse) == 0):
                self.checkpoint_manager_sparse.save(step)

    def update_adv_eval_data(self, step_batch, ret_obj):
        """Update advanced evaluation data.
        Args:
            step_batch: tuple; it contains the following entries:
                images: 4-D float Tensor; it contains an image batch. It
                    is of size [batch_size, H, W, C], where batch_size is
                    the number of images in the batch, H is their height,
                    W is their width, and C is the number of channels.
                labels: 2-D float Tensor; it contains the labels of the
                    images in a batch. It is of size [batch_size, 1],
                    where batch_size is the number of images in the batch.
                filenames: 2-D string Tensor; it contains the filenames of
                    the images in a batch. It is of size [batch_size, 1],
                    where batch_size is the number of images in the batch.
                labels_text: 2-D float Tensor; it contains human readable
                    labels of the images in a batch. It is of size
                    [batch_size, 1], where batch_size is the number of
                    images in the batch.
            ret_obj: tuple; it contains the following entries:
                logits: 2-D float Tensor; it contains classification logits
                    for an image batch. It is of size [batch_size, num_cls],
                    where batch_size is the number of images in the batch,
                    and num_cls is the number of classes.
                attended_locations: 2-D float Tensor; it indicates which
                    locations are selected each time the location module
                    is applied during the processing of each image in a
                    batch. Candidate locations within each attention grid
                    are flattened in a left to right and top to bottom fashion,
                    and the value 1 is assigned to selected locations, while 0
                    is assigned to the rest. The Tensor is of size
                    [batch_size, num_att_per_img*locs_num], where batch_size is
                    the number of images in the batch, num_att_per_img is the
                    total number of times the location module is applied
                    during the processing of each image, and locs_num is the
                    number of candidate locations within each attention grid.
                location_probs: 2-D float Tensor; it contains the attention
                    probabilities of all candidate locations considered during
                    the processing of each image in a batch. The Tensor is of
                    size [batch_size, num_att_per_img*locs_num], where batch_size
                    is the number of images in the batch, num_att_per_img is
                    the total number of times the location module is applied
                    during the processing of each image, and locs_num is the
                    number of candidate locations within each attention grid.
                feat_probs: 3-D float Tensor; it contains weighting probabilities
                    for the feature vectors extracted from each image in a batch.
                    It is of size [batch_size, N, 1], where batch_size is the
                    number of images in the batch, and N is the total number of
                    extracted features from each image (N-1 attended locations,
                    plus the feature vector from the 1st processing level).
        Returns:
            -
        """

        # Keep data related to processed images in lists
        (images, labels, filenames, labels_text) = step_batch

        self.filenames_lst.extend([tf.squeeze(e) for e in self.strategy.experimental_local_results(filenames)])
        self.labels_lst.extend([tf.squeeze(e) for e in self.strategy.experimental_local_results(labels)])
        self.labels_text_lst.extend([tf.squeeze(e) for e in self.strategy.experimental_local_results(labels_text)])
        
        logits, attended_locations, location_probs, feat_probs = ret_obj

        logits = self.strategy.experimental_local_results(logits)
        self.preds_lst.extend([tf.cast(tf.math.argmax(l, axis=1), tf.int32) for l in logits])

        labels = self.strategy.experimental_local_results(labels)
        self.top_1_lst.extend([tf.cast(tf.math.in_top_k(predictions=lg, targets=tf.squeeze(tf.cast(lb, tf.int32)), k=1), tf.int32) for lg, lb in zip(logits, labels)])
        self.top_5_lst.extend([tf.cast(tf.math.in_top_k(predictions=lg, targets=tf.squeeze(tf.cast(lb, tf.int32)), k=5), tf.int32) for lg, lb in zip(logits, labels)])
        
        attended_locations = self.strategy.experimental_local_results(attended_locations)
        self.locations_num_lst.extend([tf.math.reduce_sum(l, axis=1) for l in attended_locations])
        for e in attended_locations:
            self.attended_locations_lst.extend(tf.split(e, tf.shape(e)[0].numpy(), axis=0))
        
        location_probs = self.strategy.experimental_local_results(location_probs)
        for e in location_probs:
            self.location_probs_lst.extend(tf.split(e, tf.shape(e)[0].numpy(), axis=0))

        feat_probs = self.strategy.experimental_local_results(tf.squeeze(feat_probs))
        for e in feat_probs:
            self.feat_probs_lst.extend(tf.split(e, tf.shape(e)[0].numpy(), axis=0))
        
    def save_adv_eval_data(self, dataset_type):
        """Save advanced evaluation data.
        Args:
            dataset_type: string; type of dataset.
        Returns:
            -
        """

        # Save to an excel file data related to processed images
        labels_to_human = build_labels_to_human()

        fnames = tf.concat(self.filenames_lst, axis=0).numpy()
        fnames = [f.decode("utf-8") for f in fnames]

        labels = tf.concat(self.labels_lst, axis=0).numpy()
        human_labels = np.array([labels_to_human[c] for c in labels])

        preds_cls = tf.concat(self.preds_lst, axis=0).numpy()
        human_preds = np.array([labels_to_human[c] for c in preds_cls])
        
        locations_num = tf.concat(self.locations_num_lst, axis=0).numpy()
        locations = [l.numpy() for l in self.attended_locations_lst]
        locations_probs = [l.numpy() for l in self.location_probs_lst]
        feat_probs = [l.numpy() for l in self.feat_probs_lst]
        
        top1_mask = tf.concat(self.top_1_lst, axis=0).numpy()
        top5_mask = tf.concat(self.top_5_lst, axis=0).numpy()

        cols = ['fnames', 'labels', 'human_labels', 'predicted_class', 'human_preds',
                'locations_num', 'locations', 'locations_probs', 'feat_probs', 'top1_mask', 'top5_mask']
        df = pd.DataFrame({cols[0]: fnames,
                           cols[1]: labels,
                           cols[2]: human_labels,
                           cols[3]: preds_cls,
                           cols[4]: human_preds,
                           cols[5]: locations_num,
                           cols[6]: locations,
                           cols[7]: locations_probs,
                           cols[8]: feat_probs,
                           cols[9]: top1_mask,
                           cols[10]: top5_mask})

        df.to_excel(FLAGS.ckpt_dir + 'btr_' + '_'.join([str(l) for l in FLAGS.batches_to_time_range]) + '_lpg_' + '_'.join([str(l) for l in FLAGS.loc_per_grid]) + '_imsz_' + str(FLAGS.img_size_y) + '_' + dataset_type + '_' + 'adv_eval_data.xlsx', sheet_name='sheet1', index=False)
        print('Advanced eval data - Saved.')

    def evaluate(self, dataset_type, is_training=False, num_epochs=1, adv_eval_data=False):
        """Evaluate TNet.
        Args:
            dataset_type: string; type of dataset.
            is_training: boolean; whether the input will go through the data
                augmentation used for training.
            num_epochs: int; number of epochs during evaluation.
            adv_eval_data: boolean; whether to perform advanced evaluation of
                the model.
        Returns:
            -
        """

        # Set up input stream
        eval_dataset = self.build_distributed_dataset(dataset_type=dataset_type, is_training=is_training, num_epochs=num_epochs, adv_eval_data=adv_eval_data)
        
        # Reset evaluation metrics
        self.reset_eval_metrics()

        # Keep timing metrics periodically, in
        # order to calculate related statistics
        if (FLAGS.profile_step):
            time_lst = []
            total_p_step = 0
        
        # Iterate over the evaluation dataset
        step = 0
        for step_batch in tqdm(eval_dataset):

            # Use tf profiler
            if (FLAGS.profile_evaluation):
                if (step == FLAGS.batches_to_profile_range[0]):
                    print('Profiling starts')
                    tf.profiler.experimental.start(FLAGS.summaries_dir)
                elif (step == FLAGS.batches_to_profile_range[1]):
                    print('Profiling ends')
                    tf.profiler.experimental.stop()
                    break

            if (step == FLAGS.batches_to_time_range[0]):
                init_time = time.time()
            if (step == FLAGS.batches_to_time_range[1]):
                end_time = time.time()
                break
            
            # Perform an evaluation step
            ret_obj = self.eval_step(step_batch, False, adv_eval_data)

            # Update advanced evaluation data
            if (adv_eval_data):
                self.update_adv_eval_data(step_batch, ret_obj)
            
            # Keep timing metrics periodically
            if (FLAGS.profile_step and (step >= FLAGS.batches_to_time_range[0])):
                total_p_step += 1
                if ((total_p_step % FLAGS.profile_step) == 0):
                    end_time = time.time()
                    throughput = (FLAGS.profile_step * (FLAGS.batch_size / FLAGS.num_gpus)) / (end_time - init_time)
                    throughput_ms = throughput / 1000.0
                    time_lst.append(1./throughput_ms)
                    init_time = time.time()

            step += 1
        
        # Print and save evaluation metrics
        print('\n')
        print('accuracy @ 1 = %.3f%%' %(self.eval_accuracy.result()*100))
        print('accuracy @ 5 = %.3f%%' %(self.eval_accuracy_top5.result()*100))
        print('locations number = %.2f' %self.eval_avg_locs.result())
        if (not FLAGS.dont_save_eval_txt):
            fname = (str(FLAGS.gpus_type) + 'x' + str(FLAGS.num_gpus) + '_' + str(FLAGS.save_tag) + '_bs' + str(FLAGS.batch_size) +
                    '_loc' + '_'.join([str(l) for l in FLAGS.loc_per_grid]) + '_dt_' + dataset_type + '_isT' + str(is_training) + '.txt')
            f = open(os.path.join(FLAGS.ckpt_dir, fname), 'w')
            f.write('Accuracy @ 1: ' + str(self.eval_accuracy.result().numpy()*100) + '%')
            f.write('\nAccuracy @ 5: ' + str(self.eval_accuracy_top5.result().numpy()*100) + '%\n')
            f.write('\nLocations number = ' + str(self.eval_avg_locs.result()))
            f.close()

        # Print and save timing metrics
        if (FLAGS.profile_step):
            print('\nMean time: %.5f msec/im' %np.mean(time_lst))
            print('Std time: %.5f msec/im' %np.std(time_lst))
            f = open(os.path.join(FLAGS.ckpt_dir, 'loc' + '_'.join([str(l) for l in FLAGS.loc_per_grid]) + '_profile_step.txt'), 'w')
            f.write('Mean time: ' + str(np.mean(time_lst)) + ' msec/im')
            f.write('\nStd time: ' + str(np.std(time_lst)) + ' msec/im')
            f.close()

        # Save advanced evaluation data
        if (adv_eval_data):
            self.save_adv_eval_data(dataset_type)

def main(argv=None):
    """Main function for TNet training.
    Args:
        -
    Returns:
        -
    """

    # Set up directories for checkpoints and summaries
    set_up_dirs()

    # Define strategy for distributing training to GPUs
    strategy = tf.distribute.MirroredStrategy(devices=["device:GPU:%d" %i for i in range(FLAGS.num_gpus)])

    # Move under strategy scope all code that involves distributed processing,
    # e.g. building the model (variables should be mirrored)
    with strategy.scope():
        trainer = distributedTrainer()
        if (FLAGS.to_train):
            trainer.train()

        # Evaluation after training
        if (FLAGS.to_evaluate_val):
            print("\n-------------Computing validation set evaluation metrics-------------\n")
            trainer.evaluate(dataset_type='validation', is_training=False, num_epochs=FLAGS.eval_epochs_num, adv_eval_data=FLAGS.adv_eval_data)
        if (FLAGS.to_evaluate_train):
            print("\n-------------Computing training set evaluation metrics-------------\n")
            trainer.evaluate(dataset_type='train', is_training=False, num_epochs=FLAGS.eval_epochs_num, adv_eval_data=FLAGS.adv_eval_data)
        
        print('End.')

if __name__ == '__main__':
    main()
