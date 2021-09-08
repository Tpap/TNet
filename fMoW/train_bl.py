"""Train and evaluate baseline networks.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import time
import pickle
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import input_fMoW as tf_input
import models_fMoW as tf_models
from models_fMoW import adjust_var_name



parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, help='Global batch size (recommended to be divisible by the number of GPUs).')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--num_classes', type=int, default=62, help='Number of classes.')
parser.add_argument('--initial_lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Learning rate decay factor.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout drop probability.')
parser.add_argument('--ls_dim', type=int, default=1280, help='Latent space dimensionality.')
parser.add_argument('--l2_reg', type=float, default=0., help='Importance weight for l2 regularization.')
parser.add_argument('--descr_tag', type=str, help='Description of the model to build')
parser.add_argument('--width_coefficient', type=float, default=1.0, help='Coefficient for scaling the width of the feature extraction netwrok.')
parser.add_argument('--depth_coefficient', type=float, default=1.0, help='Coefficient for scaling the depth of the feature extraction netwrok.')
parser.add_argument('--block_drop_rate', type=float, default=0.2, help='The maximum probability for dropping model blocks during feature extraction (stochastic depth parameter).')
parser.add_argument('--depth_divisor', type=int, default=8, help='Determines quantization during depth scaling.')
parser.add_argument('--activation', type=str, default='swish', help='Type of activation to be used.')
parser.add_argument('--batch_norm', action='store_true', help='Whether to use Batch Normalization.')
parser.add_argument('--gpus_type', type=str, default='NotReported', help='Type of GPUs in use')
parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use.')
parser.add_argument('--img_size_y', type=int, default=224, help='Resolution of the input image in the vertical dimension.')
parser.add_argument('--img_size_x', type=int, default=224, help='Resolution of the input image in the horizontal dimension.')
parser.add_argument('--lr_scedule_2step', action='store_true', help='Whether to drop learning rate 2 times.')
parser.add_argument('--lr_scedule_1step', action='store_true', help='Whether to drop learning rate once.')


parser.add_argument('--batches_to_profile_range', nargs='+', type=float, default=[-1, -1], help='Range of batches to profile.')
parser.add_argument('--profile_evaluation', action='store_true', help='Whether to profile final evaluation by using batches in batches_to_profile_range range.')
parser.add_argument('--profile_step', type=float, default=0., help='Step for profiling; used to calculate statistics, mean and std.')
parser.add_argument('--to_train', action='store_true', help='Whether to train the model.')
parser.add_argument('--to_evaluate_val', action='store_true', help='Whether to evaluate the model with validation data.')
parser.add_argument('--to_evaluate_test', action='store_true', help='Whether to evaluate the model with test data.')
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
parser.add_argument('--summary_frequency', type=int, default=2000, help='How often to log results to the summary file.') #default=200
parser.add_argument('--ckpt_frequency', type=int, default=5000, help='How often to make a checkpoint.')
parser.add_argument('--ckpt_sparse_num', type=int, default=10, help='How many checkpoints to make sparsily throughout training.')

parser.add_argument('--keep_grads_summary', action='store_true', help='Keep tf summary for the trainable weights gradients.')
parser.add_argument('--keep_weights_summary', action='store_true', help='Keep tf summary for the model variables.')
parser.add_argument('--keep_activations_summary', action='store_true', help='Keep tf summary for model activations.')

FLAGS = parser.parse_args()

category_names = ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site',
                  'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant',
                  'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond',
                  'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage',
                  'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway',
                  'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool',
                  'toll_booth','tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

NUM_IMAGES = {
    'train': 363572,
    'validation': 53041,
    'test': 53473
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
    print('num_epochs = %s' %str(FLAGS.num_epochs))
    print('num_classes = %s' %str(FLAGS.num_classes))
    print('initial_lr = %s' %str(FLAGS.initial_lr))
    print('lr_decay_factor = %s' %str(FLAGS.lr_decay_factor))
    print('dropout_rate = %s' %str(FLAGS.dropout_rate))
    print('ls_dim = %s' %str(FLAGS.ls_dim))
    print('l2_reg = %s' %str(FLAGS.l2_reg))
    print('descr_tag = %s' %str(FLAGS.descr_tag))
    print('width_coefficient = %s' %str(FLAGS.width_coefficient))
    print('depth_coefficient = %s' %str(FLAGS.depth_coefficient))
    print('block_drop_rate = %s' %str(FLAGS.block_drop_rate))
    print('depth_divisor = %s' %str(FLAGS.depth_divisor))
    print('activation = %s' %str(FLAGS.activation))
    print('batch_norm = %s' %str(FLAGS.batch_norm))
    print('gpus_type = %s' %str(FLAGS.gpus_type))
    print('num_gpus = %s' %str(FLAGS.num_gpus))
    print('img_size_y = %s' %str(FLAGS.img_size_y))
    print('img_size_x = %s' %str(FLAGS.img_size_x))
    print('lr_scedule_2step = %s' %FLAGS.lr_scedule_2step)
    print('lr_scedule_1step = %s' %FLAGS.lr_scedule_1step)
    print('lr_boundaries = %s' %str(lr_boundaries))
    print('lr_values = %s' %str(lr_values))

    print('batches_to_profile_range = %s' %(', '.join([str(l) for l in FLAGS.batches_to_profile_range])))
    print('profile_evaluation = %s' %FLAGS.profile_evaluation)
    print('profile_step = %s' %FLAGS.profile_step)
    print('to_train = %s' %FLAGS.to_train)
    print('to_evaluate_val = %s' %FLAGS.to_evaluate_val)
    print('to_evaluate_test = %s' %FLAGS.to_evaluate_test)
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
    f.write('\nnum_epochs: ' + str(FLAGS.num_epochs))
    f.write('\nnum_classes: ' + str(FLAGS.num_classes))
    f.write('\ninitial_lr: ' + str(FLAGS.initial_lr))
    f.write('\nlr_decay_factor: ' + str(FLAGS.lr_decay_factor))
    f.write('\ndropout_rate: ' + str(FLAGS.dropout_rate))
    f.write('\nls_dim: ' + str(FLAGS.ls_dim))
    f.write('\nl2_reg: ' + str(FLAGS.l2_reg))
    f.write('\ndescr_tag: ' + str(FLAGS.descr_tag))
    f.write('\nwidth_coefficient: ' + str(FLAGS.width_coefficient))
    f.write('\ndepth_coefficient: ' + str(FLAGS.depth_coefficient))
    f.write('\nblock_drop_rate: ' + str(FLAGS.block_drop_rate))
    f.write('\ndepth_divisor: ' + str(FLAGS.depth_divisor))
    f.write('\nactivation: ' + str(FLAGS.activation))
    f.write('\nbatch_norm: ' + str(FLAGS.batch_norm))
    f.write('\ngpus_type: ' + str(FLAGS.gpus_type))
    f.write('\nnum_gpus: ' + str(FLAGS.num_gpus))
    f.write('\nimg_size_y: ' + str(FLAGS.img_size_y))
    f.write('\nimg_size_x: ' + str(FLAGS.img_size_x))
    f.write('\nlr_scedule_2step: ' + str(FLAGS.lr_scedule_2step))
    f.write('\nlr_scedule_1step: ' + str(FLAGS.lr_scedule_1step))
    f.write('\nlr_boundaries: ' + str(lr_boundaries))
    f.write('\nlr_values: ' + str(lr_values))

    f.write('\nbatches_to_profile_range: ' + ', '.join([str(l) for l in FLAGS.batches_to_profile_range]))
    f.write('\nprofile_evaluation: ' + str(FLAGS.profile_evaluation))
    f.write('\nprofile_step: ' + str(FLAGS.profile_step))
    f.write('\nto_train: ' + str(FLAGS.to_train))
    f.write('\nto_evaluate_val: ' + str(FLAGS.to_evaluate_val))
    f.write('\nto_evaluate_test: ' + str(FLAGS.to_evaluate_test))
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
        """Initialize class used for training and evaluation of baseline networks.
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

        # Set up learning rate schedule and optimizer
        lr = self.compute_lr_schedule(self.total_train_steps)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.global_step = self.optimizer.iterations

        # Initialize baseline network
        self.model = tf_models.Baseline_CNN(FLAGS.descr_tag, FLAGS.num_classes, FLAGS.ls_dim,
                                            FLAGS.dropout_rate, FLAGS.width_coefficient, FLAGS.depth_coefficient,
                                            FLAGS.block_drop_rate, FLAGS.depth_divisor, FLAGS.activation, FLAGS.batch_norm)
        # Build the baseline network by processing a small random batch eagerly
        self.model(tf.random.uniform(shape=[2, FLAGS.img_size_y, FLAGS.img_size_x, tf_input.NUM_CHANNELS]),
                   is_training=True, step=None, keep_step_summary=False)
        
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
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, FLAGS.ckpt_dir_latest, max_to_keep=5, checkpoint_name='ckpt')
        self.checkpoint_manager_sparse = tf.train.CheckpointManager(checkpoint, FLAGS.ckpt_dir_sparse, max_to_keep=FLAGS.ckpt_sparse_num, checkpoint_name='ckpt')
        FLAGS.ckpt_frequency_sparse = self.total_train_steps // FLAGS.ckpt_sparse_num

        # Flag for keeping summary of weights and/or gradients during training steps
        self.keep_step_summary = False

        # Keep a record of the hyperparameters used for training
        write_hyperparameters(self.total_train_steps, self.optimizer.lr.boundaries, self.optimizer.lr.values)

        # Print a summary of the model
        if ('EfficientNet' in FLAGS.descr_tag):
            print('blocks_num = %d' %self.model.feature_extractor.blocks_num)
            print('layers_num = %d' %(self.model.feature_extractor.blocks_num * 3 + 2))
        self.model.summary()

    def build_distributed_dataset(self, dataset_type, is_training, num_epochs):
        """Build input stream.
        Args:
            dataset_type: string; type of dataset.
            is_training: boolean; whether the input will be used for training.
            num_epochs: int; number of times to repeat the dataset.
        Returns:
            dist_input_data: tf dataset; distributed dataset.
        """

        drop_remainder = is_training
        if (FLAGS.num_gpus > 1):
            drop_remainder = True
        input_data = tf_input.input_fn(dataset_type=dataset_type, is_training=is_training, data_dir=FLAGS.data_dir,
                                       batch_size=FLAGS.batch_size, num_epochs=num_epochs, drop_remainder=drop_remainder,
                                       img_size_y=FLAGS.img_size_y, img_size_x=FLAGS.img_size_x)
        dist_input_data = self.strategy.experimental_distribute_dataset(input_data)

        return dist_input_data

    def compute_lr_schedule(self, train_steps):
        """Set learning rate schedule.
        Args:
            train_steps: int; total number of training steps.
        Returns:
            lr: tf LearningRateSchedule; learning rate schedule.
        """

        if (FLAGS.lr_scedule_2step):
            # Learning rate drops twice
            b1 = 0.7 * train_steps
            b2 = b1 + 0.15 * train_steps
            boundaries = [b1, b2]
            v1 = FLAGS.initial_lr
            v2 = v1 * FLAGS.lr_decay_factor
            v3 = v2 * FLAGS.lr_decay_factor
            values = [v1, v2, v3]
        elif (FLAGS.lr_scedule_1step):
            # Learning rate drops once
            b1 = 0.8 * train_steps
            boundaries = [b1]
            v1 = FLAGS.initial_lr
            v2 = v1 * FLAGS.lr_decay_factor
            values = [v1, v2]
        else:
            # Learning rate drops once
            b1 = 0.8 * train_steps
            boundaries = [b1]
            v1 = FLAGS.initial_lr
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
                    # Load variables without the global step
                    vars_to_restore = self.model.variables
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

        # Measure the top-1 and top-5 accuracy on the validation set
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy', dtype=tf.float32)
        self.eval_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='eval_accuracy_top5', dtype=tf.float32)

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

    def compute_total_loss(self, logits, labels):
        """Compute training loss.
        Args:
            logits: 2-D float Tensor; it contains classification logits
                for an image batch. It is of size [batch_size, num_cls],
                where batch_size is the number of images in the batch,
                and num_cls is the number of classes.
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
        Returns:
            total_loss: float; training loss.
        """

        # Calculate cross entropy loss
        cross_entropy_per_image = self.compute_cross_entropy_loss(labels, logits)
        total_loss = tf.reduce_mean(cross_entropy_per_image)

        # Calculate l2 regularization loss
        if (FLAGS.l2_reg > 0.):
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'batch_norm' not in v.name])
            total_loss += FLAGS.l2_reg * 2 * l2_loss
            
        return total_loss

    def update_train_metrics(self, logits, labels, total_loss):
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
        Returns:
            -
        """

        self.train_loss.update_state(total_loss * FLAGS.num_gpus)
        self.train_accuracy.update_state(labels, logits)
        self.train_accuracy_top5.update_state(labels, logits)

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
            summary_grads: list; it contains Tensors with the gradients
                of the trainable variables.
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
                summary_grads: list; it contains Tensors with the gradients
                    of the trainable variables.
            """

            images, labels = step_input_batch
            labels = tf.squeeze(labels)

            with tf.GradientTape() as tape:
                # Process the provided image batch
                with self.summary_writer.as_default():
                    logits = self.model(images, is_training=True, step=step,
                                        keep_step_summary=(FLAGS.keep_activations_summary and keep_step_summary))
                
                # Compute total loss
                total_loss = self.compute_total_loss(logits, labels)
                total_loss = total_loss * (1.0 / FLAGS.num_gpus)
                
            # Calculate gradients
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Update train metrics
            self.update_train_metrics(logits, labels, total_loss)
            
            summary_grads = None
            if (FLAGS.keep_grads_summary and keep_step_summary):
                summary_grads = gradients
            
            return summary_grads
        
        summary_grads = self.strategy.run(step_fn, args=(input_batch, step, keep_step_summary))

        return summary_grads

    def update_eval_metrics(self, logits, labels):
        """Update evaluation metrics.
        Args:
            logits: 2-D float Tensor; it contains classification logits
                for an image batch. It is of size [batch_size, num_cls],
                where batch_size is the number of images in the batch,
                and num_cls is the number of classes.
            labels: 1-D float Tensor; it contains the labels of an image
                batch. It is of size [batch_size], where batch_size
                is the number of images in the batch.
        Returns:
            -
        """

        self.eval_accuracy.update_state(labels, logits)
        self.eval_accuracy_top5.update_state(labels, logits)

    @tf.function
    def eval_step(self, input_batch, is_training=False):
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
        Returns:
            -
        """

        def step_fn(step_input_batch, is_training):
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
            Returns:
                -
            """

            images, labels = step_input_batch
            labels = tf.squeeze(labels)

            # Process the provided image batch
            logits = self.model(images, is_training=is_training, step=None, keep_step_summary=False)
            
            # Update evaluation metrics
            self.update_eval_metrics(logits, labels)
        
        self.strategy.run(step_fn, args=(input_batch, is_training))

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
    
    def reset_eval_metrics(self):
        """Reset evaluation metrics.
        Args:
            -
        Returns:
            -
        """

        self.eval_accuracy.reset_states()
        self.eval_accuracy_top5.reset_states()

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
        format_str = (f'step %d: total_loss = %.2f; acc1 = %.2f, acc5 = %.2f; (%.3f sec/step, %.3f sec/img)')
        print (format_str %(step, self.train_loss.result(), self.train_accuracy.result(),
                            self.train_accuracy_top5.result(), secs_per_step, secs_per_img))
    
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

        for grad, var in zip(gradients, self.model.trainable_variables):
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

            tf.summary.scalar("dev_step_accuracy_top1", self.eval_accuracy.result(), step=step)
            tf.summary.scalar("dev_step_accuracy_top5", self.eval_accuracy_top5.result(), step=step)

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

    def evaluate(self, dataset_type, is_training=False, num_epochs=1):
        """Evaluate baseline network.
        Args:
            dataset_type: string; type of dataset.
            is_training: boolean; whether the input will go through the data
                augmentation used for training.
            num_epochs: int; number of epochs during evaluation.
        Returns:
            -
        """

        # Set up input stream
        eval_dataset = self.build_distributed_dataset(dataset_type=dataset_type, is_training=is_training, num_epochs=num_epochs)

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
            self.eval_step(step_batch, False)

            # Keep timing metrics periodically
            if (FLAGS.profile_step and (step >= FLAGS.batches_to_time_range[0])):
                total_p_step += 1
                if (total_p_step % FLAGS.profile_step == 0):
                    end_time = time.time()
                    throughput = (FLAGS.profile_step * (FLAGS.batch_size / FLAGS.num_gpus)) / (end_time - init_time)
                    throughput_ms = throughput / 1000.0
                    time_lst.append(1/throughput_ms)
                    init_time = time.time()

            step += 1
        
        # Print and save evaluation metrics
        print('\n')
        print('accuracy @ 1 = %.3f%%' %(self.eval_accuracy.result()*100))
        print('accuracy @ 5 = %.3f%%' %(self.eval_accuracy_top5.result()*100))
        if (not (FLAGS.dont_save_eval_txt)):
            fname = (str(FLAGS.gpus_type) + 'x' + str(FLAGS.num_gpus) + '_' + str(FLAGS.save_tag) + '_bs' + str(FLAGS.batch_size) +
                     '_dt_' + dataset_type + '_isT' + str(is_training) + '.txt')
            f = open(os.path.join(FLAGS.ckpt_dir, fname), 'w')
            f.write('Accuracy @ 1: ' + str(self.eval_accuracy.result().numpy()*100) + '%')
            f.write('\nAccuracy @ 5: ' + str(self.eval_accuracy_top5.result().numpy()*100) + '%\n')
            f.close()
        
        # Print and save timing metrics
        if (FLAGS.profile_step):
            print('\nMean time: %.5f msec/im' %np.mean(time_lst))
            print('Std time: %.5f msec/im' %np.std(time_lst))
            f = open(os.path.join(FLAGS.ckpt_dir, 'profile_step.txt'), 'w')
            f.write('Mean time: ' + str(np.mean(time_lst)) + ' msec/im')
            f.write('\nStd time: ' + str(np.std(time_lst)) + ' msec/im')
            f.close()

def main(argv=None):
    """Main function for training of baseline networks.
    Args:
        -
    Returns:
        -
    """

    # Set up directories for checkpoints and summaries
    set_up_dirs()
    
    # Define strategy for distributing training to GPUs
    strategy = tf.distribute.MirroredStrategy(devices=["device:GPU:%d" %i for i in range(FLAGS.num_gpus)])

    if ('cropped' in FLAGS.data_dir):
        NUM_IMAGES['test'] = 55466

    # Move under strategy scope all code that involves distributed processing,
    # e.g. building the model (variables should be mirrored)
    with strategy.scope():
        trainer = distributedTrainer()
        if (FLAGS.to_train):
            trainer.train()

        # Evaluation after training
        if (FLAGS.to_evaluate_val):
            print("\n-------------Computing validation set evaluation metrics-------------\n")
            trainer.evaluate(dataset_type='validation', is_training=False, num_epochs=FLAGS.eval_epochs_num)
        if (FLAGS.to_evaluate_test):
            print("\n-------------Computing test set evaluation metrics-------------\n")
            trainer.evaluate(dataset_type='test', is_training=False, num_epochs=FLAGS.eval_epochs_num)
        if (FLAGS.to_evaluate_train):
            print("\n-------------Computing training set evaluation metrics-------------\n")
            trainer.evaluate(dataset_type='train', is_training=False, num_epochs=FLAGS.eval_epochs_num)

        print('End.')

if __name__ == '__main__':
    main()
