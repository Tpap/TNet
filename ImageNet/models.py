"""Set up models and utility functions.
"""

from __future__ import absolute_import, division, print_function

import os
import re
import sys

import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import layers



def adjust_var_name(v_name, latest_scope=None):
    """Improve readability of tf summaries. Remove the part of
       a Tensor's name that will be added anyway because of the
       sequence of inherited scopes; avoid scope duplication. In
       particular, everything before (and including) the current
       scope is removed, because is going to be added by the
       tf.summary as the sequence of active name scopes. Account
       also for the '_{i}' suffix that is added to names when
       modules are called more than once.
    Args:
        v_name: string; Tensor name.
        latest_scope: string; scope used as part of a Tensor name.
    Returns:
        v_name: string; modified Tensor name.
    """

    v_name = v_name.replace(':0', '')
    if ((latest_scope) and (latest_scope in v_name)):
        before_latest_scope, after_latest_scope = v_name.split(latest_scope, 1)
        existing_scope = before_latest_scope + latest_scope + after_latest_scope.split('/', 1)[0] + '/'
        v_name = v_name.replace(existing_scope, '')

    return v_name

def grid_positions(cur_batch_size, num_patches_y, num_patches_x):
    """Calculate spatial coordinates for the cells of the attention
       grid considered by the location module.
    Args:
        cur_batch_size: int; the number of images processed
            in the current level.
        num_patches_y: int; number of patches in the vertical
            dimension of the grid considered by the location module.
        num_patches_x: int; number of patches in the horizontal
            dimension of the grid considered by the location module.
    Returns:
        h: 2_D float tensor; it contains vertical coordinates for
            the patches in the attention grid considered by the
            location module. Coordinates start with 0 for the patch at
            the top left corner, and increase linearly with step 1.
            The Tensor is of size [cur_batch_size, patches_num], where
            cur_batch_size is the number of images processed in the
            current level, and patches_num is the total number of
            patches in the attention grid.
        w: 2_D float tensor; it contains horizontal coordinates for
            the patches in the attention grid considered by the
            location module. Coordinates start with 0 for the patch at
            the top left corner, and increase linearly with step 1.
            The Tensor is of size [cur_batch_size, patches_num], where
            cur_batch_size is the number of images processed in the
            current level, and patches_num is the total number of
            patches in the attention grid.
    """

    patches_num = num_patches_y * num_patches_x
    h = tf.matmul(tf.expand_dims(tf.range(num_patches_y, dtype=tf.float32), 1), tf.ones(shape=[1, num_patches_x]))
    w = tf.matmul(tf.ones(shape=[num_patches_y, 1]), tf.expand_dims(tf.range(num_patches_x, dtype=tf.float32), 0))
    h = tf.reshape(h, [1, patches_num])
    h = tf.tile(h, [cur_batch_size, 1])
    w = tf.reshape(w, [1, patches_num])
    w = tf.tile(w, [cur_batch_size, 1])
    
    return h, w

def create_positional_vectors_components(pos_y, pos_x, pos_rl, mask, cur_batch_size,
                                         num_patches_y, num_patches_x, res_level):
    """Create positional triplets (y, x, s) for attended
       locations. Given a processing level l, we assume
       that a grid is superposed onto the original input
       image, where its cells correspond to all possible
       candidate locations of the level. The spatial
       coordinates of the grid cells start with (0, 0) in
       the top left corner, and increase linearly with
       step 1 both horizontally and vertically. For the
       positional triplet (y, x, s) of each attended
       location, holds that y and x are the spatial
       coordinates of the corresponding grid cell,
       while s represents scale and is set equal to l-1.
    Args:
        pos_y: 2-D float Tensor; it contains the vertical
            spatial coordinates of the images processed in
            the current level. It is of size
            [cur_batch_size, 1], where cur_batch_size is
            the number of images processed in the current
            level.
        pos_x: 2-D float Tensor; it contains the horizontal
            spatial coordinates of the images processed in
            the current level. It is of size
            [cur_batch_size, 1], where cur_batch_size is
            the number of images processed in the current
            level.
        pos_rl: 2-D float Tensor; it contains the scale
            values that identify the current processing
            level. It is of size [cur_batch_size, 1],
            where cur_batch_size is the number of images
            processed in the current level.
        mask: 1-D boolean Tensor; it indicates which locations are
            selected to be processed in the next level. Candidate
            locations within each considered attention grid are
            flattened in a left to right and top to bottom fashion,
            and the value True is assigned to selected locations,
            while False is assigned to the rest. It is of size
            [cur_batch_size*locs_num], where cur_batch_size is the
            number of images processed in the current level, and
            locs_num is the number of candidate locations within
            each attention grid.
        cur_batch_size: int; the number of images processed
            in the current level.
        num_patches_y: int; number of patches in the vertical
            dimension of the grid considered by the location module.
        num_patches_x: int; number of patches in the horizontal
            dimension of the grid considered by the location module.
        res_level: int; value that identifies the next processing level.
    Returns:
        pos_y: 2-D float Tensor; it contains the vertical
            spatial coordinates of the locations that will
            be processed in the next level. It is of size
            [cur_batch_size*loc_per_grid, 1], where
            cur_batch_size is the number of images processed
            in the current level, and loc_per_grid is the number
            of locations per attention grid that are selected
            to be attended in the next processing level.
        pos_x: 2-D float Tensor; it contains the horizontal
            spatial coordinates of the locations that will
            be processed in the next level. It is of size
            [cur_batch_size*loc_per_grid, 1], where
            cur_batch_size is the number of images processed
            in the current level, and loc_per_grid is the number
            of locations per attention grid that are selected
            to be attended in the next processing level.
        pos_rl: 2-D float Tensor; it contains the scale
            values that identify the next processing
            level. It is of size [cur_batch_size*loc_per_grid, 1],
            where cur_batch_size is the number of images processed
            in the current level, and loc_per_grid is the number
            of locations per attention grid that are selected
            to be attended in the next processing level.
    """
    
    # Calculate spatial coordinates for all candidate locations considered
    # by the location module in the current processing level (res_level-1)
    grid_pos_y, grid_pos_x = grid_positions(cur_batch_size, num_patches_y, num_patches_x)
    pos_y = pos_y * num_patches_y + grid_pos_y
    pos_x = pos_x * num_patches_x + grid_pos_x
    # Create positional triplets for the locations selected to be
    # processed in the next level (res_level)
    inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
    pos_y = tf.gather_nd(params=tf.reshape(pos_y, [-1, 1]), indices=inds)
    pos_x = tf.gather_nd(params=tf.reshape(pos_x, [-1, 1]), indices=inds)
    pos_rl = tf.ones_like(pos_y, dtype=tf.float32) * res_level

    return pos_y, pos_x, pos_rl

def create_positional_vectors(pos_y, pos_x, pos_rl, channels, min_inv_afreq=1.0, max_inv_afreq=1.0e4):
    """Use sine and cosine functions of different frequencies
       to encode positional triplets (y, x, s), according to
       https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        pos_y: 2-D float Tensor; it contains the vertical
            spatial coordinates of the images that will
            be processed in the next level. It is of size
            [cur_batch_size*loc_per_grid, 1], where
            cur_batch_size is the number of images processed
            in the current level, and loc_per_grid is the number
            of locations per attention grid that are selected
            to be attended in the next processing level.
        pos_x: 2-D float Tensor; it contains the horizontal
            spatial coordinates of the images that will
            be processed in the next level. It is of size
            [cur_batch_size*loc_per_grid, 1], where
            cur_batch_size is the number of images processed
            in the current level, and loc_per_grid is the number
            of locations per attention grid that are selected
            to be attended in the next processing level.
        pos_rl: 2-D float Tensor; it contains the scale
            values that identify the next processing
            level. It is of size [cur_batch_size*loc_per_grid, 1],
            where cur_batch_size is the number of images processed
            in the current level, and loc_per_grid is the number
            of locations per attention grid that are selected
            to be attended in the next processing level.
        channels: int; dimensionality of the positional
            encodings.
        min_inv_afreq: float; minimum value of the inverse angular
            frequency that is used in the sine and cosine functions.
        max_inv_afreq: float; maximum value of the inverse angular
            frequency that is used in the sine and cosine functions.
    Returns:
        f_pos: 2-D float Tensor; it contains positional encodings
            for the images that will be processed in the next
            level. It is of size [cur_batch_size*loc_per_grid, channels],
            where cur_batch_size is the number of images processed
            in the current level, loc_per_grid is the number of
            locations per attention grid that are selected to be
            attended in the next processing level, and channels
            is the dimensionality of the positional encodings.
    """
    
    # Calculate the angular frequencies of the sine and cosine functions
    # that will be used for positional encoding, based on the specified
    # dimensionality of the encodings
    num_dims = 3
    num_afreq = channels // (num_dims * 2)
    log_afreq_increment = (math.log(float(min_inv_afreq) / float(max_inv_afreq))) / (float(num_afreq) - 1)
    afrequencies = (1.0 / min_inv_afreq) * tf.exp(tf.cast(tf.range(num_afreq), dtype=tf.float32) * log_afreq_increment)
    
    # Calculate sine and cosine functions of different frequencies at positions
    # provided by pos_y, pos_x, and pos_rl
    pos_lst = [pos_y, pos_x, pos_rl]
    f_pos = tf.zeros(shape=[tf.shape(pos_y)[0], channels], dtype=tf.float32)
    for dim in range(num_dims):
        positions = pos_lst[dim]
        scaled_positions = positions * tf.expand_dims(afrequencies, 0)
        signal = tf.concat([tf.sin(scaled_positions), tf.cos(scaled_positions)], axis=1)
        prepad = dim * 2 * num_afreq
        postpad = channels - (dim + 1) * 2 * num_afreq
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        f_pos += signal
    
    return f_pos

class meshgrid_positional_encoding(layers.Layer):
    def __init__(self):
        super(meshgrid_positional_encoding, self).__init__(name='meshgrid_positional_encoding')
        """Initialize class that creates 2-D positional encodings
           according to https://arxiv.org/pdf/2004.13621.pdf.
           Given a feature map, it calculates horizontal and
           vertical coordinates in the normalized range
           [-1, 1], leading to a 2-D positional encoding for
           each spatial location of the feature map, which
           is then scaled to a learned value range through
           linear layers.
        Args:
            -
        Returns:
            -
        """

        # Initialize linear layers that scale positional
        # encodings to a learned value range
        self.height_scaling = layers.Dense(1, use_bias=True, kernel_initializer='ones',
                                           bias_initializer='zeros', name='height_scaling')
        self.width_scaling = layers.Dense(1, use_bias=True, kernel_initializer='ones',
                                          bias_initializer='zeros', name='width_scaling')
    
    def add_positional_encoding(self, feature_map):
        """Add 2-D positional encodings to a given feature map.
        Args:
            feature_map: 4-D float Tensor; feature map to enrich with
                positional encodings. It is of size [cur_batch_size, H, W, C],
                where cur_batch_size is the number of feature maps in the
                batch, H is their spatial height, W is their spatial width,
                and C is the number of channels.
        Returns:
            feature_map: 4-D float Tensor; feature map enriched with
                positional encodings. It is of size [cur_batch_size, H, W, C+2],
                where cur_batch_size is the number of feature maps in the
                batch, H is their spatial height, W is their spatial width,
                and C is the number of channels of the input feature map,
                which is increased by concatenating 2-D positional encodings.
        """

        feature_map_shape = tf.shape(feature_map)
        batch_size = feature_map_shape[0]
        height = feature_map_shape[1]
        width = feature_map_shape[2]
        # Construct meshgrid
        h = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=[1, width]))
        w = tf.matmul(tf.ones(shape=[height, 1]), tf.expand_dims(tf.linspace(-1.0, 1.0, width), 0))
        # Scale meshgrid values
        h = tf.reshape(h, [-1, 1])
        h = self.height_scaling(h)
        w = tf.reshape(w, [-1, 1])
        w = self.width_scaling(w)
        # Concatenate positional encoding to the input feature map
        h = tf.reshape(h, [1, height, width, 1])
        h = tf.tile(h, [batch_size, 1, 1, 1])
        w = tf.reshape(w, [1, height, width, 1])
        w = tf.tile(w, [batch_size, 1, 1, 1])
        feature_map = tf.concat([feature_map, h, w], axis=-1)

        return feature_map

class convolutional_block(layers.Layer):
    def __init__(self, f, f_channels, s, in_c, part, block, padding='same'):
        super(convolutional_block, self).__init__(name='part' + str(part) + '_block' + str(block))
        """Initialize a convolutional residual block with
           three convolutional layers, according to
           https://arxiv.org/pdf/1512.03385.pdf.
        Args:
            f: int; kernel size of the intermediate
                convolutional layer.
            f_channels: list of ints; it contains the number
                of output channels in the convolutional layers.
            s: int; stride of the intermediate
                convolutional layer.
            in_c: int; number of input channnels.
            part: int; indicates the part of a network
                that the convolutional block belongs to.
            block: int; indicates the position of the 
                convolutional block within a part of a
                network.
            padding: string; the kind of padding in use,
                either 'same' or 'valid'.
        Returns:
            -
        """

        self.padding = padding
        self.f = f
        self.f_channels = f_channels
        self.s = s
        self.in_c = in_c
        
        # Initialize the convolutional layers
        self.conv1 = layers.Conv2D(f_channels[0], (1, 1), strides=(1, 1), padding='same',
                                   kernel_initializer='glorot_normal', bias_initializer='zeros', name='comp_1/conv2d')
        self.conv2 = layers.Conv2D(f_channels[1], (self.f, self.f), strides=(s, s), padding=self.padding,
                                   kernel_initializer='glorot_normal', bias_initializer='zeros', name='comp_2/conv2d')
        self.conv3 = layers.Conv2D(f_channels[2], (1, 1), strides=(1, 1), padding='same',
                                   kernel_initializer='glorot_normal', bias_initializer='zeros', name='comp_3/conv2d')
        # Initialize the residual convolutional layer, only if
        # stride is bigger than 1, or if the input and output
        # channels differ
        if ((self.s > 1) or (self.in_c != self.f_channels[2])):
            self.shortcut = layers.Conv2D(f_channels[2], (1, 1), strides=(s, s), padding='same',
                                          kernel_initializer='glorot_normal', bias_initializer='zeros', name='shortcut_comp/conv2d')

    def call(self, x, step=None, keep_step_summary=False):
        """Apply the convolutional residual block.
        Args:
            x: 4-D float Tensor; feature map to be processed.
                It is of size [batch_size, H, W, C], where batch_size
                is the number of images in the batch, H and W are the
                vertical and horizontal spatial dimensions of the
                feature map respectively, and C is the number of its
                channels.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            features: 4-D float Tensor; output feature map. It is of
                size [batch_size, H, W, C], where batch_size is the
                number of images in the batch, H and W are the vertical
                and horizontal spatial dimensions of the feature map
                respectively, and C is the number of its channels.
        """

        shortcut_features = x
        # component 1
        features = self.conv1(x)
        features = tf.nn.leaky_relu(features, name='comp_1/leaky_relu')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        # component 2
        features = self.conv2(features)
        features = tf.nn.leaky_relu(features, name='comp_2/leaky_relu')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        # component 3
        features = self.conv3(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # component 4 - residual connection
        if ((self.padding == 'valid') and (self.f > 1)):
            lt_margin = tf.cast(tf.math.floor((self.f-1.0) / 2.0), tf.int32)
            rb_margin = tf.cast(tf.math.ceil((self.f-1.0) / 2.0), tf.int32)
            shortcut_features = shortcut_features[:, lt_margin:-rb_margin, lt_margin:-rb_margin, :]
        if ((self.s > 1) or (self.in_c != self.f_channels[2])):
            shortcut_features = self.shortcut(shortcut_features)
            if (keep_step_summary):
                tf.summary.histogram(name=adjust_var_name(shortcut_features.name, self.name) + '/activations', data=shortcut_features, step=step)
        features = tf.add(features, shortcut_features, name='comp_4/add')
        features = tf.nn.leaky_relu(features, name='comp_4/leaky_relu')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        
        return features

class BagNet_77(layers.Layer):
    def __init__(self, descr_tag, multi_factor):
        super(BagNet_77, self).__init__(name='feature_extraction')
        """Initialize BagNet-77.
        Args:
            descr_tag: string; description of the model to use
                for feature extraction.
            multi_factor: int; multiplication factor of channels
                dimension in convolutional blocks.
        Returns:
            -
        """

        self.descr_tag = descr_tag
        self.multi_factor = multi_factor
        
        c = [64, 128, 256, 512]
        padding_lst = ['valid',
                        'same', 'same',
                        'same', 'same',
                        'valid', 'same',
                        'valid', 'same',
                        'same'] 

        # Initialize part 1
        self.part_1_conv = layers.Conv2D(c[0], (3, 3), strides=(1, 1), padding=padding_lst[0],
                                         kernel_initializer='glorot_normal', bias_initializer='zeros',
                                         name='part1/conv2d')
        
        # Initialize part2
        self.part2_cb = convolutional_block(f=3, f_channels=[c[0], c[0], self.multi_factor*c[0]], s=2, in_c=c[0], part=2, block=1, padding=padding_lst[1])
        self.part2_ib1 = convolutional_block(f=3, f_channels=[c[0], c[0], self.multi_factor*c[0]], s=1, in_c=self.multi_factor*c[0], part=2, block=2, padding=padding_lst[2])
        self.part2_ib2 = convolutional_block(f=1, f_channels=[c[0], c[0], self.multi_factor*c[0]], s=1, in_c=self.multi_factor*c[0], part=2, block=3)
        
        # Initialize part3
        self.part3_cb = convolutional_block(f=3, f_channels=[c[1], c[1], self.multi_factor*c[1]], s=2, in_c=self.multi_factor*c[0], part=3, block=1, padding=padding_lst[3])
        self.part3_ib1 = convolutional_block(f=3, f_channels=[c[1], c[1], self.multi_factor*c[1]], s=1, in_c=self.multi_factor*c[1], part=3, block=2, padding=padding_lst[4])
        self.part3_ib2 = convolutional_block(f=1, f_channels=[c[1], c[1], self.multi_factor*c[1]], s=1, in_c=self.multi_factor*c[1], part=3, block=3)
        self.part3_ib3 = convolutional_block(f=1, f_channels=[c[1], c[1], self.multi_factor*c[1]], s=1, in_c=self.multi_factor*c[1], part=3, block=4)

        # Initialize part4
        self.part4_cb = convolutional_block(f=3, f_channels=[c[2], c[2], self.multi_factor*c[2]], s=2, in_c=self.multi_factor*c[1], part=4, block=1, padding=padding_lst[5])
        self.part4_ib1 = convolutional_block(f=3, f_channels=[c[2], c[2], self.multi_factor*c[2]], s=1, in_c=self.multi_factor*c[2], part=4, block=2, padding=padding_lst[6])
        self.part4_ib2 = convolutional_block(f=1, f_channels=[c[2], c[2], self.multi_factor*c[2]], s=1, in_c=self.multi_factor*c[2], part=4, block=3)
        self.part4_ib3 = convolutional_block(f=1, f_channels=[c[2], c[2], self.multi_factor*c[2]], s=1, in_c=self.multi_factor*c[2], part=4, block=4)
        self.part4_ib4 = convolutional_block(f=1, f_channels=[c[2], c[2], self.multi_factor*c[2]], s=1, in_c=self.multi_factor*c[2], part=4, block=5)
        self.part4_ib5 = convolutional_block(f=1, f_channels=[c[2], c[2], self.multi_factor*c[2]], s=1, in_c=self.multi_factor*c[2], part=4, block=6)

        # Initialize part5
        self.part5_cb = convolutional_block(f=3, f_channels=[c[3], c[3], self.multi_factor*c[3]], s=1, in_c=self.multi_factor*c[2], part=5, block=1, padding=padding_lst[7])
        self.part5_ib1 = convolutional_block(f=3, f_channels=[c[3], c[3], self.multi_factor*c[3]], s=1, in_c=self.multi_factor*c[3], part=5, block=2, padding=padding_lst[8])
        self.part5_ib2 = convolutional_block(f=1, f_channels=[c[3], c[3], self.multi_factor*c[3]], s=1, in_c=self.multi_factor*c[3], part=5, block=3)

        # Initialize part6
        self.part_6_conv = layers.Conv2D(c[3], (1, 1), strides=(1, 1), padding=padding_lst[9],
                                         kernel_initializer='glorot_normal', bias_initializer='zeros',
                                         name='part6/conv2d')

    def call(self, x, is_training, num_do_layers, keep_prob, step=None, keep_step_summary=False):
        """Execute BagNet-77 inference.
        Args:
            x: 4-D float Tensor; image batch to be processed.
                It is of size [batch_size, H, W, C], where batch_size
                is the number of images in the batch, H is their height,
                W is their width, and C is the number of channels.
            is_training: boolean; whether the model is in training phase.
            num_do_layers: int; number of spatial dropout layers.
            keep_prob: float; dropout keep probability.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            loc_features: 4-D float Tensor; feature map to be provided
                as input to the location module. It is of size
                [batch_size, num_patches_y, num_patches_x, c], where
                batch_size is the number of images in the batch,
                num_patches_y and num_patches_x are the number of
                patches in the vertical and horizontal dimensions
                of the grid considered by the location module
                respectively, and c is the number of channels.
            features: 4-D float Tensor; the extracted feature vector.
                It is of size [batch_size, 1, 1, C], where batch_size
                is the number of images in the batch, and C is the
                number of output channels.
        """

        loc_features = None
        # Execute part 1
        features = self.part_1_conv(x)
        features = tf.nn.leaky_relu(features, name='part1/leaky_relu')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Execute part 2
        features = self.part2_cb(features, step, keep_step_summary)
        features = self.part2_ib1(features, step, keep_step_summary)
        features = self.part2_ib2(features, step, keep_step_summary)

        # Execute part 3
        features = self.part3_cb(features, step, keep_step_summary)
        features = self.part3_ib1(features, step, keep_step_summary)
        features = self.part3_ib2(features, step, keep_step_summary)
        features = self.part3_ib3(features, step, keep_step_summary)
        if (is_training and (num_do_layers >= 4)):
            features = tf.nn.dropout(features, rate=(1.0-keep_prob), noise_shape=[tf.shape(features)[0], 1, 1, tf.shape(features)[3]])

        # Execute part 4
        features = self.part4_cb(features, step, keep_step_summary)
        features = self.part4_ib1(features, step, keep_step_summary)
        features = self.part4_ib2(features, step, keep_step_summary)
        features = self.part4_ib3(features, step, keep_step_summary)
        features = self.part4_ib4(features, step, keep_step_summary)
        features = self.part4_ib5(features, step, keep_step_summary)
        if ('TNet' in self.descr_tag):
            loc_features = tf.strided_slice(features, [0, 0, 0, 0], tf.shape(features), [1, 2, 2, 1])
        if (is_training and (num_do_layers >= 3)):
            features = tf.nn.dropout(features, rate=(1.0-keep_prob), noise_shape=[tf.shape(features)[0], 1, 1, tf.shape(features)[3]])

        # Execute part 5
        features = self.part5_cb(features, step, keep_step_summary)
        features = self.part5_ib1(features, step, keep_step_summary)
        features = self.part5_ib2(features, step, keep_step_summary)
        if (is_training and (num_do_layers >= 2)):
            features = tf.nn.dropout(features, rate=(1.0-keep_prob), noise_shape=[tf.shape(features)[0], 1, 1, tf.shape(features)[3]])

        # Execute part 6
        features = self.part_6_conv(features)
        features = tf.nn.leaky_relu(features, name='part6/leaky_relu')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        if (is_training and (num_do_layers >= 1)):
            features = tf.nn.dropout(features, rate=(1.0-keep_prob), noise_shape=[tf.shape(features)[0], 1, 1, tf.shape(features)[3]])

        # Execute part 7
        features = tf.reduce_mean(features, axis=[1, 2], keepdims=True, name='part7/GAP')

        return loc_features, features

class location_module_cls(layers.Layer):
    def __init__(self, num_channels):
        super(location_module_cls, self).__init__(name='location_prediction_earlyLocs')
        """Initialize the location module.
        Args:
            num_channels: int; number of output channels in the
                first convolutional layer.
        Returns:
            -
        """

        # Initialize linear layers for 2-D positional encodings
        self.meshgrid_pos_enc = meshgrid_positional_encoding()
        # Initialize convolutional layers
        self.conv = layers.Conv2D(num_channels, (1, 1), strides=(1, 1), padding='same',
                                  kernel_initializer='glorot_normal', use_bias=True, name='conv2d')
        self.output_linear = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same',
                                           kernel_initializer='glorot_normal', use_bias=True, name='output_linear')

    def call(self, loc_features, features, is_training, step=None, keep_step_summary=False):
        """Apply the location module.
        Args:
            loc_features: 4-D float Tensor; feature map consisted
                of feature vectors (one at each spatial position)
                that describe the image patches within the attention
                grid considered by the location module. It is of size
                [batch_size, num_patches_y, num_patches_x, c], where
                batch_size is the number of images in the batch,
                num_patches_y and num_patches_x are the number of
                patches in the vertical and horizontal dimensions
                of the grid considered by the location module
                respectively, and c is the number of channels.
            features: 4-D float Tensor; contextual feature vector.
                It is of size [batch_size, 1, 1, C], where batch_size
                is the number of images in the batch, and C is the
                number of output channels.
            is_training: boolean; whether the model is in training phase.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            location_probs: 2-D float Tensor; it contains attention
                probabilities. It is of size [cur_batch_size, locs_num],
                where cur_batch_size is the number of images processed
                in the current level, and locs_num is the number of
                locations in an attention grid.
        """

        # Stop gradients from backpropagating to the
        # feature extraction module
        if (is_training):
            loc_features = tf.stop_gradient(loc_features)
            features = tf.stop_gradient(features)
        
        # Concatenate loc_features, loc_features and positional encodings
        features = tf.tile(features, [1, tf.shape(loc_features)[1], tf.shape(loc_features)[2], 1])
        loc_features = tf.concat([loc_features, features], axis=-1)
        loc_features = self.meshgrid_pos_enc.add_positional_encoding(loc_features)

        # Fuse the concatenated features
        loc_features = self.conv(loc_features)
        loc_features = tf.nn.leaky_relu(loc_features, name='leaky_relu')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(loc_features.name, self.name) + '/activations', data=loc_features, step=step)

        # Project features to l2 normalized logits of a
        # categorical distribution for each image in the batch
        loc_features = self.output_linear(loc_features)
        loc_features = tf.reshape(loc_features, [tf.shape(loc_features)[0], -1])
        loc_features, _ = tf.linalg.normalize(loc_features, ord='euclidean', axis=-1)
        # Calculate the attention probabilities of candidate locations
        location_probs = tf.nn.softmax(loc_features, axis=-1, name='softmax_probs')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(location_probs.name, self.name) + '/activations', data=location_probs, step=step)
        
        return location_probs

class positional_encoding_module_cls(layers.Layer):
    def __init__(self, ls_dim):
        super(positional_encoding_module_cls, self).__init__(name='feature_posBurn')
        """Initialize the positional encoding module.
        Args:
            ls_dim: int; dimensionality of the feature latent space.
        Returns:
            -
        """
        
        # Initialize weights
        self.ls_dim = ls_dim
        self.fc = layers.Dense(self.ls_dim, use_bias=True, kernel_initializer='glorot_normal',
                               bias_initializer='zeros', name='linear')

    def call(self, features, f_pos, step=None, keep_step_summary=False):
        """Apply the positional encoding module.
        Args:
            features: 4-D float Tensor; features vectors to be
                enhanced with positional information. It is of size
                [batch_size, 1, 1, ls_dim], where batch_size is the
                number of feature vectors, and ls_dim is their
                dimensionality.
            f_pos: 2-D float Tensor; it contains positional encodings
                for the provided feature vectors. It is of size
                [batch_size, c], where batch_size is the number of
                feature vectors to enhance with positional information,
                and c is the dimensionality of the positional encodings.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            features: 4-D float Tensor; features enhanced with positional
                information. It is of size [batch_size, 1, 1, ls_dim],
                where batch_size is the number of feature vectors, and
                ls_dim is their dimensionality.
        """

        # Concatenate the feature vectors with their positional
        # encodings, and fuse them through a linear projection
        features = tf.reshape(features, [-1, self.ls_dim])
        f_pos = tf.reshape(f_pos, [-1, self.ls_dim])
        concat_features = tf.concat([features, f_pos], axis=-1)
        features = self.fc(concat_features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        
        features = tf.expand_dims(tf.expand_dims(features, 1), 1)

        return features

class classification_module_cls(layers.Layer):
    def __init__(self, num_cls, ls_dim):
        super(classification_module_cls, self).__init__(name='feature_extraction/logits_layer')
        """Initialize the classification module.
        Args:
            num_cls: int; number of classes.
            ls_dim: int; dimensionality of the feature latent space
                used for classification.
        Returns:
            -
        """
        
        # Initialize the linear classification layer
        self.ls_dim = ls_dim
        self.linear = layers.Dense(num_cls, use_bias=True, kernel_initializer='glorot_normal',
                                   bias_initializer='zeros', name='linear')

    def call(self, x, step=None, keep_step_summary=False):
        """Apply the classification module.
        Args:
            x: 2-D float Tensor; features that represented images to
                classify. It is of size [batch_size, ls_dim], where
                batch_size is the number of images to classify, and
                ls_dim is the dimensionality of the features.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            logits: 2-D float Tensor; it contains classification
                logits for every image. It is of size [batch_size, num_cls],
                where batch_size is the number of images to
                classify, and num_cls is the number of classes.
        """

        # Features are reshaped because they may be of size [batch_size, 1, 1, ls_dim]
        x = tf.reshape(x, [-1, self.ls_dim])
        logits = self.linear(x)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(logits.name, self.name) + '/activations', data=logits, step=step)
        
        return logits

class TNet(tf.keras.Model):
    def __init__(self, descr_tag, ls_dim, num_patches_y, num_patches_x, overlap, num_res_levels,
                 num_cls, base_res_y, base_res_x, multi_factor, num_do_layers, keep_prob, loc_per_grid):
        super(TNet, self).__init__(name='TNet')
        """Initialize TNet.
        Args:
            descr_tag: string; description of the model to use
                as the feature extraction module.
            ls_dim: int; dimensionality of the feature latent space
                used for the final classification.
            num_patches_y: int; number of patches in the vertical dimension
                of the grid considered by the location module.
            num_patches_x: int; number of patches in the horizontal dimension
                of the grid considered by the location module.
            overlap: float; the fraction of each spatial image dimension
                occupied by the correspending dimension of a grid patch
                (same for both spatial dimensions).
            num_res_levels: int; number of processing levels that TNet
                goes through.
            num_cls: int; number of classes.
            base_res_y: int; base resolution of the feature extractor
                in the vertical dimension.
            base_res_x: int; base resolution of the feature extractor
                in the horizontal dimension.
            multi_factor: int; multiplication factor of channels
                dimension in convolutional blocks.
            num_do_layers: int; number of spatial dropout layers.
            keep_prob: float; dropout keep probability.
            loc_per_grid: list of floats; number of locations to attend
                per attention grid. It contains num_res_levels-1 entries.
        Returns:
            -
        """
        
        self.descr_tag = descr_tag
        self.ls_dim = ls_dim
        self.num_patches_y = num_patches_y
        self.num_patches_x = num_patches_x
        self.overlap = overlap
        self.num_res_levels = num_res_levels
        self.num_cls = num_cls
        self.base_res_y = base_res_y
        self.base_res_x = base_res_x
        self.multi_factor = multi_factor
        self.num_do_layers = num_do_layers
        self.keep_prob = keep_prob
        self.loc_per_grid = loc_per_grid

        # Initialize TNet modules
        if ('BagNet_77' in self.descr_tag):
            self.feature_extraction_module = BagNet_77(self.descr_tag, self.multi_factor)
        self.positional_encoding_module = positional_encoding_module_cls(self.ls_dim)
        self.location_module = location_module_cls(self.ls_dim)
        self.classification_module = classification_module_cls(self.num_cls, self.ls_dim)

        # Baseline variable used during training
        self.baseline_var = tf.Variable(0.5, trainable=False, name='baseline_var',
                                        dtype=tf.float32, aggregation=tf.VariableAggregation.MEAN)

    def create_boolean_mask(self, location_probs, is_training, adv_eval_data,
                            loc_per_grid, loc_per_lvl, batch_size):
        """Select the locations that will be processed in
           the next level, and calculate quantities needed
           or training and evaluation.
        Args:
            location_probs: 2-D float Tensor; it contains attention
                probabilities. It is of size [cur_batch_size, locs_num],
                where cur_batch_size is the number of images processed
                in the current level, and locs_num is the number of
                locations in an attention grid.
            is_training: boolean; whether the model is in training phase.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
            loc_per_grid: int; number of locations to attend per attention
                grid (they will be processed in the next level).
            loc_per_lvl: int; number of locations attended per
                image in the current processing level.
            batch_size: float; number of provided images to classify.
        Returns:
            ret_dict: dictionary; includes the following entries:
                mask: 1-D boolean Tensor; it indicates which locations are
                    selected to be processed in the next level. Candidate
                    locations within each considered attention grid are
                    flattened in a left to right and top to bottom fashion,
                    and the value True is assigned to selected locations,
                    while False is assigned to the rest. It is of size
                    [cur_batch_size*locs_num], where cur_batch_size is the
                    number of images processed in the current level, and
                    locs_num is the number of candidate locations within
                    each attention grid.
                location_num_per_img: 2-D float Tensor; it contains the number
                    of locations per image that are selected to be processed in
                    the next level. It is of size [batch_size, 1], where
                    batch_size is the number of provided images to classify.
                location_log_probs (optional): 2-D float Tensor; it contains
                    the sum of log probabilities of locations selected to be
                    processed in the next level, and is used for training.
                    It is of size [batch_size, 1], where batch_size is the
                    number of provided images to classify.
                logprobs_per_feature (optional): 2-D float Tensor; it contains
                    the log probabilities of locations selected to be
                    processed in the next level, and is used for per-feature
                    regularization during training. It is of size
                    [loc_per_grid*cur_batch_size, 1], where loc_per_grid is
                    the number of locations per attention grid that are
                    selected to be attended in the next processing level,
                    and cur_batch_size is the number of images processed
                    in the current level.
                attended_locations (optional): 2-D float Tensor; it indicates
                    which locations are selected to be processed in the next
                    level for each image. Candidate locations within each
                    considered attention grid are flattened in a left to right
                    and top to bottom fashion, and the value 1 is assigned to
                    selected locations, while 0 is assigned to the rest. It is
                    used for advanced evaluation of the model. The Tensor is of
                    size [batch_size, loc_per_lvl*locs_num], where batch_size is
                    the number of provided images to classify, loc_per_lvl is
                    the number of locations attended per image in the current
                    processing level, and locs_num is the number of candidate
                    locations within each attention grid.
                location_probs (optional): 2-D float Tensor; it contains the
                    attention probabilities of all candidate locations
                    considered for processing in the next level for each image.
                    It is used for advanced evaluation of the model. The Tensor
                    is of size [batch_size, loc_per_lvl*locs_num], where
                    batch_size is the number of provided images to classify,
                    loc_per_lvl is the number of locations attended per image
                    in the current processing level, and locs_num is the number
                    of candidate locations within each attention grid.
        """

        # Select loc_per_grid locations to attend in the the next
        # processing level, by finding the top loc_per_grid
        # probabilities in location_probs for each image
        _, indices = tf.math.top_k(location_probs, k=loc_per_grid)
        # Create a mask of size [batch_size, locs_num], with 1 for
        # selected locations, and 0 for the rest
        cur_batch_size = tf.shape(location_probs)[0]
        locs_num = tf.shape(location_probs)[1]
        r = tf.expand_dims(tf.range(start=0, limit=cur_batch_size*locs_num, delta=locs_num, dtype=tf.int32), -1)
        indices += r
        indices = tf.reshape(indices, shape=[-1, 1])
        mask_shape = tf.shape(tf.reshape(location_probs, shape=[-1, 1]))
        mask = tf.scatter_nd(indices=indices, updates=tf.ones_like(indices, tf.float32), shape=mask_shape)
        mask = tf.reshape(mask, tf.shape(location_probs))
        mask = tf.stop_gradient(mask)
        
        ret_dict = {}
        if (is_training): # training
            # location_log_probs are used in the main learning rule, and
            # logprobs_per_feature are used in per-feature regularization
            location_log_probs = mask * tf.math.log(location_probs + 1e-10)
            logprobs_per_feature = tf.reshape(location_log_probs, [-1, 1])
            inds = tf.cast(tf.where(tf.reshape(mask, [-1])), tf.int32)
            logprobs_per_feature = tf.gather_nd(params=logprobs_per_feature, indices=inds)
            location_log_probs = tf.reduce_sum(location_log_probs, axis=1)
            location_log_probs = tf.reshape(location_log_probs, [batch_size, loc_per_lvl, 1])
            location_log_probs = tf.reduce_sum(location_log_probs, axis=1)

            ret_dict['location_log_probs'] = location_log_probs
            ret_dict['logprobs_per_feature'] = logprobs_per_feature
        elif (adv_eval_data): # advanced evaluation
            ret_dict['attended_locations'] = tf.reshape(mask, [batch_size, loc_per_lvl*locs_num])
            ret_dict['location_probs'] = tf.reshape(location_probs, [batch_size, loc_per_lvl*locs_num])
        
        location_num_per_img = tf.reduce_sum(mask, axis=1)
        location_num_per_img = tf.reshape(location_num_per_img, [batch_size, loc_per_lvl, 1])
        location_num_per_img = tf.reduce_sum(location_num_per_img, axis=1)
        mask = tf.reshape(tf.equal(mask, 1.0), [-1])
        ret_dict['location_num_per_img'] = location_num_per_img
        ret_dict['mask'] = mask

        return ret_dict

    def create_masked_batch(self, image_batch, mask):
        """Organize selected locations in a batch
           for the next processing level.
        Args:
            image_batch: 4-D float Tensor; image batch to be reorganized
                for the next processing level. It is of size
                [cur_batch_size, H, W, C], where cur_batch_size is the
                number of images processed in the current level, H is
                their height, W is their width, and C is the number of
                channels.
            mask: 1-D boolean Tensor; it indicates which locations are
                selected to be processed in the next level. Candidate
                locations within each considered attention grid are
                flattened in a left to right and top to bottom fashion,
                and the value True is assigned to selected locations,
                while False is assigned to the rest. It is of size
                [cur_batch_size*locs_num], where cur_batch_size is the
                number of images processed in the current level, and
                locs_num is the number of candidate locations within
                each attention grid.
        Returns:
            masked_image_batch: 4-D float Tensor; image batch that contains
                the locations selected to be processed in the next level.
                It is of size [cur_batch_size*loc_per_grid, h, w, C],
                where cur_batch_size is the number of images processed in
                the current level, loc_per_grid is the number of locations
                per attention grid that are selected to be attended in the
                next processing level, h is their height, w is their width,
                and C is the number of channels.
        """

        img_size_y = float(image_batch.shape[1])
        img_size_x = float(image_batch.shape[2])

        # Find stride and patch size per dimension, in order to
        # crop the selected locations
        num_patches_y = float(self.num_patches_y)
        patch_size_y = np.round(img_size_y*self.overlap)
        stride_y = (img_size_y - patch_size_y) / (num_patches_y - 1)
        stride_y = np.ceil(stride_y)
        img_size_y_new = (num_patches_y - 1) * stride_y + patch_size_y
        total_pad_y = img_size_y_new - img_size_y
        prepad_y = int(np.ceil(total_pad_y / 2))
        postpad_y = int(total_pad_y - prepad_y)

        num_patches_x = float(self.num_patches_x)
        patch_size_x = np.round(img_size_x*self.overlap)
        stride_x = (img_size_x - patch_size_x) / (num_patches_x - 1)
        stride_x = np.ceil(stride_x)
        img_size_x_new = (num_patches_x - 1) * stride_x + patch_size_x
        total_pad_x = img_size_x_new - img_size_x
        prepad_x = int(np.ceil(total_pad_x / 2))
        postpad_x = int(total_pad_x - prepad_x)

        if ((total_pad_y > 0) or (total_pad_x > 0)):
            image_batch = tf.pad(image_batch, [[0, 0], [prepad_y, postpad_y], [prepad_x, postpad_x], [0, 0]])

        # Crop selected locations and stack them in a batch
        ksizes = [1, patch_size_y, patch_size_x, 1]
        strides = [1, stride_y, stride_x, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'
        masked_image_batch = tf.image.extract_patches(image_batch, ksizes, strides, rates, padding)
        shape = (tf.shape(mask)[0], patch_size_y, patch_size_x, tf.shape(image_batch)[3])
        masked_image_batch = tf.reshape(masked_image_batch, shape)
        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        masked_image_batch = tf.gather_nd(params=masked_image_batch, indices=inds, name='gather_masked_batch')

        return masked_image_batch

    def call(self, image_batch, is_training, adv_eval_data=False, step=None, keep_step_summary=False):
        """Execute TNet inference.
        Args:
            image_batch: 4-D float Tensor; image batch to be processed.
                It is of size [batch_size, H, W, C], where batch_size
                is the number of images in the batch, H is their height,
                W is their width, and C is the number of channels.
            is_training: boolean; whether the model is in training phase.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            ret_lst: list; includes the following entries:
                logits: 2-D float Tensor; it contains classification logits for every
                    image. It is of size [batch_size, num_cls], where batch_size is
                    the number of images to classify, and num_cls is the number of classes.
                location_num_per_img: 2-D float Tensor; it contains the total number
                    of attended locations per image (the processing of the downsampled
                    version of each image in the 1st processing level is not counted).
                    It is of size [batch_size, 1], where batch_size is the number of
                    provided images to classify.
                location_log_probs (optional): 2-D float Tensor; it contains the sum
                    of log probabilities of attended locations, and is used for training.
                    It is of size [batch_size, 1], where batch_size is the number of
                    provided images to classify.
                logits_per_feature_lst (optional): list of 2-D float Tensors; each
                    Tensor contains classification logits from individual feature
                    vectors extracted at each processing level, and is used for
                    per-feature regularization during training. The list has
                    num_res_levels entries, and each entry is of size
                    [loc_per_lvl*batch_size, num_cls], where num_res_levels is the
                    number of processing levels that TNet goes through,
                    loc_per_lvl is the number of locations attended for each image
                    at each processing level (may be different for different levels),
                    batch_size is the number of provided images to classify, and
                    num_cls is the number of classes.
                logprobs_per_feature_lst (optional): list of 2-D float Tensors; each
                    Tensor contains the log probabilities of attended locations from
                    a different processing level, and is used for per-feature
                    regularization during training. The list has num_res_levels-1
                    entries, and each entry is of size [loc_per_lvl*batch_size, 1],
                    where num_res_levels is the number of processing levels
                    that TNet goes through, loc_per_lvl is the number of locations
                    attended for each image at each processing level (may be different
                    for different levels), and batch_size is the number of provided
                    images to classify.
                attended_locations (optional): 2-D float Tensor; it indicates which
                    locations are selected each time the location module is applied
                    during the processing of each image. Candidate locations within
                    each attention grid are flattened in a left to right and top to
                    bottom fashion, and the value 1 is assigned to selected
                    locations, while 0 is assigned to the rest. It is used for
                    advanced evaluation of the model. The Tensor is of size
                    [batch_size, num_att_per_img*locs_num], where batch_size is
                    the number of provided images to classify, num_att_per_img
                    is the total number of times the location module is applied
                    during the processing of each image, and locs_num is the
                    number of candidate locations within each attention grid.
                location_probs (optional): 2-D float Tensor; it contains the
                    attention probabilities of all candidate locations considered
                    during the processing of each image. It is used for advanced
                    evaluation of the model. The Tensor is of size
                    [batch_size, num_att_per_img*locs_num], where batch_size is
                    the number of provided images to classify, num_att_per_img
                    is the total number of times the location module is applied
                    during the processing of each image, and locs_num is the
                    number of candidate locations within each attention grid.
        """

        # Lists with information used in
        # per-feature regularization
        if (is_training):
            features_lst = []
            logits_per_feature_lst = []
            logprobs_per_feature_lst = []

        batch_size = tf.cast(tf.shape(image_batch)[0], tf.int32)
        # Create positional encodings for the feature vectors
        # of the first processing level; 2 spatial
        # dimensions and scale are encoded
        pos_y = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        pos_x = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        pos_rl = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, self.ls_dim, max_inv_afreq=1.0e2)

        # Number of locations that are attended for each image at
        # each processing level. loc_per_lvl is initialized to 1
        # for the 1st procesisng level; no actual locations are
        # attended, the downsampled version of each image is processed
        loc_per_lvl = 1

        # Iterate over the number of processing levels
        for res_level in range(self.num_res_levels):
            # Downscale the current processing level input to base resolution
            level_image_batch = tf.image.resize(image_batch, [self.base_res_y, self.base_res_x],
                                                method=tf.image.ResizeMethod.BILINEAR, antialias=False)   
            # Extract features with the feature extraction module;
            # loc_features and context_features will be provided as input
            # to the location module if processing will be extended to
            # an additional processing level
            loc_features, features = self.feature_extraction_module(level_image_batch, is_training, self.num_do_layers,
                                                                    self.keep_prob, step, keep_step_summary)
            context_features = features
            
            # Apply the positional encoding module
            features = self.positional_encoding_module(features, f_pos, step, keep_step_summary)
            # Keep all extracted features in a list
            # for per-feature regularizarion
            if (is_training):
                features_lst.append(features)

            # All extracted features for each image are kept in features_prime,
            # which is of size [batch_size, N, ls_dim], where batch_size is the
            # number of input images, N is the total number of exctracted features
            # for each image (N-1 attended locations, plus the feature vector from
            # the 1st processing level), and ls_dim is the dimensionality of the
            # final latent space
            features_cur = tf.reshape(features, [batch_size, loc_per_lvl, self.ls_dim])
            if (res_level == 0):
                features_prime = features_cur
            else:
                features_prime = tf.concat([features_prime, features_cur], axis=1)

            # Don't apply the location module in the last processing level
            if not (res_level == (self.num_res_levels - 1)):
                # Predict attention probabilities for all candidate
                # locations by applying the location module
                location_probs = self.location_module(loc_features, context_features, is_training, step, keep_step_summary)

                # Select the locations that will be processed in the next level,
                # and calculate quantities needed for training and evaluation
                lpg = tf.cast(tf.round(self.loc_per_grid[res_level]), dtype=tf.int32)
                ret_dict = self.create_boolean_mask(location_probs, is_training, adv_eval_data,
                                                    lpg, loc_per_lvl, batch_size)
                loc_per_lvl *= lpg

                # Update information used in training and evaluation
                if (is_training):
                    if (res_level == 0):
                        location_log_probs = ret_dict['location_log_probs']
                    else:
                        location_log_probs += ret_dict['location_log_probs']
                    logprobs_per_feature_lst.append(ret_dict['logprobs_per_feature'])
                elif (adv_eval_data):
                    if (res_level == 0):
                        attended_locations = ret_dict['attended_locations']
                        location_probs = ret_dict['location_probs']
                    else:
                        attended_locations = tf.concat([attended_locations, ret_dict['attended_locations']], axis=1)
                        location_probs = tf.concat([attended_locations, ret_dict['location_probs']], axis=1)
                if (res_level == 0):
                    location_num_per_img = ret_dict['location_num_per_img']
                else:
                    location_num_per_img += ret_dict['location_num_per_img']
                mask = ret_dict['mask']

                # Stack selected locations in a batch
                # for the next processing level
                image_batch = self.create_masked_batch(image_batch, mask)
                
                # Create positional vectors for the attended locations
                pos_y, pos_x, pos_rl = create_positional_vectors_components(pos_y, pos_x, pos_rl, mask, tf.shape(features)[0],
                                                                            self.num_patches_y, self.num_patches_x, res_level+1)
                f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, self.ls_dim, max_inv_afreq=1.0e2)
        
        # Calculate logits for the final classification,
        # by using the classification module
        features_prime = tf.reduce_mean(features_prime, axis=1)
        logits = self.classification_module(features_prime, step, keep_step_summary)
        
        # Calculate logits for each extracted feature vector
        # to use in per-feature regularization
        if (is_training):
            for i in range(self.num_res_levels):
                logits_per_feature_lst.append(self.classification_module(features_lst[i]))

        ret_lst = []
        ret_lst.append(logits) # pos 0
        ret_lst.append(location_num_per_img) # pos 1
        if (is_training):
            ret_lst.append(location_log_probs) # pos 2
            ret_lst.append(logits_per_feature_lst) # pos 3
            ret_lst.append(logprobs_per_feature_lst) # pos 4
        elif (adv_eval_data):
            ret_lst.append(attended_locations) # pos 2
            ret_lst.append(location_probs) # pos 3

        return ret_lst

class Baseline_CNN(tf.keras.Model):
    def __init__(self, descr_tag, num_cls, ls_dim, multi_factor, num_do_layers, keep_prob):
        super(Baseline_CNN, self).__init__(name='Baseline_CNN')
        """Initialize the baseline network.
        Args:
            descr_tag: string; description of the model to use for
                feature extraction.
            num_cls: int; number of classes.
            ls_dim: int; dimensionality of the feature latent space
                used for the final classification.
            multi_factor: int; multiplication factor of channels
                dimension in convolutional blocks.
            num_do_layers: int; number of spatial dropout layers.
            keep_prob: float; dropout keep probability.
        Returns:
            -
        """

        self.descr_tag = descr_tag
        self.num_cls = num_cls
        self.ls_dim = ls_dim
        self.multi_factor = multi_factor
        self.num_do_layers = num_do_layers
        self.keep_prob = keep_prob

        # Initialize baseline network components
        if ('BagNet_77' in self.descr_tag):
            self.feature_extractor = BagNet_77(self.descr_tag, self.multi_factor)
        self.classification_layer = classification_module_cls(self.num_cls, self.ls_dim)

    def call(self, image_batch, is_training, step=None, keep_step_summary=False):
        """Execute baseline CNN inference.
        Args:
            image_batch: 4-D float Tensor; image batch to be processed.
                It is of size [batch_size, H, W, C], where batch_size
                is the number of images in the batch, H is their height,
                W is their width, and C is the number of channels.
            is_training: boolean; whether the model is in training phase.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            logits: 2-D float Tensor; it contains classification
                logits for every image. It is of size [batch_size, num_cls],
                where batch_size is the number of images to
                classify, and num_cls is the number of classes.
        """

        # Extract features
        _, features = self.feature_extractor(image_batch, is_training, self.num_do_layers, self.keep_prob, step, keep_step_summary)
        # Calculate logits for the final classification
        logits = self.classification_layer(features, step, keep_step_summary)

        return logits


















































class TNet_0(tf.keras.Model):
    def __init__(self, descr_tag, ls_dim, num_patches_y, num_patches_x, overlap, num_res_levels,
                 num_cls, base_res_y, base_res_x, multi_factor, num_do_layers, keep_prob, loc_per_grid):
        super(TNet_0, self).__init__(name='TNet')
        """Initialization of TNet.
        Args:
            descr_tag: string; description of the model to use
                as the feature extraction module.
            ls_dim: int; dimensionality of the feature latent space
                used for the final classification.
            num_patches_y: int; number of patches on the vertical dimension
                of the grid considered by the location module.
            num_patches_x: int; number of patches on the horizontal dimension
                of the grid considered by the location module.
            overlap: float; the fraction of each spatial image dimension
                occupied by the correspending dimension of a grid patch
                (same for both spatial dimensions).
            num_res_levels: int; number of maximum processing levels
                that TNet goes through.
            num_cls: int; number of classes.
            base_res_y: int; base resolution of the feature extractor
                in the vertical dimension.
            base_res_x: int; base resolution of the feature extractor
                in the horizontal dimension.
            multi_factor: int; multiplication factor of channels
                dimension in convolutional blocks.
            num_do_layers: int; number of spatial dropout layers.
            keep_prob: float; dropout keep probability.
            loc_per_grid: float list; number of locations to attend
                per attention grid; one entry per processing level.
        Returns:
            -
        """
        
        self.descr_tag = descr_tag
        self.ls_dim = ls_dim
        self.num_patches_y = num_patches_y
        self.num_patches_x = num_patches_x
        self.overlap = overlap
        self.num_res_levels = num_res_levels
        self.num_cls = num_cls
        self.base_res_y = base_res_y
        self.base_res_x = base_res_x
        self.multi_factor = multi_factor
        self.num_do_layers = num_do_layers
        self.keep_prob = keep_prob
        self.loc_per_grid = loc_per_grid

        if ('BagNet_77' in self.descr_tag):
            self.feature_extraction_module = BagNet_77(self.descr_tag, self.multi_factor)
        self.positional_encoding_module = positional_encoding_module_cls(self.ls_dim)
        self.location_module = location_module_cls(self.ls_dim, descr_tag=self.descr_tag)
        self.classification_module = classification_module_cls(self.num_cls, self.ls_dim)

        # Baseline variable used during training
        self.baseline_var = tf.Variable(0.5, trainable=False, name='baseline_var',
                                        dtype=tf.float32, aggregation=tf.VariableAggregation.MEAN)

    def create_boolean_mask(self, location_probs, is_training, adv_eval_data, loc_per_grid):
        """Select the locations that will be processed in
           the next level, and calculate quantities needed
           or training and evaluation.
        Args:
            location_probs: 2-D float Tensor; it contains attention probabilities,
                it is of size [batch_size, locs_num], where batch_size is the
                number of images processed in the current level, and locs_num is
                the number of locations in an attention grid.
            is_training: boolean; whether the model is training.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
            loc_per_grid: int; number of locations to attend.
        Returns:
            ret_dict: dictionary; includes a number of entries
                used for training and evaluation.
        """

        # Select loc_per_grid locations to attend in the the next
        # processing level, by finding the top loc_per_grid
        # probabilities in location_probs for each image
        _, indices = tf.math.top_k(location_probs, k=loc_per_grid)
        # Create a mask of size [batch_size, locs_num], with 1 for
        # selected locations, and 0 for the rest
        batch_size = tf.shape(location_probs)[0]
        locs_num = tf.shape(location_probs)[1]
        r = tf.expand_dims(tf.range(start=0, limit=batch_size*locs_num, delta=locs_num, dtype=tf.int32), -1)
        indices += r
        indices = tf.reshape(indices, shape=[-1, 1])
        mask_shape = tf.shape(tf.reshape(location_probs, shape=[-1, 1]))
        mask = tf.scatter_nd(indices=indices, updates=tf.ones_like(indices, tf.float32), shape=mask_shape)
        mask = tf.reshape(mask, tf.shape(location_probs))
        mask = tf.stop_gradient(mask)
        
        ret_dict = {}
        if (is_training): # training
            # location_log_probs are used in the main learning rule, and
            # logprobs_per_feature are used in per-feature regularization
            location_log_probs = mask * tf.math.log(location_probs + 1e-10)
            logprobs_per_feature = tf.reshape(location_log_probs, [-1, 1])
            inds = tf.cast(tf.where(tf.reshape(mask, [-1])), tf.int32)
            logprobs_per_feature = tf.gather_nd(params=logprobs_per_feature, indices=inds)
            location_log_probs = tf.reduce_sum(location_log_probs, axis=1, keepdims=True)
            
            ret_dict['location_log_probs'] = location_log_probs
            ret_dict['logprobs_per_feature'] = logprobs_per_feature
        elif (adv_eval_data): # advanced evaluation
            ret_dict['attended_locations'] = mask
            ret_dict['location_probs'] = location_probs

        location_num_per_img = tf.reduce_sum(mask, axis=1, keepdims=True)
        mask = tf.reshape(tf.equal(mask, 1.0), [-1])
        ret_dict['location_num_per_img'] = location_num_per_img
        ret_dict['mask'] = mask

        return ret_dict

    def create_masked_batch(self, image_batch, mask):
        """Organize selected locations in a batch
           for the next processing level.
        Args:
            image_batch: 4-D float Tensor; image batch to be reorganized
                for the next processing level.
            mask: 1-D boolean Tensor; mask of size [batch_size*locs_num],
                where batch_size is the number of images processed in the
                current level, and locs_num is the number of locations in
                an attention grid. It contains True values for the selected
                locations, and False for the rest.
        Returns:
            masked_image_batch: 4-D float Tensor; image batch that contains
                the selected locations to be processed in the next level.
        """

        img_size_y = float(image_batch.shape[1])
        img_size_x = float(image_batch.shape[2])

        # Find stride and patch size per dimension, in order to
        # crop the selected locations
        num_patches_y = float(self.num_patches_y)
        patch_size_y = np.round(img_size_y*self.overlap)
        stride_y = (img_size_y - patch_size_y) / (num_patches_y - 1)
        stride_y = np.ceil(stride_y)
        img_size_y_new = (num_patches_y - 1) * stride_y + patch_size_y
        total_pad_y = img_size_y_new - img_size_y
        prepad_y = int(np.ceil(total_pad_y / 2))
        postpad_y = int(total_pad_y - prepad_y)

        num_patches_x = float(self.num_patches_x)
        patch_size_x = np.round(img_size_x*self.overlap)
        stride_x = (img_size_x - patch_size_x) / (num_patches_x - 1)
        stride_x = np.ceil(stride_x)
        img_size_x_new = (num_patches_x - 1) * stride_x + patch_size_x
        total_pad_x = img_size_x_new - img_size_x
        prepad_x = int(np.ceil(total_pad_x / 2))
        postpad_x = int(total_pad_x - prepad_x)

        if ((total_pad_y > 0) or (total_pad_x > 0)):
            image_batch = tf.pad(image_batch, [[0, 0], [prepad_y, postpad_y], [prepad_x, postpad_x], [0, 0]])

        # Crop selected locations and stack them in a batch
        ksizes = [1, patch_size_y, patch_size_x, 1]
        strides = [1, stride_y, stride_x, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'
        masked_image_batch = tf.image.extract_patches(image_batch, ksizes, strides, rates, padding)
        shape = (tf.shape(mask)[0], patch_size_y, patch_size_x, tf.shape(image_batch)[3])
        masked_image_batch = tf.reshape(masked_image_batch, shape)
        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        masked_image_batch = tf.gather_nd(params=masked_image_batch, indices=inds, name='gather_masked_batch')

        return masked_image_batch

    def create_unmasked_features(self, input_features, mask, num_images):
        feature_size_y = input_features.shape[1]
        feature_size_x = input_features.shape[2]
        num_feature_channels = input_features.shape[3]

        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        # shape = tf.constant([tf.shape(mask)[0], feature_size_y, feature_size_x, num_feature_channels])
        shape = (tf.shape(mask)[0], feature_size_y, feature_size_x, num_feature_channels)
        scattered_features = tf.scatter_nd(indices=inds, updates=input_features, shape=tf.shape(tf.zeros(shape)))

        shape = (num_images, self.num_patches_y, self.num_patches_x, feature_size_y*feature_size_x, num_feature_channels)
        unmasked_features = tf.reshape(scattered_features, shape)
        unmasked_features = tf.split(unmasked_features, feature_size_y*feature_size_x, axis=3)
        unmasked_features = tf.stack(unmasked_features, axis=0)
        shape = (num_images*feature_size_y*feature_size_x, self.num_patches_y, self.num_patches_x, num_feature_channels)
        unmasked_features = tf.reshape(unmasked_features, shape)

        unmasked_features = tf.compat.v1.batch_to_space_nd(unmasked_features, [feature_size_y, feature_size_x], [[0,0],[0,0]])

        return unmasked_features

    def create_unmasked_locations(self, locations_L, mask, to_sum, to_sum_per_grid_loc=False):
        num_pathes_per_image = self.num_patches_y * self.num_patches_x
        num_l = tf.shape(locations_L)[1]

        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        shape = (tf.shape(mask)[0], num_l)
        scattered_l = tf.scatter_nd(indices=inds, updates=locations_L, shape=tf.shape(tf.zeros(shape)))

        if (to_sum_per_grid_loc):
            unmasked_locations = tf.reshape(scattered_l, [-1, num_pathes_per_image, num_l])
            unmasked_locations = tf.reduce_sum(unmasked_locations, axis=1, keepdims=False)
        else:
            unmasked_locations = tf.reshape(scattered_l, [-1, (num_l * num_pathes_per_image)])
            if (to_sum):
                unmasked_locations = tf.reduce_sum(unmasked_locations, axis=1, keepdims=True)

        return unmasked_locations

    def feature_back_prop(self, features_lst, masks_lst, location_num_per_img_helper_lst,
                          res_level, is_training, adv_eval_data,
                          location_log_probs_helper_lst=None,
                          attended_locations_helper_lst=None, location_probs_helper_lst=None):
        features_prime = features_lst[res_level]
        for j in range(res_level, 0, -1):
            mask = masks_lst[j-1]
            features = features_lst[j-1]
            features_prime = self.create_unmasked_features(features_prime, mask, tf.shape(features)[0])
            features_prime = features + tf.reduce_sum(features_prime, axis=[1, 2], keepdims=True)
            # features_prime = tf.reduce_sum(features_prime, axis=[1, 2], keepdims=True)

            if (j == res_level):
                if (is_training):
                    location_log_probs_prime = location_log_probs_helper_lst[j-1]
                elif (adv_eval_data):
                    attended_locations_prime = attended_locations_helper_lst[j-1]
                    location_probs_prime = location_probs_helper_lst[j-1]
                location_num_per_img_prime = location_num_per_img_helper_lst[j-1]
            else:
                if (is_training):
                    location_log_probs = location_log_probs_helper_lst[j-1]
                    location_log_probs_prime = self.create_unmasked_locations(location_log_probs_prime, mask, to_sum=True)
                    location_log_probs_prime = location_log_probs + location_log_probs_prime

                elif (adv_eval_data):
                    attended_locations = attended_locations_helper_lst[j-1]
                    attended_locations_prime = self.create_unmasked_locations(attended_locations_prime, mask, to_sum=False)
                    attended_locations_prime = tf.concat([attended_locations, attended_locations_prime], axis=1)

                    location_probs = location_probs_helper_lst[j-1]
                    location_probs_prime = self.create_unmasked_locations(location_probs_prime, mask, to_sum=False)
                    location_probs_prime = tf.concat([location_probs, location_probs_prime], axis=1)

                location_num_per_img = location_num_per_img_helper_lst[j-1]
                location_num_per_img_prime = self.create_unmasked_locations(location_num_per_img_prime, mask, to_sum=True)
                location_num_per_img_prime = location_num_per_img + location_num_per_img_prime
        
        ret_dict = {}
        if (res_level > 0):
            if is_training:
                ret_dict['location_log_probs'] = location_log_probs_prime
            elif (adv_eval_data):
                ret_dict['attended_locations'] = attended_locations_prime
                ret_dict['location_probs'] = location_probs_prime
            ret_dict['location_num_per_img'] = tf.reduce_mean(location_num_per_img_prime)
        else: # Only 1 processing level was executed and location module wasn't used.
            if is_training:
                ret_dict['location_log_probs'] = tf.constant([[0.0]])
            elif (adv_eval_data):
                ret_dict['attended_locations'] = tf.constant([1])
                ret_dict['location_probs'] = tf.constant([1.0])
            ret_dict['location_num_per_img'] = tf.constant([0.0])

        # Get the mean of the added feature vectors for magnitude invariance
        loc_per_lvl = 1
        total_loc_per_img = 1
        for i in range(res_level):
            loc_per_lvl *= self.loc_per_grid[i]
            total_loc_per_img += loc_per_lvl
        total_loc_per_img = tf.cast(total_loc_per_img, tf.float32)
        # tf.print(total_loc_per_img)
        features_prime = features_prime * (1.0 / total_loc_per_img)
        ret_dict['features_prime'] = features_prime
        
        return ret_dict

    def call(self, image_batch, is_training, adv_eval_data=False, step=None, keep_step_summary=False, lpg_arg=None):
        """TNet inference.
        Args:
            image_batch: 4-D float Tensor; image batch to be processed.
            is_training: boolean; whether the model is training.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
            lpg_arg: float list; number of locations to attend
                per attention grid; one entry per processing level
                (allows the number of attended locations to change
                dynamically).
        Returns:
            ret_lst: list; includes a number of entries tha include
                classification logits, and information needed for
                training and evaluation.
        """

        # Set up helper lists
        features_lst = []
        masks_lst = []
        if (is_training):
            location_log_probs_helper_lst = []
            logits_per_feature_lst = []
            logprobs_per_feature_lst = []
        elif (adv_eval_data):
            attended_locations_helper_lst = []
            location_probs_helper_lst = []
        location_num_per_img_helper_lst = []
            

        batch_size = tf.cast(tf.shape(image_batch)[0], tf.float32)
        # Tensors with the positions of attended locations; 2 spatial
        # dimensions and scale
        pos_y = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        pos_x = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        pos_rl = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)

        # Iterate over the number of processing levels
        for res_level in range(self.num_res_levels):
            # Downscale the current processing level input to base resolution
            layer_image_batch = tf.image.resize(image_batch, [self.base_res_y, self.base_res_x],
                                                method=tf.image.ResizeMethod.BILINEAR, antialias=False)   
            # Extract features with the feature extraction module
            feature_maps, features = self.feature_extraction_module(layer_image_batch, is_training, self.num_do_layers,
                                                                    self.keep_prob, step, keep_step_summary)
            context_features = features
            # Create positional encodings for the extracted feature vectors
            f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, self.ls_dim, max_inv_afreq=1.0e2)
            # Apply the positional encoding module
            features = self.positional_encoding_module(features, f_pos, step, keep_step_summary)
            # Keep the features that will be averaged for the final
            # classification in a list
            features_lst.append(features)

            # In the last processing level we don't apply the location module
            if not(res_level == (self.num_res_levels - 1)):
                # Select locations to be attended in the next level
                # by applying the location module
                location_probs = self.location_module(feature_maps, context_features, is_training, step, keep_step_summary)

                # Determine the number of locations to attend
                if (lpg_arg is None):
                    lpg = self.loc_per_grid[res_level]
                else:
                    lpg = lpg_arg[res_level]
                # Select the locations that will be processed in
                # the next level, and calculate quantities needed
                # for training and evaluation
                ret_dict = self.create_boolean_mask(location_probs, is_training, adv_eval_data, tf.cast(tf.round(lpg), dtype=tf.int32))

                # Update helper lists
                if (is_training):
                    location_log_probs_helper_lst.append(ret_dict['location_log_probs'])
                    logprobs_per_feature_lst.append(ret_dict['logprobs_per_feature'])
                elif (adv_eval_data):
                    attended_locations_helper_lst.append(ret_dict['attended_locations'])
                    location_probs_helper_lst.append(ret_dict['location_probs'])
                location_num_per_img_helper_lst.append(ret_dict['location_num_per_img'])
                mask = ret_dict['mask']
                masks_lst.append(mask)

                # Stack selected locations in a batch
                # for the next processing level
                image_batch = self.create_masked_batch(image_batch, mask)
                
                # Create positional vectors of attended locations
                pos_y, pos_x, pos_rl = create_positional_vectors_components(pos_y, pos_x, pos_rl, mask, tf.shape(features)[0],
                                                                            self.num_patches_y, self.num_patches_x, res_level+1)
        
        # Calculate logits after going through the maximum number of allowed processing levels
        if (is_training):
            ret_dict = self.feature_back_prop(features_lst, masks_lst, location_num_per_img_helper_lst,
                                              res_level, is_training, adv_eval_data,
                                              location_log_probs_helper_lst=location_log_probs_helper_lst)
        elif (adv_eval_data):
            ret_dict = self.feature_back_prop(features_lst, masks_lst, location_num_per_img_helper_lst,
                                              res_level, is_training, adv_eval_data,
                                              attended_locations_helper_lst=attended_locations_helper_lst,
                                              location_probs_helper_lst=location_probs_helper_lst)
        else:
            ret_dict = self.feature_back_prop(features_lst, masks_lst, location_num_per_img_helper_lst,
                                              res_level, is_training, adv_eval_data)
        
        # Calculate logits for the final classification,
        # by using the classification module
        features_prime = ret_dict['features_prime']
        logits_rl = self.classification_module(features_prime, step, keep_step_summary)
        
        # Calculate logits for each extracted feature vector
        # to use in per-feature regularization
        if (is_training):
            for i in range(self.num_res_levels):
                logits_per_feature_lst.append(self.classification_module(features_lst[i]))

        ret_lst = []
        ret_lst.append(logits_rl) # pos 0
        ret_lst.append(ret_dict['location_num_per_img']) # pos 1
        if (is_training):
            ret_lst.append(ret_dict['location_log_probs']) # pos 2
            ret_lst.append(logits_per_feature_lst) # pos 3
            ret_lst.append(logprobs_per_feature_lst) # pos 4
        elif (adv_eval_data):
            ret_lst.append(ret_dict['attended_locations']) # pos 2
            ret_lst.append(ret_dict['location_probs']) # pos 3

        return ret_lst

class TNet_check_changes(tf.keras.Model):
    def __init__(self, descr_tag, ls_dim, num_patches_y, num_patches_x, overlap, num_res_levels,
                 num_cls, base_res_y, base_res_x, multi_factor, num_do_layers, keep_prob, loc_per_grid):
        super(TNet_check_changes, self).__init__(name='TNet')
        """Initialization of TNet.
        Args:
            descr_tag: string; description of the model to use
                as the feature extraction module.
            ls_dim: int; dimensionality of the feature latent space
                used for the final classification.
            num_patches_y: int; number of patches on the vertical dimension
                of the grid considered by the location module.
            num_patches_x: int; number of patches on the horizontal dimension
                of the grid considered by the location module.
            overlap: float; the fraction of each spatial image dimension
                occupied by the correspending dimension of a grid patch
                (same for both spatial dimensions).
            num_res_levels: int; number of maximum processing levels
                that TNet goes through.
            num_cls: int; number of classes.
            base_res_y: int; base resolution of the feature extractor
                in the vertical dimension.
            base_res_x: int; base resolution of the feature extractor
                in the horizontal dimension.
            multi_factor: int; multiplication factor of channels
                dimension in convolutional blocks.
            num_do_layers: int; number of spatial dropout layers.
            keep_prob: float; dropout keep probability.
            loc_per_grid: float list; number of locations to attend
                per attention grid; one entry per processing level.
        Returns:
            -
        """
        
        self.descr_tag = descr_tag
        self.ls_dim = ls_dim
        self.num_patches_y = num_patches_y
        self.num_patches_x = num_patches_x
        self.overlap = overlap
        self.num_res_levels = num_res_levels
        self.num_cls = num_cls
        self.base_res_y = base_res_y
        self.base_res_x = base_res_x
        self.multi_factor = multi_factor
        self.num_do_layers = num_do_layers
        self.keep_prob = keep_prob
        self.loc_per_grid = loc_per_grid

        if ('BagNet_77' in self.descr_tag):
            self.feature_extraction_module = BagNet_77(self.descr_tag, self.multi_factor)
        self.positional_encoding_module = positional_encoding_module_cls(self.ls_dim)
        self.location_module = location_module_cls(self.ls_dim, descr_tag=self.descr_tag)
        self.classification_module = classification_module_cls(self.num_cls, self.ls_dim)

        # Baseline variable used during training
        self.baseline_var = tf.Variable(0.5, trainable=False, name='baseline_var',
                                        dtype=tf.float32, aggregation=tf.VariableAggregation.MEAN)

    def create_boolean_mask(self, location_probs, is_training, adv_eval_data,
                            loc_per_grid, loc_per_lvl, batch_size):
        """Select the locations that will be processed in
           the next level, and calculate quantities needed
           or training and evaluation.
        Args:
            location_probs: 2-D float Tensor; it contains attention probabilities,
                it is of size [batch_size, locs_num], where batch_size is the
                number of images processed in the current level, and locs_num is
                the number of locations in an attention grid.
            is_training: boolean; whether the model is training.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
            loc_per_grid: int; number of locations to attend per attention
                grid (they will be processed in the next level)
            loc_per_lvl: int; number of locations attended per
                image in the current processing level
            batch_size: float; number of images provided as input in
                the 1st processing level
        Returns:
            ret_dict: dictionary; includes a number of entries
                used for training and evaluation.
        """

        # Select loc_per_grid locations to attend in the the next
        # processing level, by finding the top loc_per_grid
        # probabilities in location_probs for each image
        _, indices = tf.math.top_k(location_probs, k=loc_per_grid)
        # Create a mask of size [batch_size, locs_num], with 1 for
        # selected locations, and 0 for the rest
        batch_size = tf.shape(location_probs)[0]
        locs_num = tf.shape(location_probs)[1]
        r = tf.expand_dims(tf.range(start=0, limit=batch_size*locs_num, delta=locs_num, dtype=tf.int32), -1)
        indices += r
        indices = tf.reshape(indices, shape=[-1, 1])
        mask_shape = tf.shape(tf.reshape(location_probs, shape=[-1, 1]))
        mask = tf.scatter_nd(indices=indices, updates=tf.ones_like(indices, tf.float32), shape=mask_shape)
        mask = tf.reshape(mask, tf.shape(location_probs))
        mask = tf.stop_gradient(mask)
        
        ret_dict = {}
        if (is_training): # training
            # location_log_probs are used in the main learning rule, and
            # logprobs_per_feature are used in per-feature regularization
            location_log_probs = mask * tf.math.log(location_probs + 1e-10)
            logprobs_per_feature = tf.reshape(location_log_probs, [-1, 1])
            inds = tf.cast(tf.where(tf.reshape(mask, [-1])), tf.int32)
            logprobs_per_feature = tf.gather_nd(params=logprobs_per_feature, indices=inds)

            location_log_probs_2 = tf.reduce_sum(location_log_probs, axis=1, keepdims=True)

            location_log_probs = tf.reduce_sum(location_log_probs, axis=1)
            # location_log_probs = tf.reshape(location_log_probs, [tf.cast(batch_size, tf.int32), loc_per_lvl, 1])
            location_log_probs = tf.reshape(location_log_probs, [batch_size, loc_per_lvl, 1])
            location_log_probs = tf.reduce_sum(location_log_probs, axis=1)
            
            ret_dict['location_log_probs'] = location_log_probs
            ret_dict['location_log_probs_2'] = location_log_probs_2
            ret_dict['logprobs_per_feature'] = logprobs_per_feature
        elif (adv_eval_data): # advanced evaluation
            # ret_dict['attended_locations'] = tf.reshape(mask, [tf.cast(batch_size, tf.int32), -1])
            # ret_dict['location_probs'] = tf.reshape(location_probs, [tf.cast(batch_size, tf.int32), -1])
            # ret_dict['attended_locations'] = tf.reshape(mask, [batch_size, -1])
            # ret_dict['location_probs'] = tf.reshape(location_probs, [batch_size, -1])
            ret_dict['attended_locations'] = tf.reshape(mask, [batch_size, loc_per_lvl*locs_num])
            ret_dict['location_probs'] = tf.reshape(location_probs, [batch_size, loc_per_lvl*locs_num])

            ret_dict['attended_locations_2'] = mask
            ret_dict['location_probs_2'] = location_probs

        
        location_num_per_img_2 = tf.reduce_sum(mask, axis=1, keepdims=True)
        ret_dict['location_num_per_img_2'] = location_num_per_img_2
        
        location_num_per_img = tf.reduce_sum(mask, axis=1)
        # location_num_per_img = tf.reshape(location_num_per_img, [tf.cast(batch_size, tf.int32), loc_per_lvl, 1])
        location_num_per_img = tf.reshape(location_num_per_img, [batch_size, loc_per_lvl, 1])
        location_num_per_img = tf.reduce_sum(location_num_per_img, axis=1)
        mask = tf.reshape(tf.equal(mask, 1.0), [-1])
        ret_dict['location_num_per_img'] = location_num_per_img
        ret_dict['mask'] = mask

        return ret_dict

    def create_masked_batch(self, image_batch, mask):
        """Organize selected locations in a batch
           for the next processing level.
        Args:
            image_batch: 4-D float Tensor; image batch to be reorganized
                for the next processing level.
            mask: 1-D boolean Tensor; mask of size [batch_size*locs_num],
                where batch_size is the number of images processed in the
                current level, and locs_num is the number of locations in
                an attention grid. It contains True values for the selected
                locations, and False for the rest.
        Returns:
            masked_image_batch: 4-D float Tensor; image batch that contains
                the selected locations to be processed in the next level.
        """

        img_size_y = float(image_batch.shape[1])
        img_size_x = float(image_batch.shape[2])

        # Find stride and patch size per dimension, in order to
        # crop the selected locations
        num_patches_y = float(self.num_patches_y)
        patch_size_y = np.round(img_size_y*self.overlap)
        stride_y = (img_size_y - patch_size_y) / (num_patches_y - 1)
        stride_y = np.ceil(stride_y)
        img_size_y_new = (num_patches_y - 1) * stride_y + patch_size_y
        total_pad_y = img_size_y_new - img_size_y
        prepad_y = int(np.ceil(total_pad_y / 2))
        postpad_y = int(total_pad_y - prepad_y)

        num_patches_x = float(self.num_patches_x)
        patch_size_x = np.round(img_size_x*self.overlap)
        stride_x = (img_size_x - patch_size_x) / (num_patches_x - 1)
        stride_x = np.ceil(stride_x)
        img_size_x_new = (num_patches_x - 1) * stride_x + patch_size_x
        total_pad_x = img_size_x_new - img_size_x
        prepad_x = int(np.ceil(total_pad_x / 2))
        postpad_x = int(total_pad_x - prepad_x)

        if ((total_pad_y > 0) or (total_pad_x > 0)):
            image_batch = tf.pad(image_batch, [[0, 0], [prepad_y, postpad_y], [prepad_x, postpad_x], [0, 0]])

        # Crop selected locations and stack them in a batch
        ksizes = [1, patch_size_y, patch_size_x, 1]
        strides = [1, stride_y, stride_x, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'
        masked_image_batch = tf.image.extract_patches(image_batch, ksizes, strides, rates, padding)
        shape = (tf.shape(mask)[0], patch_size_y, patch_size_x, tf.shape(image_batch)[3])
        masked_image_batch = tf.reshape(masked_image_batch, shape)
        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        masked_image_batch = tf.gather_nd(params=masked_image_batch, indices=inds, name='gather_masked_batch')

        return masked_image_batch

    def create_unmasked_features(self, input_features, mask, num_images):
        feature_size_y = input_features.shape[1]
        feature_size_x = input_features.shape[2]
        num_feature_channels = input_features.shape[3]

        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        # shape = tf.constant([tf.shape(mask)[0], feature_size_y, feature_size_x, num_feature_channels])
        shape = (tf.shape(mask)[0], feature_size_y, feature_size_x, num_feature_channels)
        scattered_features = tf.scatter_nd(indices=inds, updates=input_features, shape=tf.shape(tf.zeros(shape)))

        shape = (num_images, self.num_patches_y, self.num_patches_x, feature_size_y*feature_size_x, num_feature_channels)
        unmasked_features = tf.reshape(scattered_features, shape)
        unmasked_features = tf.split(unmasked_features, feature_size_y*feature_size_x, axis=3)
        unmasked_features = tf.stack(unmasked_features, axis=0)
        shape = (num_images*feature_size_y*feature_size_x, self.num_patches_y, self.num_patches_x, num_feature_channels)
        unmasked_features = tf.reshape(unmasked_features, shape)

        unmasked_features = tf.compat.v1.batch_to_space_nd(unmasked_features, [feature_size_y, feature_size_x], [[0,0],[0,0]])

        return unmasked_features

    def create_unmasked_locations(self, locations_L, mask, to_sum, to_sum_per_grid_loc=False):
        num_pathes_per_image = self.num_patches_y * self.num_patches_x
        num_l = tf.shape(locations_L)[1]

        inds = tf.cast(tf.where(tf.equal(mask, True)), tf.int32)
        shape = (tf.shape(mask)[0], num_l)
        scattered_l = tf.scatter_nd(indices=inds, updates=locations_L, shape=tf.shape(tf.zeros(shape)))

        if (to_sum_per_grid_loc):
            unmasked_locations = tf.reshape(scattered_l, [-1, num_pathes_per_image, num_l])
            unmasked_locations = tf.reduce_sum(unmasked_locations, axis=1, keepdims=False)
        else:
            unmasked_locations = tf.reshape(scattered_l, [-1, (num_l * num_pathes_per_image)])
            if (to_sum):
                unmasked_locations = tf.reduce_sum(unmasked_locations, axis=1, keepdims=True)

        return unmasked_locations

    def feature_back_prop(self, features_lst, masks_lst, location_num_per_img_helper_lst,
                          res_level, is_training, adv_eval_data,
                          location_log_probs_helper_lst=None,
                          attended_locations_helper_lst=None, location_probs_helper_lst=None):
        features_prime = features_lst[res_level]
        for j in range(res_level, 0, -1):
            mask = masks_lst[j-1]
            features = features_lst[j-1]
            features_prime = self.create_unmasked_features(features_prime, mask, tf.shape(features)[0])
            features_prime = features + tf.reduce_sum(features_prime, axis=[1, 2], keepdims=True)
            # features_prime = tf.reduce_sum(features_prime, axis=[1, 2], keepdims=True)

            if (j == res_level):
                if (is_training):
                    location_log_probs_prime = location_log_probs_helper_lst[j-1]
                elif (adv_eval_data):
                    attended_locations_prime = attended_locations_helper_lst[j-1]
                    location_probs_prime = location_probs_helper_lst[j-1]
                location_num_per_img_prime = location_num_per_img_helper_lst[j-1]
            else:
                if (is_training):
                    location_log_probs = location_log_probs_helper_lst[j-1]
                    location_log_probs_prime = self.create_unmasked_locations(location_log_probs_prime, mask, to_sum=True)
                    location_log_probs_prime = location_log_probs + location_log_probs_prime

                elif (adv_eval_data):
                    attended_locations = attended_locations_helper_lst[j-1]
                    attended_locations_prime = self.create_unmasked_locations(attended_locations_prime, mask, to_sum=False)
                    attended_locations_prime = tf.concat([attended_locations, attended_locations_prime], axis=1)

                    location_probs = location_probs_helper_lst[j-1]
                    location_probs_prime = self.create_unmasked_locations(location_probs_prime, mask, to_sum=False)
                    location_probs_prime = tf.concat([location_probs, location_probs_prime], axis=1)

                location_num_per_img = location_num_per_img_helper_lst[j-1]
                location_num_per_img_prime = self.create_unmasked_locations(location_num_per_img_prime, mask, to_sum=True)
                location_num_per_img_prime = location_num_per_img + location_num_per_img_prime
        
        ret_dict = {}
        if (res_level > 0):
            if is_training:
                ret_dict['location_log_probs'] = location_log_probs_prime
            elif (adv_eval_data):
                ret_dict['attended_locations'] = attended_locations_prime
                ret_dict['location_probs'] = location_probs_prime
            ret_dict['location_num_per_img'] = tf.reduce_mean(location_num_per_img_prime)
        else: # Only 1 processing level was executed and location module wasn't used.
            if is_training:
                ret_dict['location_log_probs'] = tf.constant([[0.0]])
            elif (adv_eval_data):
                ret_dict['attended_locations'] = tf.constant([1])
                ret_dict['location_probs'] = tf.constant([1.0])
            ret_dict['location_num_per_img'] = tf.constant([0.0])

        # Get the mean of the added feature vectors for magnitude invariance
        loc_per_lvl = 1
        total_loc_per_img = 1
        for i in range(res_level):
            loc_per_lvl *= self.loc_per_grid[i]
            total_loc_per_img += loc_per_lvl
        total_loc_per_img = tf.cast(total_loc_per_img, tf.float32)
        # tf.print(total_loc_per_img)
        features_prime = features_prime * (1.0 / total_loc_per_img)
        ret_dict['features_prime'] = features_prime
        
        return ret_dict

    def call(self, image_batch, is_training, adv_eval_data=False, step=None, keep_step_summary=False, lpg_arg=None):
        """TNet inference.
        Args:
            image_batch: 4-D float Tensor; image batch to be processed.
            is_training: boolean; whether the model is training.
            adv_eval_data: boolean; whether to return additional information
                that is used for advanced evaluation of the model.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
            lpg_arg: float list; number of locations to attend
                per attention grid; one entry per processing level
                (its purpose is to allow the number of attended
                locations to change at different calls).
        Returns:
            ret_lst: list; includes classification logits, and 
                information needed for training and evaluation.
        """

        # Set up helper lists
        features_lst = []
        masks_lst = []
        if (is_training):
            location_log_probs_helper_lst = []
            logits_per_feature_lst = []
            logprobs_per_feature_lst = []
        elif (adv_eval_data):
            attended_locations_helper_lst = []
            location_probs_helper_lst = []
        location_num_per_img_helper_lst = []
            

        # batch_size = tf.cast(tf.shape(image_batch)[0], tf.float32)
        batch_size = tf.cast(tf.shape(image_batch)[0], tf.int32)
        # Create positional encodings for the feature vectors
        # of the first processing level; 2 spatial
        # dimensions and scale are encoded.
        pos_y = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        pos_x = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        pos_rl = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
        f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, self.ls_dim, max_inv_afreq=1.0e2)

        # Number of locations that are attended for each image at
        # each processing level. loc_per_lvl is initialized to 1
        # for the 1st procesisng level; no actual locations are
        # attended, the downsampled version of each image is processed
        loc_per_lvl = 1

        # Iterate over the number of processing levels
        for res_level in range(self.num_res_levels):
            # Downscale the current processing level input to base resolution
            level_image_batch = tf.image.resize(image_batch, [self.base_res_y, self.base_res_x],
                                                method=tf.image.ResizeMethod.BILINEAR, antialias=False)   
            # Extract features with the feature extraction module;
            # loc_features and context_features will be provided as input
            # to the location module if processing will be extended to
            # an additional processing level
            loc_features, features = self.feature_extraction_module(level_image_batch, is_training, self.num_do_layers,
                                                                    self.keep_prob, step, keep_step_summary)
            context_features = features
            
            # Apply the positional encoding module
            features = self.positional_encoding_module(features, f_pos, step, keep_step_summary)
            # Keep all extracted features in a list
            features_lst.append(features)

            # All extracted features for each image are kept in features_prime,
            # which is of size [batch_size, N, ls_dim], where batch_size is the
            # number of input images, N is the total number of exctracted features
            # for each image (N-1 attended locations, plus the feature vector from
            # the 1st processing level), and ls_dim is the dimensiolity of the final
            # latent space
            # f_cur = tf.reshape(features, [tf.cast(batch_size, tf.int32), loc_per_lvl, self.ls_dim])
            f_cur = tf.reshape(features, [batch_size, loc_per_lvl, self.ls_dim])
            if (res_level == 0):
                features_prime = f_cur
            else:
                features_prime = tf.concat([features_prime, f_cur], axis=1)

            # In the last processing level we don't apply the location module
            if not(res_level == (self.num_res_levels - 1)):
                # Select locations to be attended in the next level
                # by applying the location module
                location_probs = self.location_module(loc_features, context_features, is_training, step, keep_step_summary)

                # Determine the number of locations to attend
                # in the next processing level
                if (lpg_arg is None):
                    lpg = self.loc_per_grid[res_level]
                else:
                    lpg = lpg_arg[res_level]
                lpg = tf.cast(tf.round(lpg), dtype=tf.int32)
                # Select the locations that will be processed in
                # the next level, and calculate quantities needed
                # for training and evaluation
                ret_dict = self.create_boolean_mask(location_probs, is_training, adv_eval_data,
                                                    lpg, loc_per_lvl, batch_size)
                loc_per_lvl *= lpg

                # Update helper lists
                if (is_training):
                    if (res_level == 0):
                        location_log_probs = ret_dict['location_log_probs']
                    else:
                        location_log_probs += ret_dict['location_log_probs']
                    location_log_probs_helper_lst.append(ret_dict['location_log_probs_2'])
                    logprobs_per_feature_lst.append(ret_dict['logprobs_per_feature'])
                elif (adv_eval_data):
                    if (res_level == 0):
                        attended_locations = ret_dict['attended_locations']
                        location_probs = ret_dict['location_probs']
                    else:
                        attended_locations = tf.concat([attended_locations, ret_dict['attended_locations']], axis=1)
                        location_probs = tf.concat([attended_locations, ret_dict['location_probs']], axis=1)
                    attended_locations_helper_lst.append(ret_dict['attended_locations_2'])
                    location_probs_helper_lst.append(ret_dict['location_probs_2'])
                if (res_level == 0):
                    location_num_per_img = ret_dict['location_num_per_img']
                else:
                    location_num_per_img += ret_dict['location_num_per_img']
                location_num_per_img_helper_lst.append(ret_dict['location_num_per_img_2'])
                mask = ret_dict['mask']
                masks_lst.append(mask)

                # Stack selected locations in a batch
                # for the next processing level
                image_batch = self.create_masked_batch(image_batch, mask)
                
                # Create positional vectors of the attended locations
                pos_y, pos_x, pos_rl = create_positional_vectors_components(pos_y, pos_x, pos_rl, mask, tf.shape(features)[0],
                                                                            self.num_patches_y, self.num_patches_x, res_level+1)
                f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, self.ls_dim, max_inv_afreq=1.0e2)
        
        # Calculate logits after going through the maximum number of allowed processing levels
        if (is_training):
            ret_dict = self.feature_back_prop(features_lst, masks_lst, location_num_per_img_helper_lst,
                                              res_level, is_training, adv_eval_data,
                                              location_log_probs_helper_lst=location_log_probs_helper_lst)
        elif (adv_eval_data):
            ret_dict = self.feature_back_prop(features_lst, masks_lst, location_num_per_img_helper_lst,
                                              res_level, is_training, adv_eval_data,
                                              attended_locations_helper_lst=attended_locations_helper_lst,
                                              location_probs_helper_lst=location_probs_helper_lst)
        else:
            ret_dict = self.feature_back_prop(features_lst, masks_lst, location_num_per_img_helper_lst,
                                              res_level, is_training, adv_eval_data)
        
        
        
        tf.print(tf.math.reduce_mean(tf.cast(tf.math.equal(ret_dict['location_num_per_img'], location_num_per_img), tf.float32)))
        if (is_training):
            tf.print(tf.math.reduce_mean(tf.cast(tf.math.equal(ret_dict['location_log_probs'], location_log_probs), tf.float32)))
        elif (adv_eval_data):
            tf.print(tf.math.reduce_mean(tf.cast(tf.math.equal(ret_dict['attended_locations'], attended_locations), tf.float32)))
            tf.print(tf.math.reduce_mean(tf.cast(tf.math.equal(ret_dict['location_probs'], location_probs), tf.float32)))
        
        
        
        
        # Calculate logits for the final classification,
        # by using the classification module
        features_prime = tf.reduce_mean(features_prime, axis=1)
        # features_prime = ret_dict['features_prime']
        logits_rl = self.classification_module(features_prime, step, keep_step_summary)
        
        # Calculate logits for each extracted feature vector
        # to use in per-feature regularization
        if (is_training):
            for i in range(self.num_res_levels):
                logits_per_feature_lst.append(self.classification_module(features_lst[i]))

        ret_lst = []
        ret_lst.append(logits_rl) # pos 0
        # ret_lst.append(ret_dict['location_num_per_img']) # pos 1
        ret_lst.append(location_num_per_img) # pos 1
        if (is_training):
            # ret_lst.append(ret_dict['location_log_probs']) # pos 2
            ret_lst.append(location_log_probs) # pos 2
            ret_lst.append(logits_per_feature_lst) # pos 3
            ret_lst.append(logprobs_per_feature_lst) # pos 4
        elif (adv_eval_data):
            ret_lst.append(attended_locations) # pos 2
            ret_lst.append(location_probs) # pos 3

        return ret_lst

