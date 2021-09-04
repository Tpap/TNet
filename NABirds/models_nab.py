"""Set up models and utility functions.
"""

from __future__ import absolute_import, division, print_function

import os
import re
import sys

import tensorflow as tf
import numpy as np
import math
import copy
from tensorflow.keras import layers



CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

DEFAULT_EFFICIENTNET_BLOCKS_ARGS = [{
        # part 2
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'strides': 1,
        'se_ratio': 0.25,
        'padding': 'same'
    }, {
        # part 3
        'kernel_size': 3,
        'repeats': 2,
        'filters_in': 16,
        'filters_out': 24,
        'expand_ratio': 6,
        'strides': 2,
        'se_ratio': 0.25,
        'padding': 'same'
    }, {
        # part 4
        'kernel_size': 5,
        'repeats': 2,
        'filters_in': 24,
        'filters_out': 40,
        'expand_ratio': 6,
        'strides': 2,
        'se_ratio': 0.25,
        'padding': 'same'
    }, {
        # part 5
        'kernel_size': 3,
        'repeats': 3,
        'filters_in': 40,
        'filters_out': 80,
        'expand_ratio': 6,
        'strides': 2,
        'se_ratio': 0.25,
        'padding': 'same'
    }, {
        # part 6
        'kernel_size': 5,
        'repeats': 3,
        'filters_in': 80,
        'filters_out': 112,
        'expand_ratio': 6,
        'strides': 1,
        'se_ratio': 0.25,
        'padding': 'same'
    }, {
        # part 7
        'kernel_size': 5,
        'repeats': 4,
        'filters_in': 112,
        'filters_out': 192,
        'expand_ratio': 6,
        'strides': 2,
        'se_ratio': 0.25,
        'padding': 'same'
    }, {
        # part 8
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 192,
        'filters_out': 320,
        'expand_ratio': 6,
        'strides': 1,
        'se_ratio': 0.25,
        'padding': 'same'
}]

EFFICIENTNET_DICT = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224),
    'efficientnet-b1': (1.0, 1.1, 240),
    'efficientnet-b2': (1.1, 1.2, 260),
    'efficientnet-b3': (1.2, 1.4, 300),
    'efficientnet-b4': (1.4, 1.8, 380),
    'efficientnet-b5': (1.6, 2.2, 456),
    'efficientnet-b6': (1.8, 2.6, 528),
    'efficientnet-b7': (2.0, 3.1, 600),
    'efficientnet-b8': (2.2, 3.6, 672),
    'efficientnet-l2': (4.3, 5.3, 800),
    'efficientnet-l2-475': (4.3, 5.3, 475)
}

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

def round_filters(filters, width_coefficient, depth_divisor):
    """Scale the number of filters based on width coefficient.
    Args:
        filters: int; number of filters to scale.
        width_coefficient: float; coefficient for scaling filters.
        depth_divisor: int; determines quantization during depth scaling.
    Returns:
        new_filters: int; number of scaled filters.
    """

    filters *= width_coefficient
    new_filters = max(depth_divisor, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%
    if (new_filters < 0.9 * filters):
        new_filters += depth_divisor
    new_filters = int(new_filters)
    
    return new_filters

def round_repeats(repeats, depth_coefficient):
    """Scale the number of blocks based on depth coefficient.
    Args:
        repeats: int; number of blocks to scale.
        depth_coefficient: float; coefficient for scaling blocks.
    Returns:
        repeats: int; scaled number of blocks.
    """
    
    repeats = int(math.ceil(depth_coefficient * repeats))

    return repeats

class meshgrid_positional_encoding(layers.Layer):
    def __init__(self):
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
        super(meshgrid_positional_encoding, self).__init__(name='meshgrid_positional_encoding')

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

class MBconvBlock(layers.Layer):
    def __init__(self, kernel_size, filters_in, filters_out, stride, expand_ratio,
                 activation, part, block, se_ratio, block_drop_rate, batch_norm):
        super(MBconvBlock, self).__init__(name='part' + str(part) + '_block' + str(block))
        """Initialize a mobile inverted bottleneck convolutional (MBConv) block,
           according to https://arxiv.org/pdf/1807.11626.pdf.
        Args:
            kernel_size: int; kernel size of the depthwise convolutional layer.
            filters_in: int; number of input channnels.
            filters_out: int; number of output channnels.
            stride: int; stride of the depthwise
                convolutional layer.
            expand_ratio: int; multiplier for expanding the number of input
                channels.
            activation: string; type of activation.
            part: int; indicates the part of a network that the convolutional
                block belongs to.
            block: int; indicates the position of the convolutional block
                within a part of a network.
            se_ratio: float; multiplier for squeezing the number of input
                channels during squeeze and excitation.
            block_drop_rate: float; probability for dropping the blocks in the
                context of stochastic depth regularization.
            batch_norm: boolean; whether to use Batch Normalixation.
        Returns:
            -
        """

        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.activation = activation
        self.se_ratio = se_ratio
        self.block_drop_rate = block_drop_rate
        self.batch_norm = batch_norm
        self.filters = self.filters_in * self.expand_ratio
        if (self.batch_norm):
            use_bias = False
        else:
            use_bias = True
        
        # Initialize expansion layer
        if (self.expand_ratio != 1):
            self.expand_conv = layers.Conv2D(self.filters, kernel_size=1, strides=1, padding='same', use_bias=use_bias,
                                             kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                             name='expand/conv2d')
            self.expand_act = layers.Activation(self.activation, name='expand/act')
            if (self.batch_norm):
                self.batch_norm_expand = layers.BatchNormalization(axis=3, name='expand/batch_norm')

        # Initialize depthwise convolution layer
        self.dw_conv = layers.DepthwiseConv2D(kernel_size, strides=self.stride, padding='same', depth_multiplier=1,
                                              use_bias=use_bias, depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                              bias_initializer='zeros', name='depthwise/conv2d')
        self.dw_act = layers.Activation(self.activation, name='depthwise/act')
        if (self.batch_norm):
            self.batch_norm_dw = layers.BatchNormalization(axis=3, name='depthwise/batch_norm')

        # Initialize squeeze and excitation layers
        assert (0 < self.se_ratio <= 1)
        filters_se = max(1, int(self.filters_in * self.se_ratio))
        self.se_squeeze_gap = layers.GlobalAveragePooling2D(name='se/squeeze_gap')
        self.se_squeeze_reshape = layers.Reshape((1, 1, self.filters), name='se/squeeze_reshape')
        self.se_conv_1 = layers.Conv2D(filters_se, kernel_size=1, strides=1, padding='same', use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                       name='se/conv2d_1')
        self.se_act_1 = layers.Activation(self.activation, name='se/act_conv_1')
        self.se_conv_2 = layers.Conv2D(self.filters, kernel_size=1, strides=1, padding='same', use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                       name='se/conv2d_2')
        self.se_act_2 = layers.Activation('sigmoid', name='se/act_conv_2')
        
        # Initialize bottleneck convolutional layer
        self.bottleneck_conv = layers.Conv2D(self.filters_out, kernel_size=1, strides=1, padding='same', use_bias=use_bias,
                                             kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                             name='bottleneck/conv2d')
        if (self.batch_norm):
            self.batch_norm_bottleneck = layers.BatchNormalization(axis=3, name='bottleneck/batch_norm')

        # Initialize skip connection
        if ((self.stride == 1) and (self.filters_in == self.filters_out)):
            if (self.block_drop_rate > 0.):
                self.block_drop = layers.Dropout(self.block_drop_rate, noise_shape=(None, 1, 1, 1), name='skip/dropout')

    def call(self, features, is_training, step=None, keep_step_summary=False):
        """Apply the mobile inverted bottleneck convolutional (MBConv) block.
        Args:
            features: 4-D float Tensor; feature map to be processed.
                It is of size [batch_size, H, W, C], where batch_size
                is the number of images in the batch, H and W are the
                vertical and horizontal spatial dimensions of the
                feature map respectively, and C is the number of its
                channels.
            is_training: boolean; whether the model is in training phase.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            features: 4-D float Tensor; output feature map. It is of
                size [batch_size, H, W, C], where batch_size is the
                number of images in the batch, H and W are the vertical
                and horizontal spatial dimensions of the feature map
                respectively, and C is the number of its channels.
        """

        shortcut_features = features
        
        # Apply expansion layer
        if (self.expand_ratio != 1):
            features = self.expand_conv(features)
            if (self.batch_norm):
                features = self.batch_norm_expand(features, training=is_training)
            features = self.expand_act(features)
            if (keep_step_summary):
                tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Apply depthwise convolution layer
        features = self.dw_conv(features)
        if (self.batch_norm): 
            features = self.batch_norm_dw(features, training=is_training)
        features = self.dw_act(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Apply squeeze and excitation layers
        features_se = self.se_squeeze_gap(features)
        features_se = self.se_squeeze_reshape(features_se)
        features_se = self.se_conv_1(features_se)
        features_se = self.se_act_1(features_se)
        features_se = self.se_conv_2(features_se)
        features_se = self.se_act_2(features_se)
        features = layers.multiply([features, features_se], name='se/excite')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Apply bottleneck convolutional layer
        features = self.bottleneck_conv(features)
        if (self.batch_norm):
            features = self.batch_norm_bottleneck(features, training=is_training)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Add skip connection
        if ((self.stride == 1) and (self.filters_in == self.filters_out)):
            if (self.block_drop_rate > 0):
                features = self.block_drop(features, training=is_training)
            features = layers.add([features, shortcut_features], name='skip/add')
            if (keep_step_summary):
                tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        
        return features

class EfficientNet(layers.Layer):
    def __init__(self, descr_tag, width_coefficient, depth_coefficient,
                 block_drop_rate, depth_divisor, activation, batch_norm):
        super(EfficientNet, self).__init__(name='feature_extraction')
        """Initialize EfficientNet.
        Args:
            descr_tag: string; description of the model to use
                as the feature extraction module.
            width_coefficient: float; coefficient for scaling the width of
                of the feature extraction netwrok.
            depth_coefficient: float; coefficient for scaling the depth of
                of the feature extraction netwrok.
            block_drop_rate: float; the maximum probability for dropping model
                blocks during feature extraction (stochastic depth parameter).
            depth_divisor: int; determines quantization during depth scaling.
            activation: string; type of activation.
            batch_norm: boolean; whether to use Batch Normalixation.
        Returns:
            -
        """

        self.descr_tag = descr_tag
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.block_drop_rate = block_drop_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.batch_norm = batch_norm
        if (self.batch_norm):
            use_bias = False
        else:
            use_bias = True
        
        # Initialize stem block
        stem_filters = round_filters(32, self.width_coefficient, self.depth_divisor)
        self.stem_conv = layers.Conv2D(stem_filters, kernel_size=3, strides=2, padding='same', use_bias=use_bias,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros', name='stem/conv2d')
        if (self.batch_norm):
            self.batch_norm_stem = layers.BatchNormalization(axis=3, name='stem/batch_norm')
        self.stem_act = layers.Activation(self.activation, name='stem/act')

        # Initialize core blocks
        self.blocks_args = copy.deepcopy(DEFAULT_EFFICIENTNET_BLOCKS_ARGS)
        b = 0
        self.blocks_num = float(sum(round_repeats(args['repeats'], self.depth_coefficient) for args in self.blocks_args))
        self.core_blocks = []
        for (i, args) in enumerate(self.blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on width multiplier
            args['filters_in'] = round_filters(args['filters_in'], self.width_coefficient, self.depth_divisor)
            args['filters_out'] = round_filters(args['filters_out'], self.width_coefficient, self.depth_divisor)
            # Update block number based on depth multiplier
            self.blocks_args[i]['repeats'] = round_repeats(args.pop('repeats'), self.depth_coefficient)

            for j in range(self.blocks_args[i]['repeats']):
                b += 1
                # The first block needs to take care of stride and filter size increase
                if (j > 0):
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                bdr = self.block_drop_rate * b / self.blocks_num
                self.core_blocks.append(MBconvBlock(args['kernel_size'], args['filters_in'], args['filters_out'],
                                                    args['strides'], args['expand_ratio'], self.activation, part=(i+2),
                                                    block=(j+1), se_ratio=args['se_ratio'], block_drop_rate=bdr,
                                                    batch_norm=self.batch_norm))
        assert b == self.blocks_num

        # Initialize top block
        top_filters = round_filters(1280, self.width_coefficient, self.depth_divisor)
        self.top_conv = layers.Conv2D(top_filters, kernel_size=1, strides=1, padding='same', use_bias=use_bias,
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                      name='top/conv2d')
        if (self.batch_norm):
            self.batch_norm_top = layers.BatchNormalization(axis=3, name='top/batch_norm')
        self.top_act = layers.Activation(self.activation, name='top/act')
        self.top_gap = layers.GlobalAveragePooling2D(name='top/gap')
        self.top_reshape = layers.Reshape((1, 1, top_filters), name='top/reshape')
    
    def call(self, x, is_training, step=None, keep_step_summary=False):
        """Execute EfficientNet inference.
        Args:
            x: 4-D float Tensor; image batch to be processed.
                It is of size [batch_size, H, W, C], where batch_size
                is the number of images in the batch, H is their height,
                W is their width, and C is the number of channels.
            is_training: boolean; whether the model is in training phase.
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
        
        # Execute stem block
        features = self.stem_conv(x)
        if (self.batch_norm):
            features = self.batch_norm_stem(features, training=is_training)
        features = self.stem_act(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Execute core blocks
        b = 0
        for prt in range(len(self.blocks_args)):
            for blck in range(self.blocks_args[prt]['repeats']):
                features = self.core_blocks[b](features, is_training, step, keep_step_summary)
                if ('EfficientNetB0' in self.descr_tag):
                    if ((prt == 3) and (blck == 0)):
                        loc_features = features[:, 2:12:2, 2:12:2, :]
                elif (('EfficientNetB1' in self.descr_tag) or ('EfficientNetB2' in self.descr_tag)
                       or ('EfficientNetB3' in self.descr_tag) or ('EfficientNetB4' in self.descr_tag)
                       or ('EfficientNetB5' in self.descr_tag)):
                    if ((prt == 2) and (blck == 1)):
                        loc_features = features[:, 5:22:4, 5:22:4, :]
                elif (('EfficientNetB6' in self.descr_tag) or ('EfficientNetB7' in self.descr_tag)):
                    if ((prt == 2) and (blck == 0)):
                        loc_features = features[:, 5:22:4, 5:22:4, :]
                elif ('EfficientNetL2' in self.descr_tag):
                    if ((prt == 1) and (blck == 7)):
                        loc_features = features[:, 9:46:9, 9:46:9, :]
                b += 1
            if (keep_step_summary):
                tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        assert (b == self.blocks_num)
        
        # Execute top block
        features = self.top_conv(features)
        if (self.batch_norm):
            features = self.batch_norm_top(features, training=is_training)
        features = self.top_act(features)
        features = self.top_gap(features)
        features = self.top_reshape(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        return loc_features, features

class location_module_cls(layers.Layer):
    def __init__(self, activation, descr_tag, width_coefficient, depth_divisor):
        super(location_module_cls, self).__init__(name='location_prediction')
        """Initialize the location module.
        Args:
            activation: string; type of activation.
            descr_tag: string; description of the model to use
                as the feature extraction module.
            width_coefficient: float; coefficient for scaling the width of
                of the feature extraction netwrok.
            depth_divisor: int; determines quantization during depth scaling.
        Returns:
            -
        """

        self.activation = activation
        self.descr_tag = descr_tag
        self.width_coefficient = width_coefficient
        self.depth_divisor = depth_divisor
        self.blocks_args = copy.deepcopy(DEFAULT_EFFICIENTNET_BLOCKS_ARGS)

        # Initialize linear layers for 2-D positional encodings
        self.meshgrid_pos_enc = meshgrid_positional_encoding()

        # Determine the number of input channels based on the size of the attention grid
        if ('EfficientNetB0' in self.descr_tag):
            self.num_channels = round_filters(self.blocks_args[3]['filters_out'], self.width_coefficient, self.depth_divisor)
        elif ('EfficientNetL2' in self.descr_tag):
            self.num_channels = round_filters(self.blocks_args[1]['filters_out'], self.width_coefficient, self.depth_divisor)
        else:
            self.num_channels = round_filters(self.blocks_args[2]['filters_out'], self.width_coefficient, self.depth_divisor)

        # Initialize first convolutional layer
        self.init_conv = layers.Conv2D(self.num_channels, kernel_size=1, strides=1, padding='same', use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                       name='init/conv2d')
        self.init_act = layers.Activation(self.activation, name='init/act')

        # Initialize squeeze and excitation layers
        num_channels_se = max(1, int(self.num_channels * 0.5))
        self.se_squeeze_gap = layers.GlobalAveragePooling2D(name='se/squeeze_gap')
        self.se_squeeze_reshape = layers.Reshape((1, 1, self.num_channels), name='se/squeeze_reshape')
        self.se_conv_1 = layers.Conv2D(num_channels_se, kernel_size=1, strides=1, padding='same', use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                       name='se/conv2d_1')
        self.se_act_1 = layers.Activation(self.activation, name='se/act_conv_1')
        self.se_conv_2 = layers.Conv2D(self.num_channels, kernel_size=1, strides=1, padding='same', use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                       name='se/conv2d_2')
        self.se_act_2 = layers.Activation('sigmoid', name='se/act_conv_2')

        # Initialize final convolutional and linear layers
        self.final_conv = layers.Conv2D(self.num_channels, kernel_size=1, strides=1, padding='same', use_bias=True,
                                        kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                        name='final/conv2d')
        self.final_act = layers.Activation(self.activation, name='final/act')

        self.output_linear = layers.Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=True,
                                           kernel_initializer=CONV_KERNEL_INITIALIZER, bias_initializer='zeros',
                                           name='output_linear')

    def call(self, features, is_training, step=None, keep_step_summary=False):
        """Apply the location module.
        Args:
            features: 4-D float Tensor; feature map consisted
                of feature vectors (one at each spatial position)
                that describe the image patches within the attention
                grid considered by the location module. It is of size
                [batch_size, num_patches_y, num_patches_x, c], where
                batch_size is the number of images in the batch,
                num_patches_y and num_patches_x are the number of
                patches in the vertical and horizontal dimensions
                of the grid considered by the location module
                respectively, and c is the number of channels.
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
            features = tf.stop_gradient(features)
        
        # Apply first convolutional layer
        features = self.init_conv(features)
        features = self.init_act(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        # Apply squeeze and excitation to integrate context
        features_se = self.se_squeeze_gap(features)
        features_se = self.se_squeeze_reshape(features_se)
        features_se = self.se_conv_1(features_se)
        features_se = self.se_act_1(features_se)
        features_se = self.se_conv_2(features_se)
        features_se = self.se_act_2(features_se)
        features = layers.multiply([features, features_se], name='se/excite')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        
        # Add positional encodings
        features = self.meshgrid_pos_enc.add_positional_encoding(features)
        
        # Fuse features augmented with positional information
        features = self.final_conv(features)
        features = self.final_act(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        
        # Project features to l2 normalized logits of a
        # categorical distribution for each image in the batch
        features = self.output_linear(features)
        features = tf.reshape(features, [tf.shape(features)[0], -1])
        features, _ = tf.linalg.normalize(features, ord='euclidean', axis=-1)
        # Calculate the attention probabilities of candidate locations
        location_probs = tf.nn.softmax(features, axis=-1, name='softmax_probs')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(location_probs.name, self.name) + '/activations', data=location_probs, step=step)
        
        return location_probs

class positional_encoding_module_cls(layers.Layer):
    def __init__(self, ls_dim, activation, pos_dim_divisor):
        super(positional_encoding_module_cls, self).__init__(name='feature_posBurn')
        """Initialize the positional encoding module.
        Args:
            ls_dim: int; dimensionality of the feature latent space.
            activation: string; type of activation.
            pos_dim_divisor: int; it determines the dimensionality of positional
                embeddings, by dividing the latent space dimensionality (ls_dim).
        Returns:
            -
        """
        
        # Initialize layers
        self.ls_dim = ls_dim
        self.activation = activation
        self.pos_dim_divisor = pos_dim_divisor
        self.fc = layers.Dense(self.ls_dim, use_bias=True, kernel_initializer=DENSE_KERNEL_INITIALIZER,
                               bias_initializer='zeros', name='linear')
        self.act = layers.Activation(self.activation, name='posBurn/act') 

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

        features = tf.reshape(features, [-1, self.ls_dim])
        f_pos = tf.reshape(f_pos, [-1, self.ls_dim // self.pos_dim_divisor])
        # Project positional encodings to the features dimensionality
        f_pos = self.fc(f_pos)
        # Add positional encodings to the features
        features = layers.add([features, f_pos], name='posBurn/add')
        # Pass features through a non-linearity
        features = self.act(features)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)
        
        features = tf.expand_dims(tf.expand_dims(features, 1), 1)

        return features

class feature_weighting_module_cls(layers.Layer):
    def __init__(self, num_channels, activation):
        super(feature_weighting_module_cls, self).__init__(name='feat_weighting')
        """Initialize the feature weighting module.
        Args:
            num_channels: int; dimensionality of the feature latent space.
            activation: string; type of activation.
        Returns:
            -
        """

        self.activation = activation
        self.num_channels = num_channels

        # Initialize squeeze and excitation layers
        num_channels_se = max(1, int(self.num_channels * 0.25))
        self.se_dense_1 = layers.Dense(num_channels_se, use_bias=True, kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                       bias_initializer='zeros', name='se/dense_1')
        self.se_act_1 = layers.Activation(self.activation, name='se/act_1')
        self.se_dense_2 = layers.Dense(self.num_channels, use_bias=True, kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                       bias_initializer='zeros', name='se/dense_2')
        self.se_act_2 = layers.Activation('sigmoid', name='se/act_2')

        # Initialize output linear layer
        self.output_linear = layers.Dense(1, use_bias=True, kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                          bias_initializer='zeros', name='output_linear')

    def call(self, features, is_training, step=None, keep_step_summary=False):
        """Apply the feature weighting module.
        Args:
            features: 3-D float Tensor; it contains all feature vectors extracted 
                during the processing of a bach of images. It is of size
                [batch_size, N, ls_dim], where batch_size is the number of images,
                N is the total number of exctracted features for each image (N-1
                attended locations, plus the feature vector from the 1st processing
                level), and ls_dim is the dimensionality of the features' latent space.
            is_training: boolean; whether the model is in training phase.
            step: int; global optimization step (used for tf summaries).
            keep_step_summary: boolean; whether to keep tf summaries.
        Returns:
            features: 2-D float Tensor; it contains a weighted average of the feature
                vectors extracted from each image. It is of size [batch_size, ls_dim],
                where batch_size is the number of images, and ls_dim is the
                dimensionality of the features' latent space.
            feat_probs: 3-D float Tensor; it contains weighting probabilities
                for the feature vectors extracted from each image. It is of
                size [batch_size, N, 1], where batch_size is the number of
                images, and N is the total number of extracted features from
                each image (N-1 attended locations, plus the feature vector
                from the 1st processing level).
        """

        features_init = features

        # Apply squeeze and excitation to integrate context
        features_se = tf.reduce_mean(features, axis=1)
        # features_se size [batch_size, C]
        features_se = self.se_dense_1(features_se)
        features_se = self.se_act_1(features_se)
        features_se = self.se_dense_2(features_se)
        features_se = self.se_act_2(features_se)
        features_se = tf.expand_dims(features_se, axis=1)
        # features_se size [batch_size, 1, C]
        features = layers.multiply([features, features_se], name='se/excite')
        
        # Apply a linear projection to estimate logits;
        # the resulting features are of size [batch_size, N]
        features = self.output_linear(features)
        features = tf.squeeze(features, axis=2)
        
        # Calculate weighting probabilities
        feat_probs = tf.nn.softmax(features, axis=-1, name='softmax_probs')
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(feat_probs.name, self.name) + '/activations', data=feat_probs, step=step)
        feat_probs = tf.expand_dims(feat_probs, axis=-1)
        
        # Calculate a weighted average of feature vectors for each image
        weighted_features = layers.multiply([features_init, feat_probs], name='weighting')
        features = tf.reduce_sum(weighted_features, axis=1)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(features.name, self.name) + '/activations', data=features, step=step)

        feat_probs = tf.stop_gradient(feat_probs)

        return features, feat_probs

class classification_module_cls(layers.Layer):
    def __init__(self, num_cls, ls_dim, dropout_rate):
        super(classification_module_cls, self).__init__(name='feature_extraction/logits_layer')
        """Initialize the classification module.
        Args:
            num_cls: int; number of classes.
            ls_dim: int; dimensionality of the feature latent space
                used for classification.
            dropout_rate: float; dropout drop probability.
        Returns:
            -
        """
        
        # Initialize layers
        self.ls_dim = ls_dim
        self.dropout_rate = dropout_rate
        if (self.dropout_rate > 0.):
            self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
        self.linear = layers.Dense(num_cls, use_bias=True, kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                   bias_initializer='zeros', name='linear')

    def call(self, x, is_training, step=None, keep_step_summary=False):
        """Apply the classification module.
        Args:
            x: 2-D float Tensor; features that represented images to
                classify. It is of size [batch_size, ls_dim], where
                batch_size is the number of images to classify, and
                ls_dim is the dimensionality of the features.
            is_training: boolean; whether the model is in training phase.
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
        if (self.dropout_rate > 0.):
            x = self.dropout(x, training=is_training)
        logits = self.linear(x)
        if (keep_step_summary):
            tf.summary.histogram(name=adjust_var_name(logits.name, self.name) + '/activations', data=logits, step=step)
        
        return logits

class TNet(tf.keras.Model):
    def __init__(self, descr_tag, ls_dim, num_patches_y, num_patches_x, overlap, num_res_levels,
                 num_cls, base_res_y, base_res_x, dropout_rate, loc_per_grid, width_coefficient,
                 depth_coefficient, block_drop_rate, depth_divisor, activation, batch_norm,
                 pos_dim_divisor, feat_weighting):
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
            dropout_rate: float; dropout drop probability.
            loc_per_grid: list of floats; number of locations to attend
                per attention grid. It contains num_res_levels-1 entries.
            width_coefficient: float; coefficient for scaling the width of
                the feature extraction netwrok.
            depth_coefficient: float; coefficient for scaling the depth of
                the feature extraction netwrok.
            block_drop_rate: float; the maximum probability for dropping model
                blocks during feature extraction (stochastic depth parameter).
            depth_divisor: int; determines quantization during depth scaling.
            activation: string; type of activation.
            batch_norm: boolean; whether to use Batch Normalixation.
            pos_dim_divisor: int; it determines the dimensionality of positional
                embeddings, by dividing the latent space dimensionality (ls_dim).
            feat_weighting: boolean; whether to use the feature weighting module.
        Returns:
            -
        """

        self.descr_tag = descr_tag
        self.num_patches_y = num_patches_y
        self.num_patches_x = num_patches_x
        self.overlap = overlap
        self.num_res_levels = num_res_levels
        self.num_cls = num_cls
        self.base_res_y = base_res_y
        self.base_res_x = base_res_x
        self.dropout_rate = dropout_rate
        self.loc_per_grid = loc_per_grid
        if ('origWD' in self.descr_tag):
            if ('EfficientNetB0' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b0']
            elif ('EfficientNetB1' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b1']
            elif ('EfficientNetB2' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b2']
            elif ('EfficientNetB3' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b3']
            elif ('EfficientNetB4' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b4']
            elif ('EfficientNetB5' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b5']
            elif ('EfficientNetB6' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b6']
            elif ('EfficientNetB7' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b7']
            elif ('EfficientNetL2' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-l2']
        else:
            self.width_coefficient = width_coefficient
            self.depth_coefficient = depth_coefficient
        self.block_drop_rate = block_drop_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.batch_norm = batch_norm
        self.pos_dim_divisor = pos_dim_divisor
        self.feat_weighting = feat_weighting
        self.ls_dim = round_filters(ls_dim, self.width_coefficient, self.depth_divisor)

        if ('EfficientNet' in self.descr_tag):
            self.feature_extraction_module = EfficientNet(self.descr_tag, self.width_coefficient, self.depth_coefficient,
                                                          self.block_drop_rate, self.depth_divisor, self.activation, self.batch_norm)
        self.positional_encoding_module = positional_encoding_module_cls(self.ls_dim, self.activation, self.pos_dim_divisor)
        if (self.feat_weighting):
            self.feature_weighting_module = feature_weighting_module_cls(self.ls_dim, self.activation)
        self.location_module = location_module_cls(self.activation, self.descr_tag, self.width_coefficient, self.depth_divisor)
        self.classification_module = classification_module_cls(self.num_cls, self.ls_dim, self.dropout_rate)

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
                features_prime (optional): 2-D float Tensor; it contains a weighted
                    average of the feature vectors extracted from each image. It is
                    of size [batch_size, ls_dim], where batch_size is the number of
                    provided images to classify, and ls_dim is the dimensionality
                    of the features' latent space.
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
                feat_probs (optional): 3-D float Tensor; it contains weighting
                    probabilities for the feature vectors extracted from each image.
                    It is of size [batch_size, N, 1], where batch_size is the number
                    of provided images to classify, and N is the total number of
                    extracted features from each image (N-1 attended locations, plus
                    the feature vector from the 1st processing level).
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
        pos_dim = self.ls_dim // self.pos_dim_divisor
        f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, pos_dim, max_inv_afreq=1.0e2)

        # Number of locations that are attended for each image at
        # each processing level. loc_per_lvl is initialized to 1
        # for the 1st procesisng level; no actual locations are
        # attended, the downsampled version of each image is processed
        loc_per_lvl = 1

        # Iterate over the number of processing levels
        for res_level in range(self.num_res_levels):
            # Downsize input images to base resolution
            level_image_batch = tf.image.resize(image_batch, [self.base_res_y, self.base_res_x],
                                                method=tf.image.ResizeMethod.BILINEAR, antialias=False)   
            # Extract features with the feature extraction module;
            # loc_features will be provided as input to the location
            # module if processing will be extended to an additional processing level
            loc_features, features = self.feature_extraction_module(level_image_batch, is_training, step, keep_step_summary)

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
            if not(res_level == (self.num_res_levels - 1)):
                # Predict attention probabilities for all candidate
                # locations by applying the location module
                location_probs = self.location_module(loc_features, is_training, step, keep_step_summary)

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
                
                # Create positional vectors of new attended locations
                pos_y, pos_x, pos_rl = create_positional_vectors_components(pos_y, pos_x, pos_rl, mask, tf.shape(features)[0],
                                                                            self.num_patches_y, self.num_patches_x, res_level+1)
                f_pos = create_positional_vectors(pos_y, pos_x, pos_rl, pos_dim, max_inv_afreq=1.0e2)
        
        # Apply the feature weigthing module
        if (self.feat_weighting):
            features_prime, feat_probs = self.feature_weighting_module(features_prime, is_training, step, keep_step_summary)
        else:
            features_prime = tf.reduce_mean(features_prime, axis=1)
            feat_probs = None
        # Calculate logits for the final classification,
        # by using the classification module
        logits = self.classification_module(features_prime, is_training, step, keep_step_summary)
        
        # Calculate logits for each extracted feature vector
        # to use in per-feature regularization
        if (is_training):
            for i in range(self.num_res_levels):
                logits_per_feature_lst.append(self.classification_module(features_lst[i], is_training))

        ret_lst = []
        ret_lst.append(logits) # pos 0
        ret_lst.append(location_num_per_img) # pos 1
        if (is_training):
            ret_lst.append(location_log_probs) # pos 2
            ret_lst.append(logits_per_feature_lst) # pos 3
            ret_lst.append(logprobs_per_feature_lst) # pos 4
            ret_lst.append(features_prime) # pos 5
        elif (adv_eval_data):
            ret_lst.append(attended_locations) # pos 2
            ret_lst.append(location_probs) # pos 3
            ret_lst.append(feat_probs) # pos 4

        return ret_lst

class Baseline_CNN(tf.keras.Model):
    def __init__(self, descr_tag, num_cls, ls_dim, dropout_rate, width_coefficient,
                 depth_coefficient, block_drop_rate, depth_divisor, activation, batch_norm):
        super(Baseline_CNN, self).__init__(name='Baseline_CNN')
        """Initialize the baseline network.
        Args:
            descr_tag: string; description of the model to use for
                feature extraction.
            num_cls: int; number of classes.
            ls_dim: int; dimensionality of the feature latent space
                used for the final classification.
            dropout_rate: float; dropout drop probability.
            width_coefficient: float; coefficient for scaling the width of
                of the feature extraction netwrok.
            depth_coefficient: float; coefficient for scaling the depth of
                of the feature extraction netwrok.
            block_drop_rate: float; the maximum probability for dropping model
                blocks during feature extraction (stochastic depth parameter).
            depth_divisor: int; determines quantization during depth scaling.
            activation: string; type of activation.
            batch_norm: boolean; whether to use Batch Normalixation.
        Returns:
            -
        """

        self.descr_tag = descr_tag
        self.num_cls = num_cls
        self.dropout_rate = dropout_rate
        if ('origWD' in self.descr_tag):
            if ('EfficientNetB0' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b0']
            elif ('EfficientNetB1' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b1']
            elif ('EfficientNetB2' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b2']
            elif ('EfficientNetB3' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b3']
            elif ('EfficientNetB4' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b4']
            elif ('EfficientNetB5' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b5']
            elif ('EfficientNetB6' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b6']
            elif ('EfficientNetB7' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-b7']
            elif ('EfficientNetL2' in self.descr_tag):
                self.width_coefficient, self.depth_coefficient, _ = EFFICIENTNET_DICT['efficientnet-l2']
        else:
            self.width_coefficient = width_coefficient
            self.depth_coefficient = depth_coefficient
        self.block_drop_rate = block_drop_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.batch_norm = batch_norm
        self.ls_dim = round_filters(ls_dim, self.width_coefficient, self.depth_divisor)

        # Initialize baseline network components
        if ('EfficientNet' in self.descr_tag):
            self.feature_extractor = EfficientNet(self.descr_tag, self.width_coefficient, self.depth_coefficient,
                                                  self.block_drop_rate, self.depth_divisor, self.activation, self.batch_norm)
        self.classification_layer = classification_module_cls(self.num_cls, self.ls_dim, self.dropout_rate)

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
            features: 2-D float Tensor; it contains feature vectors extracted
                from each image. It is of size [batch_size, ls_dim], where
                batch_size is the number of provided images to classify, and
                ls_dim is the dimensionality of the features' latent space.
        """

        # Extract features
        _, features = self.feature_extractor(image_batch, is_training, step, keep_step_summary)
        # Calculate logits for the final classification
        logits = self.classification_layer(features, is_training, step, keep_step_summary)

        return logits, features
