import numpy as np
import tensorflow as tf

BLOCK_WIDTH = 2
BLOCK_HEIGHT = 3
ENCODE = 1
DECODE = 2


def hparams():
    """Initializes hyper parameters."""
    return tf.contrib.training.HParams(
        num_atoms=125,
        num_points=39,
        learning_rate=0.001,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.95,
        loss_norm=2,
        batch_size=32,
        num_dictionaries=5,
        num_atoms_bottleneck=5)


def losses_mean_error(res, ans, params):
    """Computes the mean error between predictions and groundtruth."""
    if params.loss_norm == 2:
        with tf.variable_scope("Mean_squared_error"):
            weights = tf.linalg.norm(ans, ord="fro", axis=[-2, -1])
            squared_error = tf.linalg.norm(res - ans, ord="fro", axis=[-2, -1])
            mean_squared_error = tf.reduce_mean(squared_error / weights)
            tf.losses.add_loss(mean_squared_error)
            return mean_squared_error
    elif params.loss_norm == 1:
        with tf.variable_scope("Mean_absolute_error"):
            weights = tf.reduce_sum(tf.abs(ans), axis=[-2, -1])
            absolute_error = tf.reduce_sum(tf.abs(res - ans), axis=[-2, -1])
            mean_absolute_error = tf.reduce_mean(absolute_error / weights)
            tf.losses.add_loss(mean_absolute_error)
            return mean_absolute_error


def get_dictionary(index, mode, in_channels, out_channels):
    """Yields a dictionary given its name and model.

    Args:
        index: The dictionary index.
        mode: The ENCODE or DECODE stage.
        in_channels: The number of input channels.
        out_channels: The number of output channels.

    Returns:
        A tuple of (weights, bias), where weights is a four-rank convolution
        filter, bias is a 1-D tensor.
    """
    name = "dictionary{:02d}".format(index)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if index == 0:
            weights = tf.get_variable(
                name="weights",
                shape=[BLOCK_HEIGHT, 1, in_channels, out_channels],
                dtype=tf.float32)
            if mode == DECODE:
                weights = tf.transpose(weights, [1, 2, 3, 0])
                weights = tf.reshape(
                    weights, [1, 1, in_channels, BLOCK_HEIGHT * out_channels])
                bias = tf.get_variable(
                    name="bias_decode",
                    shape=[BLOCK_HEIGHT * out_channels],
                    dtype=tf.float32)
                return weights, bias
        else:
            weights = tf.get_variable(
                name="weights",
                shape=[1, 1, in_channels, out_channels],
                dtype=tf.float32)
        if mode == ENCODE:
            bias = tf.get_variable(
                name="bias_encode",
                shape=[in_channels],
                dtype=tf.float32)
        elif mode == DECODE:
            bias = tf.get_variable(
                name="bias_decode",
                shape=[out_channels],
                dtype=tf.float32)
        else:
            raise Exception("Unknown mode!")
        return weights, bias


def _encode(value, index, in_channels, out_channels, variable_scope):
    """Encode the input measurements to its hidden representations.

    Args:
        value: The input measurements.
        index: The index of current dictionary.
        in_channels: The in_channels of dictionary.
        out_channels: The out_channels of dictionary.
        variable_scope: The variable_scope of encoder.
    """
    weights, bias = get_dictionary(index, ENCODE, in_channels, out_channels)
    with tf.variable_scope(variable_scope) as variable_scope1:
        with tf.name_scope(variable_scope1.original_name_scope):
            batch_size = tf.shape(value)[0]
            output_shape = [batch_size, BLOCK_HEIGHT, BLOCK_WIDTH, in_channels]
            res = tf.nn.conv2d_transpose(
                value=value,
                filter=weights,
                output_shape=output_shape,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv2d_transpose_{:d}".format(index))
            res = tf.nn.bias_add(res, bias, name="BiasAdd_{:d}".format(index))
            res = tf.nn.relu(res, name="Relu_{:d}".format(index))
            return res, variable_scope1


def encode(measurements, params):
    """Encodes the measurements into its hidden representations.

    Args:
        measurements: The measurements tensor. In shape
            [batch_size, 1, BLOCK_WIDTH, num_points].
        params: The hyper parameters.

    Returns:
        A tensor. In shape [batch_size, BLOCK_HEIGHT, BLOCK_WIDTH, num_atoms].
    """
    channels = np.linspace(
        params.num_atoms,
        params.num_atoms_bottleneck,
        params.num_dictionaries).astype(np.int64)
    with tf.variable_scope("Encoder") as variable_scope:
        representations, variable_scope = _encode(
            value=measurements,
            index=0,
            in_channels=params.num_atoms,
            out_channels=params.num_points,
            variable_scope=variable_scope)
    for i in range(params.num_dictionaries - 1):
        representations, variable_scope = _encode(
            value=representations,
            index=i + 1,
            in_channels=channels[i + 1],
            out_channels=channels[i],
            variable_scope=variable_scope)
    return representations


def _decode(value, index, in_channels, out_channels, variable_scope):
    """Decode the hidden representations to 3D shape.

    Args:
        value: The input hidden representation.
        index: The index of current dictionary.
        in_channels: The in_channels of dictionary.
        out_channels: The out_channels of dictionary.
        variable_scope: The variable_scope of decoder.
    """
    weights, bias = get_dictionary(index, DECODE, in_channels, out_channels)
    with tf.variable_scope(variable_scope) as variable_scope1:
        with tf.name_scope(variable_scope1.original_name_scope):
            res = tf.nn.conv2d(
                input=value,
                filter=weights,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv2d_{:d}".format(index))
            res = tf.nn.bias_add(res, bias, name="BiasAdd_{:d}".format(index))
            if index != 0:
                res = tf.nn.relu(res, name="Relu_{:d}".format(index))
            return res, variable_scope1


def decode(coefficients, params):
    """Decodes the representations into its shapes.

    Args:
        coefficients: The coefficients tensor. In shape
            [batch_size, 1, 1, num_atoms].
        params: The hyper parameters.

    Returns:
        A tensor. In shape [batch_size, num_points, BLOCK_HEIGHT].
    """
    channels = np.linspace(
        params.num_atoms,
        params.num_atoms_bottleneck,
        params.num_dictionaries).astype(np.int64)
    with tf.variable_scope("Decoder") as variable_scope:
        shapes = coefficients
    for i in range(params.num_dictionaries - 1, 0, -1):
        shapes, variable_scope = _decode(
            value=shapes,
            index=i,
            in_channels=channels[i],
            out_channels=channels[i - 1],
            variable_scope=variable_scope)
    shapes, variable_scope = _decode(
        value=shapes,
        index=0,
        in_channels=params.num_atoms,
        out_channels=params.num_points,
        variable_scope=variable_scope)
    shapes = tf.reshape(shapes, [-1, params.num_points, 3])
    return shapes


def estimate_coefficients_and_cameras(representations, params):
    """Estimates coefficients and cameras from hidden representations.

    Args:
        representations: A tensor. In shape
            [batch_size, BLOCK_HEIGHT, BLOCK_WIDTH, num_atoms].
    Returns:
        A tuple of (coefficients, cameras), where coefficients is a tensor, in
        shape [batch_size, 1, 1, num_atoms], cameras is a tensor, in shape
        [batch_size, BLOCK_HEIGHT, BLOCK_WIDTH].
    """
    with tf.variable_scope("Estimate_coefficients_and_camera"):
        fs = [BLOCK_HEIGHT,
              BLOCK_WIDTH,
              params.num_atoms_bottleneck,
              params.num_atoms_bottleneck]
        weights = tf.get_variable(
            name="weights_coef",
            shape=[BLOCK_HEIGHT,
                   BLOCK_WIDTH,
                   params.num_atoms_bottleneck,
                   params.num_atoms_bottleneck],
            dtype=tf.float32)
        bias = tf.get_variable(
            name="bias_coeff",
            shape=[params.num_atoms_bottleneck],
            dtype=tf.float32)
        coefficients = tf.nn.conv2d(
            input=representations,
            filter=weights,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="conv2d_coeff")

        weights = tf.get_variable(
            name="weights_camera",
            shape=[1, 1, params.num_atoms_bottleneck, 1],
            dtype=tf.float32)
        bias = tf.get_variable(name="bias_camera", shape=[1], dtype=tf.float32)
        cameras = tf.nn.conv2d(
            input=representations,
            filter=weights,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="conv2d_camera")
        cameras = tf.reshape(cameras, [-1, BLOCK_HEIGHT, BLOCK_WIDTH])
        _, U, V = tf.svd(cameras)
        cameras = tf.matmul(U, V, transpose_b=True)
        return coefficients, cameras


def model_fn(features, labels, mode, params):
    """The model function defining the architecture of NN."""
    with tf.device("/device:GPU:0"):
        with tf.variable_scope("Measurements"):
            measurements = features["points2d"]
            measurements_transpose = tf.transpose(measurements, [0, 2, 1])
            measurements_reshape = tf.reshape(
                measurements_transpose,
                [-1, 1, BLOCK_WIDTH, params.num_points])

        # Estimates representations
        representations = encode(measurements_reshape, params)

        # Estimates shapes and cameras
        coefficients, cameras = estimate_coefficients_and_cameras(
            representations, params)

        # Estimates shapes
        shapes = decode(coefficients, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {}
            predictions["representation"] = representations
            predictions["coefficient"] = coefficients
            predictions["point3d"] = shapes
            predictions["camera"] = cameras
            predictions["measurement"] = features["points2d"]
            predictions["ground_truth"] = features["points3d"]
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        loss = losses_mean_error(shapes @ cameras, measurements, params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                params.learning_rate,
                tf.train.get_global_step(),
                params.learning_rate_decay_steps,
                params.learning_rate_decay_rate,
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              train_op=train_op)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
