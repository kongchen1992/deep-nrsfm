import os
from scipy import io as sio
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go


NUM_POINTS = 31
METER_SCALER = 0.001
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (0, 6), (6, 7), (7, 8),
               (8, 9), (8, 10), (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
               (15, 16), (13, 24), (24, 25), (25, 26), (26, 27), (27, 30),
               (27, 28), (27, 29), (13, 17), (17, 18), (18, 19), (19, 20),
               (20, 21), (20, 22), (20, 23))
BLUE = "rgb(90, 130, 238)"
RED = "rgb(205, 90, 76)"

path = {"tfrecords": "/dataset/chenk/cmu-mocap/tfrecords/"}


def _parse_function(example_proto):
    """Parses raw bytes into tensors."""
    features = {
        "points3d_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "points2d_raw": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    output_features = {
        "points3d": tf.reshape(
            tf.decode_raw(parsed_features["points3d_raw"], tf.float32),
            [NUM_POINTS, 3],
        ),
        "points2d": tf.reshape(
            tf.decode_raw(parsed_features["points2d_raw"], tf.float32),
            [NUM_POINTS, 2],
        ),
    }
    # Returns a tuple (features, labels)
    return output_features, 0


def train_input_fn(filename, buffer_size, batch_size):
    """An input function for training."""
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(filename, batch_size):
    """An input function for evaluation."""
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def get_trace3d(points3d, point_color=None, line_color=None, name="PointCloud"):
    """Yields plotly traces for visualization."""
    if point_color is None:
        point_color = "rgb(30, 20, 160)"
    if line_color is None:
        line_color = "rgb(30, 20, 160)"
    # Trace of points.
    trace_of_points = go.Scatter3d(
        x=points3d[:, 0],
        y=points3d[:, 2],
        z=points3d[:, 1],
        mode="markers",
        name=name,
        marker=dict(
            symbol="circle",
            size=3,
            color=point_color))

    # Trace of lines.
    xlines = []
    ylines = []
    zlines = []
    for line in CONNECTIONS:
        for point in line:
            xlines.append(points3d[point, 0])
            ylines.append(points3d[point, 2])
            zlines.append(points3d[point, 1])
        xlines.append(None)
        ylines.append(None)
        zlines.append(None)
    trace_of_lines = go.Scatter3d(
        x=xlines,
        y=ylines,
        z=zlines,
        mode="lines",
        name=name,
        line=dict(color=line_color))
    return [trace_of_points, trace_of_lines]


def get_figure3d(points3d, gt=None, range_scale=1):
    """Yields plotly fig for visualization"""
    traces = get_trace3d(points3d, BLUE, BLUE, "prediction")
    if gt is not None:
        traces += get_trace3d(gt, RED, RED, "groundtruth")
    layout = go.Layout(
        scene=dict(
            aspectratio=dict(x=0.8,
                             y=0.8,
                             z=2),
            xaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale),),
            yaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale),),
            zaxis=dict(range=(-1 * range_scale, 1 * range_scale),),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    return go.Figure(data=traces, layout=layout)
