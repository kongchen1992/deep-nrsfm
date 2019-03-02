"""Evaluates the neural network for solving NRSfM problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow_models
import motion_capture

BLOCK_WIDTH = 2
BLOCK_HEIGHT = 3


def calibrate_by_procrustes(points3d, camera, gt):
    """Calibrates the predictied 3d points by Procrustes algorithm.

    This function estimate an orthonormal matrix for aligning the predicted 3d
    points to the ground truth. This orhtonormal matrix is computed by
    Procrustes algorithm, which ensures the global optimality of the solution.
    """
    # Shift the center of points3d to the origin
    if camera is not None:
        singular_value = np.linalg.norm(camera, 2)
        camera = camera / singular_value
        points3d = points3d * singular_value
    scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
    points3d = points3d * scale
    U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
    rot = U.dot(Vh)
    if camera is not None:
        return points3d.dot(rot), rot.T.dot(camera)
    else:
        return points3d.dot(rot), None


def evaluate_reconstruction(predictions, params, error_dir, result_dir):
    projections = []
    shapes = []
    cameras = []
    shapes_gt = []

    metrics = {
        "error_2d": [],
        "error_3d": [],
        "error_abs": []}

    for pred in predictions:
        points2d_gt = pred["measurement"]
        points3d_gt = pred["ground_truth"]
        points3d, camera = calibrate_by_procrustes(
            pred["point3d"], pred["camera"], points3d_gt)
        points2d = points3d.dot(camera)

        # Evaluates error metrics
        metrics["error_2d"].append(np.linalg.norm(
            points2d - points2d_gt) / np.linalg.norm(points2d_gt))
        metrics["error_3d"].append(np.linalg.norm(
            points3d - points3d_gt) / np.linalg.norm(points3d_gt))
        metrics["error_abs"].append(np.linalg.norm(
            points3d - points3d_gt, axis=1).mean())

        # Collects predictions
        projections.append(points2d_gt)
        shapes_gt.append(points3d_gt)
        shapes.append(points3d)
        cameras.append(camera)

    dataframe = pd.DataFrame(metrics)
    dataframe.to_csv(error_dir)

    np.savez(result_dir,
             shapes=shapes,
             cameras=cameras,
             groundtruth=shapes_gt,
             projections=projections)


def main(_):
    # Prepares hyper parameters
    hparams = tensorflow_models.hparams()
    hparams.parse(FLAGS.hparams)

    # Specify a checkpoint to use.
    checkpoint_path = os.path.join(
        FLAGS.model_dir, "model.ckpt-" + FLAGS.checkpoint)

    # Creates the estimator
    estimator = tf.estimator.Estimator(
        model_fn=tensorflow_models.model_fn,
        model_dir=FLAGS.model_dir,
        params=hparams)

    # Prepares data for evaluation
    filename = os.path.join(
        motion_capture.path["tfrecords"], "{:02d}.train".format(FLAGS.subject))
    eval_input_fn = functools.partial(
        motion_capture.eval_input_fn,
        filename=filename,
        batch_size=hparams.batch_size)

    # Predicts
    predictions = estimator.predict(
        input_fn=eval_input_fn, checkpoint_path=checkpoint_path)

    # Evaluates
    evaluate_reconstruction(
        predictions,
        hparams,
        FLAGS.error_metrics_dir,
        FLAGS.predictions_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams",
        type=str,
        default="",
        help="Comma separated list of name=value pairs.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="The gpu used for training.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="The directory to save models.")
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="The subject in CMU MoCap.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint to use.")
    parser.add_argument(
        "--error_metrics_dir",
        type=str,
        default=None,
        help="The directory where the error metrics csv is saved.")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="The directory where the predted results is saved.")
    FLAGS, unparsed = parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
