"""Trains a neural network for solving NRSfM problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import sys

import tensorflow as tf
import numpy as np

import tensorflow_models
import motion_capture

BLOCK_WIDTH = 2
BLOCK_HEIGHT = 3


def main(_):
    hparams = tensorflow_models.hparams()
    hparams.parse(FLAGS.hparams)

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_summary_steps=FLAGS.save_summary_steps)

    estimator = tf.estimator.Estimator(
        model_fn=tensorflow_models.model_fn,
        model_dir=FLAGS.model_dir,
        config=config,
        params=hparams)

    filename = os.path.join(
        motion_capture.path["tfrecords"],
        "{:02d}.train".format(FLAGS.subject))

    train_input_fn = functools.partial(
        motion_capture.train_input_fn,
        filename=filename,
        buffer_size=FLAGS.buffer_size,
        batch_size=hparams.batch_size)

    eval_input_fn = functools.partial(
        motion_capture.eval_input_fn,
        filename=filename,
        batch_size=hparams.batch_size)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.max_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=FLAGS.throttle_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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
        "--save_checkpoints_steps",
        type=int,
        default=1000,
        help="Saves checkpoints every this many steps.")
    parser.add_argument(
        "--keep_checkpoint_max",
        type=int,
        default=0,
        help="Saves checkpoints every this many steps.")
    parser.add_argument(
        "--save_summary_steps",
        type=int,
        default=100,
        help="Saves summary every this many steps.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="The maximum training steps.")
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="The subject in CMU MoCap.")
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000,
        help="The buffer size used in training input function.")
    parser.add_argument(
        "--throttle_secs",
        type=int,
        default=60,
        help="Do not re-evaluate unless the last evaluation was started at "
        "least this many seconds ago.")
    FLAGS, unparsed = parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
