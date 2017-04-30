"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(saver, sess):
  """Load checkpoint from the train directory and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      train_dir = os.path.join(FLAGS.log_root, "train")
      ckpt_state = tf.train.get_checkpoint_state(train_dir)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", train_dir, 10)
      time.sleep(10)
