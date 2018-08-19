# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import tensorflow as tf
from tensorflow import logging as log
import time
import os


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )
    return config


def load_ckpt(saver, sess, log_root, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir)
    and restore it to saver and sess, waiting 10 secs in the case of failure.
    Also returns checkpoint name.
    """
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            log.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            log.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)
