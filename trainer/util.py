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
import json


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


def __tf_config_json():
    conf = os.environ.get('TF_CONFIG')
    if not conf:
        return None
    return json.loads(conf)


def __tf_config():  # TODO remove
    """Parse TF_CONFIG to cluster_spec
          TF_CONFIG environment variable is available when running using
          gcloud either locally or on cloud. It has all the information required
          to create a ClusterSpec which is important for running distributed code.
    """
    config_json = __tf_config_json()
    res = {
        'cluster_spec': None,
        'server': None,
        'is_chief': True
    }
    if config_json is None:
        return res
    cluster = config_json.get('cluster')
    job_name = config_json.get('task', {}).get('type')
    task_index = config_json.get('task', {}).get('index')
    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return res
    res['cluster_spec'] = tf.train.ClusterSpec(cluster)
    res['server'] = tf.train.Server(res['cluster_spec'],
                                    job_name=job_name,
                                    task_index=task_index)
    res['is_chief'] = (job_name == 'master')
    return res


def __session_config():
    """Returns a tf.ConfigProto instance that has appropriate device_filters set.
    """
    device_filters = None
    tf_config_json = __tf_config_json()
    if (tf_config_json and
            'task' in tf_config_json and
            'type' in tf_config_json['task'] and
            'index' in tf_config_json['task']):
        # Master should only communicate with itself and ps
        if tf_config_json['task']['type'] == 'master':
            device_filters = ['/job:ps', '/job:master']
        # Worker should only communicate with itself and ps
        elif tf_config_json['task']['type'] == 'worker':
            device_filters = [
                '/job:ps',
                '/job:worker/task:%d' % tf_config_json['task']['index']
            ]
    return tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=device_filters,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )


def run_config(model_dir, random_seed):
    return tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=__session_config(),
        save_checkpoints_secs=60,
        save_summary_steps=100,
        keep_checkpoint_max=3,
        tf_random_seed=random_seed
    )


def repr_run_config(conf):
    assert isinstance(conf, tf.estimator.RunConfig)
    return f"""tf.estimator.RunConfig(
        model_dir={repr(conf.model_dir)},
        cluster_spec={repr(conf.cluster_spec.as_dict())}, 
        is_chief={repr(conf.is_chief)}, 
        master={repr(conf.master)}, 
        num_worker_replicas={repr(conf.num_worker_replicas)}, 
        num_ps_replicas={repr(conf.num_ps_replicas)}, 
        task_id={repr(conf.task_id)}, 
        task_type={repr(conf.task_type)},
        tf_random_seed={repr(conf.tf_random_seed)},
        save_summary_steps={repr(conf.save_summary_steps)},
        save_checkpoints_steps={repr(conf.save_checkpoints_steps)},
        save_checkpoints_secs={repr(conf.save_checkpoints_secs)},
        session_config={repr(conf.session_config)},
        keep_checkpoint_max={repr(conf.keep_checkpoint_max)},
        keep_checkpoint_every_n_hours={repr(conf.keep_checkpoint_every_n_hours)},
        log_step_count_steps={repr(conf.log_step_count_steps)},
        train_distribute={repr(conf.train_distribute)},
        device_fn={repr(conf.device_fn)}
    )
    """
