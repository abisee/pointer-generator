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

"""This is the top-level file to train, evaluate or test your summarization model"""

import time
import os
import argparse
import tensorflow as tf
from tensorflow import logging as log
import numpy as np
from trainer.data import Vocab
from trainer.batcher import Batcher
from trainer.model import SummarizationModel
from trainer.decode import BeamSearchDecoder
import trainer.util as util
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.model_fn import ModeKeys as Modes


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
      loss: loss on the most recent eval step
      running_avg_loss: running_avg_loss so far
      summary_writer: FileWriter object to write for tensorboard
      step: training iteration step
      decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
      running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % decay
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    log.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def __restore_best_model(log_root, conf):
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    log.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=conf.session_config)
    log.info("Initializing all variables...")
    sess.run(tf.global_variables_initializer())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.global_variables() if "Adagrad" not in v.name])
    log.info("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, log_root=log_root, ckpt_dir="eval")
    log.info("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(log_root, "train", new_model_name)
    log.info("Saving model to %s..." % new_fname)
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    log.info("Saved.")
    exit()


def __convert_to_coverage_model(log_root, conf):
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    log.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=conf.session_config)
    log.info("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    log.info("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess, log_root=log_root)
    log.info("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    log.info("saving model to %s..." % new_fname)
    new_saver = tf.train.Saver()  # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    log.info("saved.")
    exit()


def setup_training(
        model,
        batcher,
        log_root,
        convert_to_coverage_model,
        coverage,
        restore_best_model,
        debug,
        max_step,
        conf
):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    model.build_graph()  # build the graph
    if convert_to_coverage_model:
        assert coverage, """\
        To convert your non-coverage model to a coverage model, 
        run with convert_to_coverage_model=True and coverage=True\
        """
        __convert_to_coverage_model(log_root, conf=conf)
    if restore_best_model:
        __restore_best_model(log_root, conf=conf)
    try:
        # this is an infinite loop until interrupted
        run_training(model, batcher, train_dir,
                     coverage=coverage, debug=debug, max_step=max_step, conf=conf)
    except KeyboardInterrupt:
        log.info("Caught keyboard interrupt on worker. Stopping...")


def __train_session(train_dir, debug, conf):
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,  # required to restore variables!
        summary_dir=train_dir,
        is_chief=conf.is_chief,
        save_summaries_secs=60,
        save_checkpoint_secs=60,
        max_wait_secs=60,
        stop_grace_period_secs=60,
        config=conf.session_config,
        scaffold=tf.train.Scaffold(
            saver=tf.train.Saver(max_to_keep=3)
        )
    )
    if debug:  # start the tensorflow debugger
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    return sess


def run_training(model, batcher, train_dir, coverage, debug, max_step, conf):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    log.debug("starting run_training")
    summary_writer = tf.summary.FileWriterCache.get(train_dir)
    with __train_session(train_dir=train_dir, debug=debug, conf=conf) as sess:
        train_step = 0
        # repeats until max_step is reached
        while not sess.should_stop() and train_step <= max_step:
            log.debug('running training step...')
            batch = batcher.next_batch()
            t0 = time.time()
            results = model.run_train_step(sess, batch)
            t1 = time.time()
            loss = results['loss']
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            train_step = results['global_step']  # we need this to update our running average loss
            log.info('step=%i, loss=%f, elapsed_secs=%.0f', train_step, loss, t1 - t0)
            if coverage:
                coverage_loss = results['coverage_loss']
                log.info("coverage_loss: %f", coverage_loss)  # print the coverage loss to screen
            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries = results['summaries']  # we will write these summaries to tensorboard using summary_writer
            summary_writer.add_summary(summaries, train_step)  # write the summaries


def run_eval(model, batcher, log_root, coverage, conf):
    """
    Repeatedly runs eval iterations, logging to screen and writing summaries.
    Saves the model with the best loss seen so far.
    """
    model.build_graph()  # build the graph
    saver = tf.train.Saver(max_to_keep=3)  # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=conf.session_config)
    eval_dir = os.path.join(log_root, "eval")  # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    running_avg_loss = 0
    best_loss = None  # will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess, log_root=log_root)  # load a new checkpoint
        batch = batcher.next_batch()  # get the next batch

        # run eval on the batch
        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        log.info('seconds for batch: %.2f', t1 - t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        log.info('loss: %f', loss)
        if coverage:
            coverage_loss = results['coverage_loss']
            log.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            log.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss,
                     bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()


def __hparams(**hparams):
    """Make a HParams object, containing the values of the hyperparameters that the model needs

    :return:
        HParams object
    """
    hps = HParams(**hparams)
    return hps


def __log_verbosity(level):
    level = level.lower()
    if level == 'info':
        log.set_verbosity(log.INFO)
    elif level == 'warn':
        log.set_verbosity(log.WARN)
    elif level == 'error':
        log.set_verbosity(log.ERROR)
    else:
        log.set_verbosity(log.DEBUG)


def __log_root(base_path, exp_name, mode):
    """Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary"""
    res = os.path.join(base_path, exp_name)
    if not os.path.exists(res):
        if mode == "train":
            os.makedirs(res)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % res)
    return res


def __main(
        mode,
        data_path,
        vocab_path,
        vocab_size,
        log_root,
        beam_size,
        pointer_gen,
        convert_to_coverage_model,
        coverage,
        restore_best_model,
        log_level,
        exp_name,
        single_pass,
        random_seed,
        debug,
        **hparams
):
    __log_verbosity(log_level)
    log.info('Starting seq2seq_attention in %s mode...', mode)
    log_root = __log_root(log_root, exp_name, mode)
    vocab = Vocab(vocab_path, vocab_size)  # create a vocabulary
    hps = __hparams(**hparams)
    conf = util.run_config(model_dir=log_root, random_seed=random_seed)
    log.info(f'hps={repr(hps)}\nconf={util.repr_run_config(conf)}')

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(
        data_path=data_path,
        vocab=vocab,
        hps=hps,
        single_pass=single_pass,
        mode=mode,
        pointer_gen=pointer_gen
    )
    model = SummarizationModel(
        hps,
        vocab,
        mode=mode,
        pointer_gen=pointer_gen,
        coverage=coverage,
        log_root=log_root,
        conf=conf
    )
    if mode == Modes.TRAIN:
        setup_training(
            model,
            batcher,
            log_root=log_root,
            convert_to_coverage_model=convert_to_coverage_model,
            coverage=coverage,
            restore_best_model=restore_best_model,
            debug=debug,
            max_step=hps.max_step,
            conf=conf
        )
    elif mode == Modes.EVAL:
        run_eval(model, batcher, log_root=log_root, coverage=coverage, conf=conf)
    elif mode == Modes.PREDICT:
        decoder = BeamSearchDecoder(model, batcher, vocab,
                                    hps=hps,
                                    single_pass=single_pass,
                                    log_root=log_root,
                                    pointer_gen=pointer_gen,
                                    data_path=data_path,
                                    beam_size=beam_size,
                                    conf=conf
                                    )
        # decode indefinitely
        # (unless single_pass=True, in which case deocde the dataset exactly once)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
    parser.add_argument(
        '--vocab_path',
        type=str,
        required=True,
        help='Path expression to text vocabulary file.')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help='must be one of train/eval/decode')
    parser.add_argument(
        '--single_pass',
        type=bool,
        default=False,
        help="""\
        For decode mode only.
        If True, run eval on the full dataset using a fixed checkpoint, 
        i.e. take the current checkpoint, and use it to produce one summary
        for each example in the dataset, write the summaries to file and then get ROUGE scores
        for the whole dataset. If False (default), run concurrent decoding,
        i.e. repeatedly load latest checkpoint,
        use it to produce summaries for randomly-chosen examples
        and log the results to screen, indefinitely.\
        """)
    parser.add_argument(
        '--log_root',
        type=str,
        required=True,
        help='Root directory for all logging.')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='default_exp_name',
        help='Name for experiment. Logs will be saved in a directory with this name, under log_root.')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='dimension of RNN hidden states')
    parser.add_argument(
        '--emb_dim',
        type=int,
        default=128,
        help='dimension of word embeddings')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='minibatch size')
    parser.add_argument(
        '--max_step',
        type=int,
        default=30,
        help='model will be trained for maximum number of steps')
    parser.add_argument(
        '--max_enc_steps',
        type=int,
        default=400,
        help='max timesteps of encoder (max source text tokens)')
    parser.add_argument(
        '--max_dec_steps',
        type=int,
        default=100,
        help='max timesteps of decoder (max summary tokens)')
    parser.add_argument(
        '--beam_size',
        type=int,
        default=4,
        help='beam size for beam search decoding.')
    parser.add_argument(
        '--min_dec_steps',
        type=int,
        default=35,
        help="""\
        Minimum sequence length of generated summary. Applies only for beam search decoding mode.\
        """)
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50000,
        help="""\
        Size of vocabulary. These will be read from the vocabulary file in order. 
        If the vocabulary file contains fewer words than this number, or if this number is set to 0, 
        will take all words in the vocabulary file.\
        """)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.15,
        help='learning rate')
    parser.add_argument(
        '--adagrad_init_acc',
        type=float,
        default=0.1,
        help='initial accumulator value for Adagrad')
    parser.add_argument(
        '--rand_unif_init_mag',
        type=float,
        default=0.02,
        help='magnitude for lstm cells random uniform inititalization')
    parser.add_argument(
        '--trunc_norm_init_std',
        type=float,
        default=1e-4,
        help='std of trunc norm init, used for initializing everything else')
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=2.0,
        help='for gradient clipping')
    parser.add_argument(
        '--pointer_gen',
        type=bool,
        default=True,
        help='If True, use pointer-generator model. If False, use baseline model.')
    parser.add_argument(
        '--coverage',
        type=bool,
        default=False,
        help="""\
        Use coverage mechanism.
        Note, the experiments reported in the ACL paper train WITHOUT coverage until converged,
        and then train for a short phase WITH coverage afterwards.
        i.e. to reproduce the results in the ACL paper,
        turn this off for most of training then turn on for a short phase at the end.\
        """)
    parser.add_argument(
        '--cov_loss_wt',
        type=float,
        default=1.0,
        help="""\
        Weight of coverage loss (lambda in the paper). 
        If zero, then no incentive to minimize coverage loss.\
        """)
    parser.add_argument(
        '--convert_to_coverage_model',
        type=bool,
        default=False,
        help="""\
        Convert a non-coverage model to a coverage model.
        Turn this on and run in train mode.
        Your current training model will be copied to a new version (same name with _cov_init appended)
        that will be ready to run with coverage flag turned on, for the coverage training stage.\
        """)
    parser.add_argument(
        '--restore_best_model',
        type=bool,
        default=False,
        help="""\
        Restore the best model in the eval/ dir and save it in the train/ dir,
        ready to be used for further training. Useful for early stopping, 
        or if your training checkpoint has become corrupted with e.g. NaN values.\
        """)
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='Run in tensorflow debug mode (watches for NaN/inf values)')
    parser.add_argument(
        '--log_level',
        type=str,
        default='info',
        help='tensorflow logging verbosity level (pick one): debug/info/warn/error')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=111,
        help='Random seed integer')
    args = parser.parse_args()
    modes = [Modes.TRAIN, Modes.EVAL, Modes.PREDICT]
    if args.mode not in modes:
        raise ValueError(f'--mode flag must be one of {repr(modes)}')
    if args.single_pass and args.mode != Modes.PREDICT:
        raise ValueError(f'--single_pass flag should only be True in {repr(Modes.PREDICT)} mode')
    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam,
    # so we need to make a batch of these hypotheses.
    if args.mode == Modes.PREDICT:
        args.batch_size = args.beam_size
    __main(**vars(args))
