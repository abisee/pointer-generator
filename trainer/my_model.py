import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes


def generate_model_fn():
    def _model_fn(features, labels, mode, params, config):
        modes = [Modes.TRAIN, Modes.EVAL, Modes.PREDICT]
        if mode not in modes:
            raise ValueError(f'mode must be one of {repr(modes)} but mode={mode}')
        if mode == Modes.TRAIN:
            loss = 0  # TODO filler
            grads = [0]  # TODO filler
            global_step = 0  # TODO filler
            tvars = tf.trainable_variables()
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=params.lr,
                initial_accumulator_value=params.adagrad_init_acc
            )
            train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=global_step,
                name='train_op'
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    return _model_fn
