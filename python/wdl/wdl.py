import tensorflow as tf
from wdl_model import WDL
import wdl_input as input
import os
from utils.flags import core as flags_core
from utils.logs import logger
from absl import flags
from wide_deep import census_dataset

from wide_deep import wide_deep_run_loop

def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='../data',
                          model_dir='../model',
                          train_epochs=40,
                          epochs_between_evals=2,
                          batch_size=40)


tf.logging.set_verbosity(tf.logging.INFO)
define_census_flags()
epochs_between_evals = 2
with logger.benchmark_context(flags.FLAGS):
    flags_obj = flags.FLAGS
    global_step = tf.Variable(initial_value=0,
                              name="global_step",
                              trainable=False,
                              collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    optimizer = tf.train.AdagradOptimizer(0.02)

    train_file = os.path.join('../data', census_dataset.TRAINING_FILE)
    test_file = os.path.join('../data', census_dataset.EVAL_FILE)

    dataset = census_dataset.input_fn(train_file, epochs_between_evals, False, 1)
    features, labels = dataset.make_one_shot_iterator().get_next()

    wide, deep = input.build_model_columns()
    print("type wide:%s" % type(wide))
    print("type deep:%s" % type(deep))
    feature_columns = {"deep": deep, "wide": wide}

    wdl = WDL()
    logits = wdl.inference(features, feature_columns)

    loss = wdl.loss(logits, features["labels"])

    train_op = optimizer.minimize(loss, global_step)

    with tf.Session() as session:
        while True:
            print(session.run(loss))
            session.run(train_op)