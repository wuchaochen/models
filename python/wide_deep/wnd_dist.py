# Distributed MNIST on grid based on TensorFlow MNIST example

from datetime import datetime
import tensorflow as tf
from tensorflow.python.summary.writer.writer_cache import FileWriterCache as SummaryWriterCache
import math
import numpy
import json
import sys
from utils.logs import logger
from wide_deep import census_dataset
from wide_deep import wide_deep_run_loop
import os
from absl import flags
from utils.flags import core as flags_core


def export_saved_model(sess, export_dir, tag_set, signatures):
    g = sess.graph
    g._unsafe_unfinalize()
    try:
        tf.gfile.DeleteRecursively(export_dir)
    except tf.errors.OpError:
        pass
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    signature_def_map = {}
    for key, sig in signatures.items():
        signature_def_map[key] = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={name: tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in
                    sig['inputs'].items()},
            outputs={name: tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in
                     sig['outputs'].items()},
            method_name=sig['method_name'] if 'method_name' in sig else key)

        builder.add_meta_graph_and_variables(
            sess,
            tag_set.split(','),
            signature_def_map=signature_def_map,
            clear_devices=True)

        g.finalize()
        builder.save()


class ExportHook(tf.train.SessionRunHook):
    def __init__(self, export_dir, input_tensor, output_tensor):
        self.export_dir = export_dir
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def end(self, session):
        print("{} ======= Exporting to: {}".format(datetime.now().isoformat(), self.export_dir))
        signatures = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
                'inputs': {'image': self.input_tensor},
                'outputs': {'prediction': self.output_tensor},
                'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            }
        }
        export_saved_model(session, self.export_dir, tf.saved_model.tag_constants.SERVING,
                           signatures)
        print("{} ======= Done exporting".format(datetime.now().isoformat()))


def decode(serialized_example):
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
               'label': tf.VarLenFeature(tf.int64)}
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label'].values
    return image, label


def input_iter(filename, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(decode)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator



def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='../data',
                          model_dir='../model',
                          train_epochs=40,
                          epochs_between_evals=2,
                          batch_size=40)


def build_estimator(model_dir, model_type, model_column_fn):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def run_census(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  if flags_obj.download_if_missing:
    census_dataset.download(flags_obj.data_dir)

  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return census_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="Census Income", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=census_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)


def map_fn(context):
    tf.logging.set_verbosity(tf.logging.INFO)
    define_census_flags()
    with logger.benchmark_context(flags.FLAGS):
        run_census(flags.FLAGS)


def map_fun(context):
    job_name = context.jobName
    task_index = context.index

    props = context.properties
    batch_size = int(props.get("batch_size"))
    input_dir = props.get("input")
    epochs = int(props.get("epochs"))
    checkpoint_dir = props.get("checkpoint_dir")
    export_dir = props.get("export_dir")
    print ("input:" + input_dir)
    print ("checkpoint_dir:" + checkpoint_dir)
    print ("export_dir:" + export_dir)
    sys.stdout.flush()

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    # if job_name == "ps":
    #   time.sleep((num_worker + 1) * 5)

    # Parameters
    IMAGE_PIXELS = 28
    hidden_units = 128

    clusterStr = props.get("cluster")
    clusterJson = json.loads(clusterStr)
    cluster = tf.train.ClusterSpec(cluster=clusterJson)
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    def feed_dict(images, labels):
        xs = numpy.array(images)
        xs = xs.astype(numpy.float32)
        xs = xs / 255.0
        ys = numpy.array(labels)
        ys = ys.astype(numpy.uint8)
        return (xs, ys)

    if job_name == "ps":
        from time import sleep
        while True:
            sleep(1)
        # sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
        #                              device_filters=["/job:ps/task:%d" % task_index])
        # sess = tf.Session(target=server.target, config=sess_config)
        # queue = create_done_queue(num_worker, task_index)
        # for i in range(num_worker):
        #     finished = sess.run(queue.dequeue())
        #     print("ps %d received done from worker %d" % (task_index, finished))
        # sess.close()
        # print("ps %d: quitting" % (task_index))
    elif job_name == "worker":

        # # Assigns ops to the local worker by default.
        # with tf.device(
        #         tf.train.replica_device_setter(worker_device="/job:worker/task:" + str(task_index), cluster=cluster)):
        #
        #     # Placeholders or QueueRunner/Readers for input data
        #     x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
        #     y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
        #
        #     # Variables of the hidden layer
        #     hid_w = tf.Variable(
        #         tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units], stddev=1.0 / IMAGE_PIXELS),
        #         name="hid_w")
        #     hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
        #     hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        #     hid = tf.nn.relu(hid_lin)
        #
        #     # Variables of the softmax layer
        #     sm_w = tf.Variable(
        #         tf.truncated_normal([hidden_units, 10], stddev=1.0 / math.sqrt(hidden_units)),
        #         name="sm_w")
        #     sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
        #     y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        #
        #     global_step = tf.train.get_or_create_global_step()
        #
        #     loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        #
        #     train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
        #
        #     # Test trained model
        #     label = tf.argmax(y_, 1, name="label")
        #     prediction = tf.argmax(y, 1, name="prediction")
        #     correct_prediction = tf.equal(prediction, label)
        #
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        #
        #     iter = input_iter(input_dir + "/" + str(task_index) + ".tfrecords", batch_size, epochs)
        #     next_batch = iter.get_next()
        #
        #     is_chief = (task_index == 0)
        #     sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
        #                                  device_filters=["/job:ps", "/job:worker/task:%d" % task_index])
        #
        #     # The MonitoredTrainingSession takes care of session initialization, restoring from
        #     #  a checkpoint, and closing when done or an error occurs
        #     mon_sess = tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,
        #                                                  checkpoint_dir=checkpoint_dir,
        #                                                  stop_grace_period_secs=10, max_wait_secs=300,
        #                                                  config=sess_config,
        #                                                  chief_only_hooks=[ExportHook(export_dir, x, prediction)])
        #     processed = 0
        #     while not mon_sess.should_stop():
        #         # Run a training step asynchronously
        #         # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        #         # perform *synchronous* training.
        #         try:
        #             images, labels = mon_sess.run(next_batch)
        #             processed += images.shape[0]
        #         except tf.errors.OutOfRangeError:
        #             break
        #
        #         batch_xs, batch_ys = feed_dict(images, labels)
        #         feed = {x: batch_xs, y_: batch_ys}
        #
        #         if len(batch_xs) > 0 and not mon_sess.should_stop():
        #             _, step = mon_sess.run([train_op, global_step], feed_dict=feed)
        #             if (step % 100 == 0):
        #                 print("{0}, Task {1} step: {2} accuracy: {3}".format(
        #                     datetime.now().isoformat(), task_index, step,
        #                     mon_sess.run(accuracy, {x: batch_xs, y_: batch_ys})))
        #                 sys.stdout.flush()
        #
        #     print(str(processed) + " records processed.")
        #     print("{0} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))
        #     mon_sess.close()

    SummaryWriterCache.clear()


if __name__ == "__main__":
    map_fun(context)
