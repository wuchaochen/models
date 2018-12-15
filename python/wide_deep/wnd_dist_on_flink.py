"""Train WDL on census income dataset."""

import os
import tensorflow as tf
import json
import sys
import numbers

TRAINING_FILE = 'adult.data'
EVAL_FILE = 'adult.test'

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def cluster_to_estimator(cluster_str):
    cluster = json.loads(cluster_str)
    worker_0 = cluster['worker'][0]
    del (cluster['worker'][0])
    cluster['chief'] = [worker_0]
    return cluster


def export_cluster_env(cluster_str, job_name, index):
    cluster = cluster_to_estimator(cluster_str)
    if 'ps' == job_name:
        task_type = 'ps'
        task_index = index
    elif 'worker' == job_name:
        if 0 == index:
            task_type = 'chief'
            task_index = 0
        else:
            task_type = 'worker'
            task_index = index - 1

    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': task_type, 'index': task_index}})
    print (os.environ['TF_CONFIG'])
    return cluster, task_type, task_index


def build_run_config(cluster_str, job_name, task_id, model_dir):
    """Build run_config for every role: worker or ps."""
    cluster_json, task_type, task_index = export_cluster_env(cluster_str, job_name, task_id)
    run_config = tf.estimator.RunConfig(model_dir=model_dir)  # save if not zero
    config = tf.ConfigProto(log_device_placement=True)
    cluster = tf.train.ClusterSpec(cluster=cluster_json)
    server = tf.train.Server(cluster, job_name=task_type, task_index=int(task_index), config=config)
    if job_name == "ps":
        server.join()
    return run_config


def build_estimator(context, model_dir, model_type, model_column_fn):
    """Build an estimator appropriate for the given model type.

    Args:
        context: Object containing params.
        model_dir: the folder for saving model
        model_type: enum of wide, deep, wide_deep
        model_column_fn: build model columns
    """
    wide_columns, deep_columns = model_column_fn()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    job_name = context.jobName
    task_index = context.index
    props = context.properties
    run_config = build_run_config(cluster_str=props["cluster"], job_name=job_name, task_id=task_index,
                                  model_dir=model_dir)

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


def map_func(context):
    """Construct all necessary functions and call run_loop.

    Args:
        context: Object containing params.
    """

    job_name = context.jobName
    task_index = context.index
    props = context.properties
    batch_size = 40
    input_dir = props.get("input")
    epochs = int(props.get("epochs"))
    checkpoint_dir = props.get("checkpoint_dir")
    export_dir = props.get("export_dir")
    data_dir = input_dir
    epochs_between_evals = 1

    train_file = os.path.join(data_dir, TRAINING_FILE)
    test_file = os.path.join(data_dir, EVAL_FILE)

    # Train and evaluate the model every `epochs_between_evals` epochs.
    def train_input_fn():
        return input_fn(
            train_file, epochs_between_evals, True, batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, batch_size)

    run_loop(
        context=context,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        model_column_fn=build_model_columns,
        build_estimator_fn=build_estimator,
        early_stop=True)


def run_loop(context, train_input_fn, eval_input_fn, model_column_fn,
             build_estimator_fn, early_stop=False):
    """Define training loop.

    Args:
        context: Object containing params.
        train_input_fn: input_fn for train data
        eval_input_fn: input_fn for evaluate data
        model_column_fn: build model columns
        build_estimator_fn: build estimator
    """

    apply_clean(context.properties["checkpoint_dir"])
    model_dir = context.properties["checkpoint_dir"]
    model_type = 'wide_deep'
    train_epochs = 40
    epochs_between_evals = 1
    stop_threshold = 0.8

    model = build_estimator_fn(
        model_dir=model_dir, model_type=model_type,
        model_column_fn=model_column_fn, context=context)

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    for n in range(train_epochs // epochs_between_evals):
        model.train(input_fn=train_input_fn, hooks=None)

        results = model.evaluate(input_fn=eval_input_fn)

        # Display evaluation metrics
        tf.logging.info('Results at epoch %d / %d',
                        (n + 1) * epochs_between_evals,
                        train_epochs)
        tf.logging.info('-' * 60)

        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))

        if early_stop and past_stop_threshold(
            stop_threshold, results['accuracy']):
            break


def past_stop_threshold(stop_threshold, eval_metric):
    """return True when stop_threshold <= eval_metric"""
    if stop_threshold is None:
        return False

    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions "
                         "must be a number.")

    if eval_metric >= stop_threshold:
        tf.logging.info(
            "Stop threshold of {} was passed with metric value {}.".format(
                stop_threshold, eval_metric))
        return True

    return False


def apply_clean(model_dir):
    """clean existing model dir when clean is True and model_dir exists"""
    clean = True
    if clean and tf.gfile.Exists(model_dir):
        tf.logging.info("--clean flag set. Removing existing model dir: {}".format(model_dir))
        tf.gfile.DeleteRecursively(model_dir)


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'],
            hash_bucket_size=_HASH_BUCKET_SIZE),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have run census_dataset.py and '
        'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        classes = tf.equal(labels, '>50K')  # binary classification
        return features, classes

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    print("sys:%s" % (sys.argv[1]))
    if sys.argv[1] == "0":
        job_name = "chief"
        task_id = 0
    elif sys.argv[1] == "1":
        job_name = "worker"
        task_id = 0
    else:
        job_name = "ps"
        task_id = 0
    print("type:%s, task_id:%s" % (job_name, task_id))
    context = {}
    context['type'] = job_name
    context['task_id'] = task_id
    context['model_dir'] = '../model'
    context['model_type'] = 'wide_deep'
    context['train_epochs'] = 40
    context['stop_threshold'] = None
    context['epochs_between_evals'] = 2
    context['batch_size'] = 5
    context['data_dir'] = '../data'
    context['clean'] = True
    context['cluster'] = {'chief': ['localhost:2222'], 'ps': ['localhost:4444'], 'worker': ['localhost:2223']}
    map_func(context)
