import tensorflow as tf
import wdl_model
import wdl_input as input

global_step = tf.Variable(initial_value=0,
                          name="global_step",
                          trainable=False,
                          collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

optimizer = tf.train.AdagradOptimizer(0.02)

features = input.input_fn()
deep, wide = input.build_model_columns
feature_columns ={"deep":deep, "wide":wide}

logits = wdl_model.inference(features, feature_columns)

loss = wdl_model.loss(logits, features["labels"])

train_op = optimizer.minimize(loss, global_step)

with tf.Session() as session:
    while True:
        print(session.run(loss))
        session.run(train_op)