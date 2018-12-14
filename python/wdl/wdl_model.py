
import tensorflow as tf
import utils

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import partitioned_variables
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.layers.python.layers import initializers

class WDL(object):

    def get_ps_num(self):
        return 10

    def get_embedding_min_slice_size(self):
        return 4096

    def get_dnn_min_slice_size(self):
        return 4096

    def get_activation_conf(self):
        return "llrelu"

    def get_dnn_hidden_units(self):
        return [8, 8, 4]

    def is_need_dropout(self):
        return False

    def get_dropout_keep_prob(self):
        return 0.8

    def get_dnn_l2_weight_decay(self):
        return 0.00001

    def is_training(self):
        return True

    def model_arg_scope(self, weight_decay=0.0005, weights_initializer=initializers.xavier_initializer(),
                        biases_initializer=init_ops.zeros_initializer()):
        with arg_scope(
                [layers.fully_connected, layers.conv2d],
                weights_initializer=weights_initializer,
                weights_regularizer=regularizers.l2_regularizer(weight_decay),
                biases_initializer=init_ops.zeros_initializer()) as arg_sc:
            return arg_sc

    def getActivationFunctionOp(self, act_name="relu"):
        if type(act_name) != str and type(act_name) != unicode:
            return act_name
        if act_name.lower() == 'relu':
            return tf.nn.relu
        elif act_name.lower() == 'tanh':
            return tf.nn.tanh
        elif act_name.lower() == 'lrelu':
            return lambda x: tf.nn.leaky_relu(x, alpha=0.01)
        elif act_name.lower() == 'llrelu':
            return lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        else:
            return tf.nn.relu

    def inference(self, features, feature_columns):
        self.features = features
        self.deep_column = feature_columns.get("deep")
        self.wide_column = feature_columns.get("wide")

        if not (self.deep_column or self.wide_column):
            raise ValueError("deep or wide must be defined.")

        self.dnn_hidden_units = self.get_dnn_hidden_units()
        if not self.dnn_hidden_units:
            raise ValueError("configuration dnn_hidden_units must be defined.")

        variable_partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=self.get_ps_num(),
            min_slice_size=self.get_dnn_min_slice_size())

        if self.deep_column:
            with tf.variable_scope(name_or_scope="Deep",
                                   reuse=tf.AUTO_REUSE):

                with tf.variable_scope(name_or_scope="Deep_Embeddings",
                                       partitioner=variable_partitioner,
                                       reuse=tf.AUTO_REUSE) as embedding_scope:
                    self.deep_input_layer = layers.input_from_feature_columns(self.features, self.deep_column, scope=embedding_scope)

                with tf.variable_scope(name_or_scope="Deep_Network",
                                       partitioner=variable_partitioner):
                    self.deep_network = tf.concat([self.deep_input_layer], axis=1)
                    with arg_scope(self.model_arg_scope(weight_decay=self.get_dnn_l2_weight_decay())):
                        for layer_id, num_hidden_units in enumerate(self.dnn_hidden_units):
                            with variable_scope.variable_scope(
                                            "Dnn_HiddenLayer_%d" % layer_id) as dnn_hidden_layer_scope:
                                self.deep_network = layers.fully_connected(
                                    self.deep_network,
                                    num_hidden_units,
                                    self.getActivationFunctionOp(self.get_activation_conf()),
                                    scope=dnn_hidden_layer_scope,
                                    variables_collections=[self.collections_dnn_hidden_layer],
                                    outputs_collections=[self.collections_dnn_hidden_output],
                                    normalizer_fn=layers.batch_norm,
                                    normalizer_params={"scale": True,
                                                       "is_training": self.is_training()})
                                if self.is_need_dropout():
                                    self.deep_network = tf.layers.dropout(
                                        self.deep_network,
                                        rate=self.get_dropout_keep_prob(),
                                        noise_shape=None,
                                        seed=None,
                                        training=self.is_training(),
                                        name=None)

                with tf.variable_scope(name_or_scope="Deep_Logits",
                                       partitioner=variable_partitioner
                                       ) as dnn_logits_scope:
                    self.dnn_logits = layers.fully_connected(
                        self.deep_network,
                        1,
                        activation_fn=None,
                        variables_collections=[self.collections_dnn_hidden_layer],
                        outputs_collections=[self.collections_dnn_hidden_output],
                        scope=dnn_logits_scope,
                        normalizer_fn=layers.batch_norm,
                        normalizer_params={"scale": True,
                                           "is_training": self.is_training()})

        self.linear_logits = None
        if self.wide_column:
            with tf.variable_scope(
                    name_or_scope="Wide",
                    partitioner=variable_partitioner,
                    reuse=tf.AUTO_REUSE) as wide_scope:
                self.linear_logits, self.collections_linear_weights, self.linear_bias = \
                    layers.weighted_sum_from_feature_columns(
                        columns_to_tensors=features,
                        feature_columns=self.wide_column,
                        num_outputs=1,
                        weight_collections=[self.collections_wide_weights],
                        scope=wide_scope)

        if self.dnn_logits is not None and self.linear_logits is not None:
            self.logits = self.dnn_logits + self.linear_logits
        elif self.dnn_logits is not None:
            self.logits = self.dnn_logits
        else:
            self.logits = self.linear_logits
        return self.logits


    def loss(self, logits, labels):
        self.labels = labels
        with tf.name_scope("loss_op"):
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels)
            self.loss = tf.reduce_mean(self.batch_loss)
            self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_losses = tf.reduce_sum(self.reg_losses)
            self.loss = self.loss + self.reg_losses
        return self.loss