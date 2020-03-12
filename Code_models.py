#--------------------------     import package    --------------------------#

from Code_layers import *
from Code_metrics import *

#--------------------------      global variable     --------------------------#

flags = tf.app.flags
FLAGS = flags.FLAGS

#---------------------------     main function     ---------------------------#

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.emb = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.emb = self.activations[-2]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class MORE(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MORE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.motifinputs = placeholders['motiffeatures']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        # property_embedding layer
        for i in range(0, len(FLAGS.property_embedding_hidden)):
            if i == 0:
                # print(">> property_embedding Layer-{} dim: {} -> {}".format(i, self.input_dim, FLAGS.property_embedding_hidden[i]))
                self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.property_embedding_hidden[i],
                                                placeholders=self.placeholders,
                                                act=tf.nn.tanh,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))
            else:
                # print(">> property_embedding Layer-{} dim: {} -> {}".format(i, FLAGS.property_embedding_hidden[i-1], FLAGS.property_embedding_hidden[i]))
                self.layers.append(GraphConvolution(input_dim=FLAGS.property_embedding_hidden[i-1],
                                            output_dim=FLAGS.property_embedding_hidden[i],
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        # motif_embedding layer
        for i in range(0, len(FLAGS.motif_embedding_hidden)):
            if i == 0:
                # print(">> motif_embedding Layer-{} dim:    {} -> {}".format(i, FLAGS.motif_feature_dim, FLAGS.motif_embedding_hidden[i]))
                self.layers.append(GraphConvolutionMotifs(input_dim=FLAGS.motif_feature_dim,
                                                output_dim=FLAGS.motif_embedding_hidden[i],
                                                placeholders=self.placeholders,
                                                act=tf.nn.tanh,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))
            else:
                # print(">> motif_embedding Layer-{} dim:    {} -> {}".format(i, FLAGS.motif_embedding_hidden[i-1], FLAGS.motif_embedding_hidden[i]))
                self.layers.append(GraphConvolutionMotifs(input_dim=FLAGS.motif_embedding_hidden[i-1],
                                            output_dim=FLAGS.motif_embedding_hidden[i],
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))
        
        # Judge embedding dim 
        if FLAGS.property_embedding_hidden[-1] != FLAGS.motif_embedding_hidden[-1]:
            raise Exception('[ERROR] embedding last layer not have same dim!')
        if FLAGS.embeding_combination_method == "Connection":
            embedding_dim = FLAGS.property_embedding_hidden[-1] * 2
        else:
            embedding_dim = FLAGS.property_embedding_hidden[-1]
        
        # Integration layer
        for i in range(0, len(FLAGS.integration_hidden)):
            if i == 0:
                # print(">> Integration Layer-{} dim:        {} -> {}".format(i, embedding_dim, FLAGS.integration_hidden[i]))
                self.layers.append(GraphConvolution(input_dim=embedding_dim,
                                                output_dim=FLAGS.integration_hidden[i],
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=False,
                                                logging=self.logging))
            else:
                # print(">> Integration Layer-{} dim: {} -> {}".format(i, FLAGS.integration_hidden[i-1], FLAGS.integration_hidden[i]))
                self.layers.append(GraphConvolution(input_dim=FLAGS.integration_hidden[i-1],
                                                output_dim=FLAGS.integration_hidden[i],
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=False,
                                                logging=self.logging))

        # Judge output layer input dim 
        if len(FLAGS.integration_hidden) == 0:
            out_dim = embedding_dim
        else:
            out_dim = FLAGS.integration_hidden[-1]

        # Output
        # print(">> Output Layer dim:               {} -> {}".format(out_dim, self.output_dim))
        self.layers.append(GraphConvolution(input_dim=out_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        # 1. property_embedding_hidden layer
        self.activations.append(self.inputs)
        for i in range(0, len(FLAGS.property_embedding_hidden)):
            # print(">> Input shape: {}".format(self.activations[-1].get_shape()))
            layer = self.layers[i]
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        property_embedding = self.activations[-1]
        # print("   property_embedding shape: {}".format(property_embedding.get_shape()))

        # 2. motif_embedding_hidden layer
        self.activations.append(self.motifinputs)
        for i in range(0, len(FLAGS.motif_embedding_hidden)):
            # print(">> Input shape: {}".format(self.activations[-1].get_shape()))
            layer = self.layers[i + len(FLAGS.property_embedding_hidden)]
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        motif_embedding = self.activations[-1]
        # print("   motif_embedding shape: {}".format(motif_embedding.get_shape()))

        # 3. embedding polymerization
        if FLAGS.embeding_combination_method == "Hadamard":
            combination = tf.multiply(property_embedding, motif_embedding)
            self.activations.append(combination)
        elif FLAGS.embeding_combination_method == "Summation":
            combination = tf.add(property_embedding, motif_embedding)
            self.activations.append(combination)
        elif FLAGS.embeding_combination_method == "Connection":
            combination = tf.concat([property_embedding, motif_embedding], 1)
            self.activations.append(combination)
        else:
            raise Exception("[ERROR] the embeding_combination_method not exist.")
        # print("   combination shape: {}".format(combination.get_shape()))

        # 4. Integration layer
        for i in range(0, len(FLAGS.integration_hidden)):
            # print(">> Input shape: {}".format(self.activations[-1].get_shape()))
            layer = self.layers[i + len(FLAGS.property_embedding_hidden) + len(FLAGS.motif_embedding_hidden)]
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        
        # 5. Output layer
        # print(">> Input shape: {}".format(self.activations[-1].get_shape()))
        layer = self.layers[-1]
        hidden = layer(self.activations[-1])
        self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softmax(self.outputs)

#######################################################