# Author:   Jin Xu
# Data:     2020-01-10
# Function: Run training

#--------------------------     import package    --------------------------#

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import winsound
from Code_utils import *
from Code_models import GCN, MLP, MORE

#---------------------------     main process     ---------------------------#

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'MORE', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'MORE'

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')

flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

flags.DEFINE_integer('motif_feature_dim', 6, 'the dim of motif features')
flags.DEFINE_list('property_embedding_hidden', [256], 'the hidden layer number of property embedding')
flags.DEFINE_list('motif_embedding_hidden', [256], 'the hidden layer number of motif embedding')
flags.DEFINE_list('integration_hidden', [], 'the hidden layer number of integration')
flags.DEFINE_string('embeding_combination_method', "Hadamard", 'the method of embedding combination') 
# embeding_combination_method ---- "Hadamard", "Summation", "Connection"

# batch_run
use_batch = False
if use_batch:
    FLAGS.model = 'MotifGCN'
    lr = [0.01, 0.001, 0.0003, 0.003]
    le = [300, 500, 1000, 2000]
    mo = ["Hadamard", "Summation", "Connection"]
    la = [32, 64, 128, 256, 512]
    mode_list = []
    for i in range(0, 4):
        temp1 = [lr[i], le[i]]
        for j in range(0, 3):
            temp2 = temp1 + [mo[j]]
            for k in range(0, 5):
                temp3 = temp2 + [la[k]]
                mode_list.append(temp3)

    mode = mode_list[59] # 0-14, 15-29, 30-44, 45-59

    print(mode)
    FLAGS.learning_rate = mode[0]
    FLAGS.epochs = mode[1]
    FLAGS.embeding_combination_method = mode[2]
    FLAGS.motif_embedding_hidden = [mode[3]]
    FLAGS.property_embedding_hidden = [mode[3]]

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, motiffeatures = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
motiffeatures = preprocess_features(motiffeatures)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'MORE':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MORE
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'motiffeatures': tf.sparse_placeholder(tf.float32, shape=tf.constant(motiffeatures[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout   
    'num_motif_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, motiffeatures, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, motiffeatures, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
train_acc, val_acc, Tacc = [], [], []
train_loss, val_loss, Tloss = [], [], []

# Train model
train_starttime = time.time()
train_time_list = []
stop_epoch = 0
for epoch in range(FLAGS.epochs):

    t = time.time()
    stop_epoch = epoch + 1
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, motiffeatures, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    train_acc.append(outs[2])
    train_loss.append(outs[1])
    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, motiffeatures, placeholders)
    cost_val.append(cost)
    val_acc.append(acc)
    val_loss.append(cost)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    train_time_list.append(time.time() - t)

    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, motiffeatures, placeholders)
    Tacc.append(test_acc)
    Tloss.append(test_cost)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")
train_time = time.time() - train_starttime

# Tacc = Tacc[-max(FLAGS.early_stopping * 2, 20):]
# Testing
test_starttime = time.time()
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, motiffeatures, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
print("Max test acc = {:.5f}".format(max(Tacc)))
test_time = time.time() - test_starttime

# Save
with open("Result\\Train_log.csv", mode='a') as f:
    f.write("{},{},{},{},{},{},{},{},{},{:.4f},{:.4f},(best={:.4f}),{:.4f},{},{:.6f},{:.6f},{:.6f}\n".\
        format(seed,FLAGS.dataset,FLAGS.model,FLAGS.learning_rate, FLAGS.dropout, FLAGS.embeding_combination_method,\
               str(FLAGS.property_embedding_hidden), str(FLAGS.motif_embedding_hidden), str(FLAGS.integration_hidden),\
               test_acc,test_cost, max(Tacc),test_duration,stop_epoch, train_time, np.mean(train_time_list), test_time))
with open("Result\\Loss.csv", mode='a') as f:
    for i in train_loss:
        f.write("{:.6f},".format(i))
    f.write("\n")
    for i in val_loss:
        f.write("{:.6f},".format(i))
    f.write("\n")
    for i in Tloss:
        f.write("{:.6f},".format(i))
    f.write("\n")
with open("Result\\Acc.csv", mode='a') as f:
    for i in train_acc:
        f.write("{:.6f},".format(i))
    f.write("\n")
    for i in val_acc:
        f.write("{:.6f},".format(i))
    f.write("\n")
    for i in Tacc:
        f.write("{:.6f},".format(i))
    f.write("\n")

# Sound   
duration = 500  # millisecond
freq = 600  # Hz
winsound.Beep(freq, duration)