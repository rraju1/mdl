""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function
from datetime import datetime

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import numpy as np
import tensorflow as tf
import os
import argparse
# Parameters
learning_rate = .005
epsilon = .1
training_epochs = 150
batch_size = 10000
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_ori"
logdir = "{}/run-{}/".format(root_logdir, now)


# tf Graph input
X = tf.placeholder("float", [None, n_input], name="x")
Y = tf.placeholder("float", [None, n_classes], name="labels")

# Store layers weight & bias
#weights = {
    #'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="feed_weight1"),
    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="feed_weight2"),
    #'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="feed_weight3")
#}
#
#biases = {
    #'b1': tf.Variable(tf.random_normal([n_hidden_1]), tf.float64),
    #'b2': tf.Variable(tf.random_normal([n_hidden_2]), tf.float64),
    #'out': tf.Variable(tf.random_normal([n_classes]), tf.float64)
#}

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([size_in, size_out], name="W"))
        b = tf.Variable(tf.random_normal([size_out], name="B"))
        mul = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        return mul

def relu(input, name="relu"):
    with tf.name_scope(name):
        relu_act = tf.nn.relu(input, name="Act")
        return relu_act


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    #layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_1_ba = fc_layer(x, n_input, n_hidden_1, name="fc1")
    layer_1 = relu(layer_1_ba)
    tf.summary.histogram("fc1/relu", layer_1)
    # Hidden fully connected layer with 256 neurons
    #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_2_ba = fc_layer(layer_1, n_hidden_1, n_hidden_2, name="fc2")
    layer_2 = relu(layer_2_ba)
    tf.summary.histogram("fc2/relu", layer_2)
    # Output fully connected layer with a neuron for each class
    #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = fc_layer(layer_2, n_hidden_2, n_classes, name="fc3")
    return out_layer

# Construct model
logits = multilayer_perceptron(X)
logits = tf.identity(logits, "logits")
# Define loss and optimizer

with tf.name_scope("xent"):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_op = tf.reduce_mean(cost) 
    tf.summary.scalar("xent", loss_op)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

with tf.name_scope("accuracy"):
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()


file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
#std_array = []

#tenList = tf.Variable(tf.concat([tf.reshape(weights['h1'], [-1]),tf.reshape(weights['h2'], [-1]), tf.reshape(weights['out'], [-1])], axis=0))
#tf.hessians(loss_op, tf.reshape(weights['h1'], [-1]))

with tf.Session() as sess:
    sess.run(init)
    export_dir = '/research/rraju2/mdl/tensorflow_scripts/results_ori'
    if not os.path.exists(export_dir):
    	os.makedirs(export_dir)
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            cost_sum = sess.run(summ, feed_dict={X: batch_x, Y: batch_y})
            file_writer.add_summary(cost_sum, epoch)
            #std_array.append(avg_cost)
            #print("Standard deviation: ", np.std(std_array))
    

    print("Optimization Finished!")
    # Test model
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    log_logits = logits.eval({X: mnist.test.images, Y: mnist.test.labels})
    
    #print(tf.reshape(weights['h1'], [-1]).shape)
    #print(tf.reshape(weights['h2'], [-1]).shape)
    #print(tf.reshape(weights['out'], [-1]).shape)
    
    #print(tenList.shape)
    ##flat_list = [item for sublist in tenList for item in sublist]
    #tvars = tf.trainable_variables()
#
    #dloss_dw = tf.gradients(loss_op, tvars)[0]
    #dim, _ = dloss_dw.get_shape()
#
    #hess = []
    #for i in range(dim):
        ## tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            #dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
            #ddfx_i = tf.gradients(dfx_i, tenList)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            #hess.append(ddfx_i)
    #hess = tf.squeeze(hess)
    #print("Hessian:", hess.eval({X: mnist.test.images, Y: mnist.test.labels}))
    np.savetxt("logits.csv",log_logits, delimiter=", ")
    saver.save(sess, "/research/rraju2/mdl/tensorflow_scripts/results_ori/model.ckpt")
file_writer.close()
