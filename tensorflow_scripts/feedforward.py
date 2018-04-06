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
from __future__ import division
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
epsilon = .01
training_epochs = 150
batch_size = 10000
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


parser = argparse.ArgumentParser()
parser.add_argument('--lambda_term', type=float, default=100000,
                    help='lambda term')
args = parser.parse_args()


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


# tf Graph input
X = tf.placeholder("float", [None, n_input], name="x")
Y = tf.placeholder("float", [None, n_classes], name="labels")

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

def get_weights():
    return [v for v in tf.trainable_variables() if "Variable:0" in v.name]


def ft(win):
    test = tf.zeros(tf.shape(win),tf.float32)
    for i in range(n_classes):
        test += tf.square(tf.gradients(logits[:,i],win))
        #test = tf.Print(test, [logits[i].shape], "ft shape of gradient with logit[i]: ")
    return tf.clip_by_value(test,1e-37,1e+37)

def sec_term():
    accumulator = 0
    weights = get_weights()
    for i in range(n_classes):
        test = 0
        for j in weights:
            num   = tf.abs(tf.gradients(logits[:,i], j))
            denom = tf.sqrt(ft(j))
            divisor = tf.div(num,denom)
            test += tf.reduce_sum(divisor)
        accumulator += tf.square(test)
    return tf.log(tf.clip_by_value(accumulator,1e-37,1e+37))

# Construct model
logits = multilayer_perceptron(X)
logits = tf.identity(logits,"logits")

with tf.name_scope("xent"):
    xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
    tf.summary.scalar("xent", xentropy) 

L = 0
#iterating over all variables
for variable in tf.trainable_variables():  
    local_parameters=1
    shape = variable.get_shape()  #getting shape of a variable
    for i in shape:
        local_parameters*=i.value  #mutiplying dimension values
    L += local_parameters
#print(L)
#----------------- try something
with tf.name_scope("regularizer"):
    first_term  = 0
    with tf.name_scope("sec_term"):
        second_term = sec_term() * L
        tf.summary.scalar("second_term",second_term)
    with tf.name_scope("first_term"):
        weights = get_weights()
        for i in weights:
            first_term  += tf.reduce_sum(tf.log(ft(i)))
        tf.summary.scalar("first_term", first_term)
    const = -1 * L * tf.log(epsilon)
    lambda1 = 1/args.lambda_term
    regularizer = 0.5 * lambda1*(const  + first_term + second_term)
    tf.summary.scalar("regularizer", regularizer)


#regularizer = tf.Print(regularizer, [regularizer], "this is the regularizer term")
# Define loss and optimizer

#loss_op = tf.reduce_mean(tf.reduce_mean(tf.losses.mean_squared_error(logits,Y)) + regularizer)
#xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#    logits=logits, labels=Y)) 
with tf.name_scope("train_loss_op"):
    loss_op = xentropy + regularizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

with tf.name_scope("accuracy"):
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
# Initializing the variables
summ = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()

saver = tf.train.Saver()
#std_array = []

with tf.Session() as sess:
    sess.run(init)
    export_dir = '/research/rraju2/mdl/tensorflow_scripts/results'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    # Training cycle
    
#    graph = tf.get_default_graph()
#    for op in graph.get_operations():
#       print(op.name)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, xentropy], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            cost_sum = sess.run(summ, feed_dict={X: batch_x, Y: batch_y})
            file_writer.add_summary(cost_sum,epoch)
            #std_array.append(avg_cost)
            #print("Standard deviation: ", np.std(std_array))
    

    print("Optimization Finished!")
    # Test model
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    log_logits = logits.eval({X: mnist.test.images, Y: mnist.test.labels})
    np.savetxt("logits_fms.csv",log_logits, delimiter=", ")
    saver.save(sess, "/research/rraju2/mdl/tensorflow_scripts/results/model.ckpt")
file_writer.close()
    
