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
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="feed_weight1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="feed_weight2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="feed_weight3")
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), tf.float64),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), tf.float64),
    'out': tf.Variable(tf.random_normal([n_classes]), tf.float64)
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

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
def ft(win):
    test = tf.zeros(tf.shape(win),tf.float32)
    for i in range(n_classes):
        test += tf.square(tf.gradients(logits[:,i],win))
        #test = tf.Print(test, [logits[i].shape], "ft shape of gradient with logit[i]: ")
    return tf.clip_by_value(test,1e-37,1e+37)

def sec_term():
    accumulator = 0
    for i in range(n_classes):
        test = 0
        for j in weights:
            num   = tf.abs(tf.gradients(logits[:,i], weights[j]))
            denom = tf.sqrt(ft(weights[j]))
            divisor = tf.div(num,denom)
            test += tf.reduce_sum(divisor)
        accumulator += tf.square(test)
    return tf.log(tf.clip_by_value(accumulator,1e-37,1e+37))

first_term  = 0
second_term = sec_term() * L
for i in weights:
    first_term  += tf.reduce_sum(tf.log(ft(weights[i])))
    
# tf.clip_by_value(test,1e-37,1e+37)

const = -1 * L * tf.log(epsilon)

lambda1 = 1/args.lambda_term
#print("this is lambda: ", lambda1)
#first_term = tf.Print(first_term, [first_term], "this is the first term in the equation")
#second_term = tf.Print(second_term, [second_term], "this is the second term in the equation")
#const = tf.Print(const, [const], "this is the constant term in the equation")
regularizer = 0.5 * lambda1*(const  + first_term + second_term)
regularizer = tf.Print(regularizer, [regularizer], "this is the regularizer term")
# Define loss and optimizer

#loss_op = tf.reduce_mean(tf.reduce_mean(tf.losses.mean_squared_error(logits,Y)) + regularizer)
xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y)) 
loss_op = xentropy + regularizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables

cost_summary = tf.summary.scalar('Cross entropy loss', xentropy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()

saver = tf.train.Saver()
std_array = []

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
            cost_sum = cost_summary.eval(feed_dict={X: batch_x, Y: batch_y})
            file_writer.add_summary(cost_sum,epoch)
            #std_array.append(avg_cost)
            #print("Standard deviation: ", np.std(std_array))
    

    print("Optimization Finished!")
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    log_logits = logits.eval({X: mnist.test.images, Y: mnist.test.labels})
    np.savetxt("logits_fms.csv",log_logits, delimiter=", ")
    saver.save(sess, "/research/rraju2/mdl/tensorflow_scripts/results/model.ckpt")
