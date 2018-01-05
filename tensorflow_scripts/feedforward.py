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

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.001
epsilon = .1
training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), tf.float64),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), tf.float64),
    'out': tf.Variable(tf.random_normal([n_classes]), tf.float64)
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return layer_1, layer_2, out_layer

# Construct model
l1, l2, logits = multilayer_perceptron(X)
#----------------- try something
def ft(win):
	test = tf.zeros(tf.shape(win),tf.float32)
	for i in range(n_classes):
	    test += tf.square(tf.gradients(logits[i],win))
	return test
test_ft = tf.reduce_sum(tf.log(ft(weights['out'])))
test_ft1 = tf.reduce_sum(tf.log(ft(weights['h2'])))
test_ft2 = tf.reduce_sum(tf.log(ft(weights['h1'])))

#----------------- try something
def st(win1):
	test = 0
	for i in range(n_classes):
	    inter1 = tf.gradients(logits[i],win1)
	    inter2 = tf.sqrt(ft(win1))
	    test += tf.square(tf.reduce_sum(tf.div(inter1,inter2)))
	return tf.log(test)
test_st = st(weights['out'])
test_st1 = st(weights['h2'])
test_st2 = st(weights['h1'])
# Define loss and optimizer
lambda1 = 0.00001
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y)) + 0.5 * lambda1*(-1 * tf.log(epsilon) + test_ft + test_st + test_ft1 + test_st1)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    export_dir = '/research/rraju2/mlp_mnist_test/'
#    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
 #   builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
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
    print("Optimization Finished!")
  #  builder.save()	

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

#    w1, w2, w3 = sess.run([weights['h1'], weights['h2'], weights['out']])
#    for i in weights:
#	wx = sess.run(weights[i])
#	txt = 'w' + str(i) + 'Updated1.csv'
#    	np.savetxt(txt,wx, delimiter=", ")
