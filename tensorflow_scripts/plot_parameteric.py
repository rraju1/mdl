import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# grab all weights from the network
def get_weights():
    return [v for v in tf.trainable_variables() if "Variable:0" in v.name]
# need to get all the weights from both graphs evaluated as numpy arrays to be converted as tensors
def eval_weights():
	return [v.eval() for v in tf.trainable_variables() if "Variable:0" in v.name]

# get all set of weights which give flat minima
clear_devices=True
checkpoint = tf.train.get_checkpoint_state("results")
input_checkpoint = checkpoint.model_checkpoint_path

with tf.Session() as sess:
	saver_flat = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
	saver_flat.restore(sess, input_checkpoint)
	flat_weights = eval_weights()

tf.reset_default_graph()
# get all set of weights which give sharp minima
checkpoint = tf.train.get_checkpoint_state("results_ori")
input_checkpoint = checkpoint.model_checkpoint_path
# get data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

alpha_range = np.linspace(-1, 2, 25)
data_for_plots = np.zeros((25, 2))

with tf.Session() as sess1:
	saver_sharp = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
	saver_sharp.restore(sess1, input_checkpoint)
	sharp_weights = eval_weights()

	graph = tf.get_default_graph()

	p1 = graph.get_tensor_by_name('x:0')
	p2 = graph.get_tensor_by_name('labels:0')
	xent = graph.get_tensor_by_name('xent/Mean:0')
	accuracy = graph.get_tensor_by_name('accuracy/Mean:0')
	x = 0
	#print(type(xent))

	for alpha in alpha_range:
		#weights in the current graph
		weights = get_weights()
		#print(weights[0].eval())
		for i in range(len(sharp_weights)):
			temp = alpha * tf.convert_to_tensor(sharp_weights[i]) + (1 - alpha) * tf.convert_to_tensor(tf.convert_to_tensor(flat_weights[i]))
			weights[i].assign(temp).eval()
		#print(weights[0].eval())
		cross, test_acc = sess1.run([xent, accuracy], {p1: mnist.test.images, p2: mnist.test.labels})
		#print(test_acc)
		data_for_plots[x, :] = [cross, test_acc]
		x += 1


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(alpha_range, data_for_plots[:, 0], 'b-')
ax2.plot(alpha_range, data_for_plots[:, 1]*100., 'r-')

ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.grid(b=True, which='both')
plt.savefig('Figures/MLP.pdf')
print('Plot saved in Figures/MLP.pdf')
