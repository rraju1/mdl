import tensorflow as tf
import os, argparse
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def posOrNeg():
	sample = random.randint(0, 1)
	if (sample == 0):
		return -1
	return 1

def add_error(a, err):
    """ Needs to return a range for specified a and err term
    """
    #return random.uniform(-1*a*err, a*err)
    a_abs = abs(a)
    mu = err * a_abs
    sigma = mu
    return posOrNeg() * random.gauss(mu, sigma)

def change_weights(model_dir, err):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 

    Args:
        model_dir: the root folder containing the checkpoint state file
        err: 	   the error introduce 
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    original_logits = np.genfromtxt('logits_fms.csv', delimiter=', ')
    
    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        graph = tf.get_default_graph()

        p1 = graph.get_tensor_by_name('Placeholder:0')
        p2 = graph.get_tensor_by_name('Placeholder_1:0')
        output = graph.get_tensor_by_name('Mean_1:0')
        log = graph.get_tensor_by_name('add:0')

        for v in tf.trainable_variables():
			if "feed_weight" in v.name:
				#v = v + posOrNeg()*err*v
				error = lambda a: a + add_error(a, err)
				v1 = tf.map_fn(error, v)
				v = tf.assign(v, v1)
				sess.run(v)
        
        numpy_log = sess.run(log,({p1: mnist.test.images, p2: mnist.test.labels}))
        print(original_logits)
        print("-----------------------")
        print(numpy_log)
        error1 = np.linalg.norm(numpy_log - original_logits)
        ori_norm = np.linalg.norm(original_logits)
        print("Norm: ", error1/ori_norm)
        print("Accuracy: ", sess.run(output,({p1: mnist.test.images, p2: mnist.test.labels})))
	#for i in tf.get_default_graph().get_operations():
	#	print(i.name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Add error and inference')
	parser.add_argument("--model_dir", type=str, default="results", help="Model folder to export")
	parser.add_argument("--err", type=float, default=0, help="Error term")
	
	args = parser.parse_args()
	change_weights(args.model_dir, args.err/100)
