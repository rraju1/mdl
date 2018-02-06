import tensorflow as tf
import argparse
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import inspect_checkpoint as chkp

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph
# prefix/add
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--frozen", default="results/frozen_model.pb", type=str, help = "Quantized/Frozen model to import")

        parser.add_argument("--model_dir", type=str, default="results", help="Model folder to export")
	args = parser.parse_args()
	
	graph = load_graph(args.frozen)
# 	for op in graph.get_operations():
# 		print(op.values())
# 		print(op.name)
	p1  = graph.get_tensor_by_name('prefix/Placeholder:0')
	p2  = graph.get_tensor_by_name('prefix/Placeholder_1:0')
	output_node = graph.get_tensor_by_name('prefix/Mean_1:0')
	log = graph.get_tensor_by_name('prefix/add:0')
	
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	original_logits = np.genfromtxt('logits.csv', delimiter=', ')
	with tf.Session(graph=graph) as sess:
		print("Accuracy:", sess.run(output_node,({p1: mnist.test.images, p2: mnist.test.labels})))
		numpy_log = sess.run(log,({p1: mnist.test.images, p2: mnist.test.labels}))
		err = numpy_log - original_logits
		print(np.linalg.norm(err, ord=2))	
