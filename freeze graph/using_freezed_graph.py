import tensorflow as tf

with tf.gfile.GFile("frozen_model.pb","rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
	tf.import_graph_def(graph_def)
	fileWriter = tf.summary.FileWriter("./freezegraph",graph)
	
fileWriter.close()