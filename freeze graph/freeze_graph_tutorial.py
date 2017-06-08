import tensorflow as tf

import numpy as np

X = tf.placeholder(dtype = tf.float32, shape = [None], name = "x")

y = tf.placeholder(dtype = tf.float32, shape = [None], name = "y")

W = tf.Variable(1.0,dtype = tf.float32,name = "weight")

b = tf.Variable(0.0, dtype = tf.float32, name = "bias")

y_predict = tf.multiply(X,W) + b


loss = tf.reduce_mean(tf.square(y_predict - y))
tf.summary.scalar("loss",loss)
optimize = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

merged_summary = tf.summary.merge_all()


init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	train_writer = tf.summary.FileWriter("./graphs",sess.graph)

	inputs = np.linspace(0,2,num = 2000)

	labels = 3 * inputs + np.random.randn(2000)


	for step in range(1000):
		batch_indexes = np.random.choice(2000,100)
		x_batch = inputs[batch_indexes]
		y_labels = labels[batch_indexes]

		loss,summary = sess.run([optimize,merged_summary],feed_dict = {X:x_batch, y : y_labels})

		train_writer.add_summary(summary)

		saver.save(sess, "./checks/LinearLineFit",global_step = step)

train_writer.close()



