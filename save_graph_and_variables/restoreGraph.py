import tensorflow as tf

with tf.Session() as sess:
    new_saver =tf.train.import_meta_graph('checks/mymodel.meta')
    new_saver.restore(sess,"checks/mymodel")


    #for var in tf.global_variables():
    	#print(var.op.name)
    W = tf.get_variable("Variable")
    print("W:",sess.run(W))