"""
Normalized architechture 
"""
def batch_norm_wrapper(inputs, is_training):
	...
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	if is_training:
		mean, var = tf.nn.moments(inputs, [0])
		...
		# learn pop_mean and pop_var here
		...
		return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
	else:
		return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


decay = 0.999  # use numbers closer to 1 if you have more data
train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))


# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, is_training, decay=0.999):
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	if is_training:
		batch_mean, batch_var = tf.nn.moments(inputs, [0])
		train_mean = tf.assign(pop_mean,
							   pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,
							  pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
											 batch_mean, batch_var, beta, scale, epsilon)
	else:
		return tf.nn.batch_normalization(inputs,
										 pop_mean, pop_var, beta, scale, epsilon)

def build_graph(is_training):
	# Placeholders
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	# Layer 1
	w1 = tf.Variable(w1_initial)
	z1 = tf.matmul(x, w1)
	bn1 = batch_norm_wrapper(z1, is_training)
	l1 = tf.nn.sigmoid(bn1)

	# Layer 2
	w2 = tf.Variable(w2_initial)
	z2 = tf.matmul(l1, w2)
	bn2 = batch_norm_wrapper(z2, is_training)
	l2 = tf.nn.sigmoid(bn2)

	# Softmax
	w3 = tf.Variable(w3_initial)
	b3 = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(l2, w3))

	# Loss, Optimizer and Predictions
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return (x, y_), train_step, accuracy, y, tf.train.Saver()

#Build training graph, train and save the trained model


tf.reset_default_graph()
(x, y_), train_step, accuracy, _, saver = build_graph(is_training=True)

acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm.tqdm(range(10000)):
        batch = mnist.train.next_batch(60)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 50 is 0:
            res = sess.run([accuracy],feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            acc.append(res[0])
    saved_model = saver.save(sess, './temp-bn-save')

print("Final accuracy:", acc[-1])


tf.reset_default_graph()
(x, y_), _, accuracy, y, saver = build_graph(is_training=False)

predictions = []
correct = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './temp-bn-save')
    for i in range(100):
        pred, corr = sess.run([tf.arg_max(y,1), accuracy],
                             feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]]})
        correct += corr
        predictions.append(pred[0])

print("PREDICTIONS:", predictions)
print("ACCURACY:", correct/100)

