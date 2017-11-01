import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#2-d tensor of floating number with shape 784. None represent that dimension can be of any length
x = tf.placeholder(tf.float32,[None, 784] )

#Tensor 784X10 full with zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x , W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

#function that shifts(varia/desplaza) the variable in the direction of minimize the cost.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#Cast from boolean vector to float32 [1,0]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


