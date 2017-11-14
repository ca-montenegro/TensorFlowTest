import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0, dtype=tf.float32)
node2=tf.constant(4.0)
print(node1, node2)
sess = tf.Session()
print(sess.run([node1,node2]))
node3 = tf.add(node1, node2)
print("Node3: ", node3)
print("sess.run(node3): ", sess.run(node3))
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b #provides shortcut fot tf.add(a,b)
print(sess.run(adder_node,{a:3, b:4.5}))
print(sess.run(adder_node,{a:[1,3], b: [2,4]})) #matriz operation
add_and_triple = adder_node*3
print(sess.run(add_and_triple,{a:3, b:4.5}))

w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w*x+b

init = tf.global_variables_initializer() #Necessary step to initialize variables.
sess.run(init)



print(sess.run(linear_model, {x:[1,2,3,4,5,6]}))

y = tf.placeholder((tf.float32))
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[0,1,2,3,4,5,6], y:[0,0,0.3000001,0.6000002,0.900004, 1.2000005, 1.5]}))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[-1,-2,-3,-4]})
print(sess.run([w,b]))



pl = tf.placeholder(tf.float32)
b = pl*2
dictionary={pl: [ [ [1,2,3]] , [ [13,14,15] ] ] }
with tf.Session() as sess:
    result = sess.run(b, feed_dict=dictionary)
print (result)


# x = tf.placeholder("float")
# y = tf.placeholder("float")
# # w is the variable storing our values. It is initialised with starting "guesses"
# # w[0] is the "a" in our equation, w[1] is the "b"
# w = tf.Variable([1.0, 2.0], name="w")
# # Our model of y = a*x + b
# y_model = tf.multiply(x, w[0]) + w[1]
#
# # Our error is defined as the square of the differences
# error = tf.square(y - y_model)
# # The Gradient Descent Optimizer does the heavy lifting
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
#
# # Normal TensorFlow - initialize values, create a session and run the model
# model = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(model)
#     for i in range(1000):
#         x_value = np.random.rand()
#         y_value = x_value * 2 + 6
#         session.run(train_op, feed_dict={x: x_value, y: y_value})
#
#     w_value = session.run(w)
#     print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))