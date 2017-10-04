import tensorflow as tf
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
print(sess.run(adder_node,{a:[1,3], b: [2,4]})) #matriz operatation
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
print(sess.run(loss, {x:[1,2,3,4,5,6], y : [0,-1,-2,-3,-4,-5]}))