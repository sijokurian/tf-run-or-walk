import tensorflow as tf

#Model parameters
W = tf.Variable([.3],tf.float32)
b = tf.Variable([.3],tf.float32)

#features & labels
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#model
linear_model = W * x + b

#Loss
sqaured_dela = tf.square(linear_model-y)
loss = tf.reduce_sum(sqaured_dela)

#optimize
optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

#initialize variables
sess = tf.Session()
var_init = tf.global_variables_initializer()
sess.run(var_init)

#Run training
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))
#file_writer = tf.summary.FileWriter('C:\\Users\\sijo.ISPG\\Documents\\work\\TF\\Study\\tf-run-or-walk\\tf-run-or-walk\\graph',sess.graph)

sess.close()