import tensorflow as tf

W = tf.Variable([.3],tf.float32)
b = tf.Variable([.3],tf.float32)
x = tf.placeholder(tf.float32)


linear_model = W * x + b
sess = tf.Session()
var_init = tf.global_variables_initializer()
sess.run(var_init)

file_writer = tf.summary.FileWriter('C:\\Users\\sijo.ISPG\\Documents\\work\\TF\\Study\\tf-run-or-walk\\tf-run-or-walk\\graph',sess.graph)

print(sess.run(linear_model,{x:[1,2]}))

sess.close()