import tensorflow as tf

node1 = tf.constant(5,tf.int8)
node2 = tf.constant(2,tf.int8)

node3 = node2 + node1
sess = tf.Session()

file_writer = tf.summary.FileWriter('C:\\Users\\sijo.ISPG\\Documents\\work\\TF\\Study\\run-or-walk\\graph',sess.graph)
print(sess.run(node3))

sess.close()