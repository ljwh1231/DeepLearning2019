import tensorflow as tf
# build graph using TF operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) # node3 = node1 + node2

print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)

# node들을 모두 tensor라는 datatype이므로 원하는 결과가 나오지는 않음

#Feed data and run graph
sess = tf.Session()
print("sess.run(node1, node2) :", sess.run([node1, node2]))
print("sess.run(node3) : ", sess.run(node3))