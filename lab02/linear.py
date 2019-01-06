import tensorflow as tf
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# weight와 bias에 random한 값을 줌
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Our hypothesis Wx + b
hypothesis = x_train * W + b

#cost/loss function
#reduce_mean : 주어진 값들의 평균을 구해주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

#Make a session
sess = tf.Session()
#initialize global variables
# W, b를 사용하기 위해서는 global variables로 initialize를 해줘야 함
sess.run(tf.global_variables_initializer())
# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))