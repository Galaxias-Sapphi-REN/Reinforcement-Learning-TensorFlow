# -*- coding:utf-8 -*-
import tensorflow as tf

data = [k*1. for k in range(10)]
optimizer = tf.train.GradientDescentOptimizer(0.05)
w = tf.Variable(0.)
b = tf.Variable(0.)
q_x = tf.FIFOQueue(100000, tf.float32)
q_y = tf.FIFOQueue(100000, tf.float32)
gs = tf.Variable(0)
i = tf.Variable(0, name='loop_i')


def cond(i):
    return i < 10

def body(i):
    # Dequeue a new example each iteration.
    x = q_x.dequeue()
    y = q_y.dequeue()

    # Compute the loss and gradient update based on the current example.
    loss = (tf.add(tf.mul(x, w), b) - y)**2
    train_op = optimizer.minimize(loss, global_step=gs)

    # Ensure that the update is applied before continuing.
    return tf.tuple([tf.add(i, 1)], control_inputs=[train_op])

s = tf.Session()
s.run(tf.global_variables_initializer())

loop = tf.while_loop(cond, body, [i])
for _ in range(1):
    s.run(q_x.enqueue_many((data, )))
    s.run(q_y.enqueue_many((data, )))

print s.run(loop)
s.close()
