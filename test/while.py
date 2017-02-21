from __future__ import print_function

import tensorflow as tf

def cond(sequence_len, step):
    return tf.less(step,sequence_len)

def body(sequence_len, step): 

    begin = tf.get_variable("begin",[3],dtype=tf.int32,initializer=tf.constant_initializer(0))
    begin = tf.scatter_update(begin,1,step,use_locking=None)

    tf.get_variable_scope().reuse_variables()
    with tf.control_dependencies([begin]):
        return (sequence_len, step+1)

with tf.Graph().as_default():

    sess = tf.Session()
    step = tf.constant(0)
    sequence_len  = tf.constant(100000)
    _,step, = tf.while_loop(cond,
                    body,
                    [sequence_len, step], 
                    parallel_iterations=1000, 
                    back_prop=True, 
                    swap_memory=False, 
                    name=None)

    begin = tf.get_variable("begin",[3],dtype=tf.int32)

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run([begin,step]))
