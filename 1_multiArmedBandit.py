# -*- coding:utf-8 -*-


'''
  n-Armed Bandit problem implemented in TensorFlow.

author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2017.2.10
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

epsilons = tf.Variable([0, 0.05, 0.15, 0.20])
epsilon = 4
times = 30
nbandit = 5
arms = 10
stepSize = 0.1
softmax = True
sampleAverage = True

## 问题回顾，一共十个arm，一共n个bandit，一共t次实验，并且分成多个epsilon来实验
#每个bandit的每个arm都有一个真实价值,并且单个bandit的价值是服从稳定的分布
#每个bandit都有一个best action, epoch, meanReward, actionCount
#一共有nbandit*epsilon*times个随机数与epsilon进行比较
def visualize(epsilons, fr, fo):
    for eps, ba in zip(epsilons, fo):
	plt.plot(ba, label='epsilon='+str(eps))
    plt.xlabel('steps')
    plt.ylabel('optimal action %')
    plt.legend()
    plt.show()
    for eps, ar in zip(epsilons, fr):
        plt.plot(ar, label='epsilon='+str(eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.show()



all_random_numbers = tf.get_variable('all_random_numbers', shape=[epsilon, nbandit, times], initializer=tf.random_uniform_initializer(), dtype=tf.float32)
all_true_value = tf.get_variable('all_true_value', shape=[epsilon, nbandit, arms], initializer=tf.random_normal_initializer(), dtype=tf.float32)
all_best_action = tf.cast(tf.argmax(all_true_value, axis=2), tf.int32)
all_arms = tf.get_variable('all_arms', shape=[arms], initializer=tf.constant_initializer(range(arms)), dtype=tf.int32)
rand = tf.get_variable('rand', shape=[epsilon, nbandit, times], initializer=tf.random_normal_initializer(), dtype=tf.float32)


with tf.variable_scope('all'):
    epoch = tf.get_variable('epoch', shape=[epsilon*nbandit], initializer=tf.zeros_initializer(), dtype=tf.int32)
    mean_reward = tf.get_variable('mean_reward', shape=[epsilon*nbandit], initializer=tf.zeros_initializer(), dtype=tf.float32)
    action_count = tf.get_variable('action_count', shape=[epsilon*nbandit*arms], initializer=tf.zeros_initializer(), dtype=tf.float32)
    estimated_value = tf.get_variable('estimated_value', shape=[epsilon*nbandit*arms], initializer=tf.constant_initializer(1.), dtype=tf.float32)

with tf.variable_scope('result'):
    final_reward = tf.get_variable('final_reward', shape=[epsilon*nbandit*times], initializer=tf.zeros_initializer(), dtype=tf.float32)
    final_optimal = tf.get_variable('final_optimal', shape=[epsilon*nbandit*times], initializer=tf.zeros_initializer(), dtype=tf.float32)
    
i1 = tf.Variable(range(epsilon),tf.int32)
i2 = tf.Variable(range(nbandit),tf.int32)
i3 = tf.Variable(range(times),tf.int32)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
def policy(index_epsilon, index_nbandit, index_time):
    # 1.get action

    index1 = index_epsilon*(index_nbandit*index_time)+index_nbandit*index_time+index_time

    index2 = index_epsilon*index_nbandit+index_nbandit
    def action_explore():
	return tf.gather_nd(tf.random_shuffle(all_arms),[0])

    def softmax_or_not():
	with tf.variable_scope('all', reuse=True):
            if softmax:
    	        estimated_value = tf.get_variable('estimated_value', dtype=tf.float32)
	        estimated_value = tf.reshape(estimated_value, [epsilon, nbandit, arms])
	        softmax_prob = tf.nn.softmax(tf.gather_nd(estimated_value, [index_epsilon, index_nbandit]))
                samples = tf.multinomial(tf.reshape(softmax_prob,[1,-1]), arms) 
                samples = tf.cast(samples, tf.int32)
                samples = tf.reshape(samples, [arms])
                return tf.gather_nd(samples,[0])
            else:
	        return tf.argmax(tf.gather_nd(estimated_value, [index_epsilon, index_nbandit]),axis=0) 

    random = tf.gather_nd(all_random_numbers, [index_epsilon, index_nbandit, index_time]) 
    eps = tf.gather_nd(epsilons, [index_epsilon])
    explore_or_not = tf.less(random, eps)

    action = tf.cond(explore_or_not, action_explore, softmax_or_not) 

    best_action = tf.gather_nd(all_best_action, [index_epsilon, index_nbandit])

    optimal = tf.equal(action, best_action)
    optimal = tf.cast(optimal, tf.float32)

    index3 = index_epsilon*(index_nbandit*action)+index_nbandit*action+action

    with tf.variable_scope('all', reuse=True):
    	estimated_value = tf.get_variable('estimated_value', dtype=tf.float32)
        estimated_value = tf.reshape(estimated_value, [epsilon*nbandit*arms])

    # 2.get reward
    reward = tf.gather_nd(all_true_value, [index_epsilon, index_nbandit, action])+tf.gather_nd(rand, [index_epsilon, index_nbandit, index_time])
    with tf.variable_scope('all', reuse=True):
        epoch = tf.get_variable('epoch', dtype=tf.int32)
        mean_reward = tf.get_variable('mean_reward', dtype=tf.float32)
        action_count = tf.get_variable('action_count', dtype=tf.float32)

        epoch = tf.scatter_add(epoch, index_epsilon*nbandit+index_nbandit, 1)
	tmp_epoch = tf.gather_nd(epoch, [index_epsilon*nbandit+index_nbandit])
	tmp_epoch = tf.to_float(tmp_epoch)
	tmp_mean_reward = tf.gather_nd(mean_reward, [index_epsilon*nbandit+index_nbandit])
	tmp_mean_reward = (tmp_epoch-1)*tmp_mean_reward/tmp_epoch + reward/tmp_epoch
	mean_reward = tf.scatter_update(mean_reward, index_epsilon*nbandit+index_nbandit, tmp_mean_reward)

    with tf.variable_scope('all', reuse=True):
    	action_count = tf.get_variable('action_count', dtype=tf.float32)
    	estimated_value = tf.get_variable('estimated_value', dtype=tf.float32)
	tmp_ev = tf.gather_nd(estimated_value, [index3])
    	if sampleAverage:
	    tmp_ac = tf.gather_nd(action_count, [index3])
	    value = 1.0/(tmp_ac+1)*(reward-tmp_ev)
	else:
	    value = stepSize*(reward-tmp_ev) 
	estimated_value = tf.scatter_add(estimated_value, index3, value)
	action_count = tf.scatter_add(action_count, index3, 1)
	    
    with tf.variable_scope('result', reuse=True):
        final_reward = tf.get_variable('final_reward', dtype=tf.float32)
        final_optimal = tf.get_variable('final_optimal', dtype=tf.float32)
	final_reward = tf.scatter_update(final_reward, index1, reward)
	final_optimal = tf.scatter_update(final_optimal, index1, optimal)
        sess.run([final_reward, final_optimal])
        return final_reward, final_optimal 


idx = 0
for i in range(epsilon):
    for j in range(nbandit):
        for k in range(times):
	    print idx
	    fr, fo = policy(tf.gather(i1,i),tf.gather(i2,j),tf.gather(i3,k))
	    idx += 1
	    

with tf.variable_scope('result', reuse=True):
    fr = tf.get_variable('final_reward', dtype=tf.float32)
    fo = tf.get_variable('final_optimal', dtype=tf.float32)
    fr = tf.reshape(fr,[epsilon, nbandit, times])
    fo = tf.reshape(fo,[epsilon, nbandit, times])

    fr = tf.reduce_mean(fr, 1)
    fo = tf.reduce_mean(fo, 1)

    fr, fo, epsilons = sess.run([fr, fo, epsilons])
    print fr
    print fo

    visualize(epsilons, fr, fo)

