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

def subset_update(old_tensor, ind, new, mode='update', scope=None):
    ''' update subset of a tensor by numpy style indexing '''
    indice = tf.constant(ind)
    scope = scope if scope is not None else 'subset_update'
    updated = tf.get_variable(scope+'/newvalue', shape=len(new), initializer=tf.constant_initializer(new))
    if mode == 'update':
        new_tensor = tf.scatter_update(old_tensor, indice, updated)
    elif mode == 'add':
        new_tensor = tf.scatter_add(old_tensor, indice, updated)
    elif mode == 'sub':
        new_tensor = tf.scatter_sub(old_tensor, indice, updated)
    else:
	raise NotImplementedError('wrong mode %s' % mode)
    return new_tensor
    

class ArmedBandit:
    def __init__(self, num_arms, session, idx, 
		 init_value = 1.,
		 epsilon = 0.1,
		 stepSize = 0.1,
		 sampleAverage = True, 
		 softmax = True):
        self.num_arms = num_arms
        self.init_value = init_value
        self.epsilon = epsilon
	self.stepSize = tf.constant(stepSize)
        self.sampleAverage = sampleAverage
	self.softmax = softmax
	self.sess = session
	self.epoch = tf.constant(0,tf.float32)
	self.meanReward = tf.constant(0.0, tf.float32)
	self.actionCount = [0]*self.num_arms
	
	idx = str(idx)+str(epsilon)+str(stepSize)+str(sampleAverage)+str(softmax)

        self.armsIndex = tf.get_variable(str(idx)+'/armsIndex', shape=[self.num_arms], initializer=tf.constant_initializer(np.arange(self.num_arms)), dtype=tf.int32)

        self.estimatedValue = tf.get_variable(str(idx)+'/estimatedValue', shape=[self.num_arms], initializer=tf.constant_initializer([self.init_value]*self.num_arms), dtype=tf.float32)

        self.trueValue = tf.get_variable(str(idx)+'/trueValue', shape=[self.num_arms], initializer=tf.random_normal_initializer(), dtype=tf.float32)

	self.bestAction = tf.cast(tf.argmax(self.trueValue, axis=0),tf.int32)

        #self.value_estimated = subset_update(self.value_estimated, [0,1,2], [1,3,5], mode='sub', scope='init')

        #self.initial_op = tf.global_variables_initializer()

    def getBestAction(self):
	# return the shuffled array

	## 1.exploration
	if self.epsilon > 0:
	    if np.random.binomial(1, self.epsilon) == 1:
	        return tf.gather(tf.random_shuffle(self.armsIndex),0)
	## 2.softmax exploration
	if self.softmax:
	    self.softmaxProb = tf.nn.softmax(self.estimatedValue) 
	    samples = tf.multinomial(tf.reshape(self.softmaxProb,[1,-1]), self.num_arms) 
	    samples = tf.cast(samples, tf.int32)
	    samples = tf.reshape(samples, [self.num_arms])
	    return tf.gather(samples,0)
	return tf.gather(tf.reshape(tf.argmax(self.estimatedValue, axis=0),[1]),0) 

    def takeBestAction(self, action):
	# reward is generated with a random added by true value
	action = action.eval()
	rand_reward = tf.constant(np.random.rand())
	reward = tf.add(rand_reward, tf.gather(self.trueValue, action))
	self.epoch = self.epoch+1
	self.meanReward = (self.epoch-1)*self.meanReward/self.epoch + reward/self.epoch
	self.actionCount[action] += 1

        c_action = tf.constant([action])
	
	if self.sampleAverage:
	    c_new = [1.0/self.actionCount[action]*(reward-tf.gather(self.estimatedValue, action))]
            self.estimatedValue = tf.scatter_add(self.estimatedValue, c_action, c_new)

	elif self.softmax:
	    ones = [0.0]*self.num_arms
	    ones[action] = 1.0
	    ones = tf.constant(ones)
	    ones = ones - self.softmaxProb
	    c_new = self.stepSize*(reward-self.meanReward)*ones
	    c_index = tf.constant(range(10))
            self.estimatedValue = tf.scatter_add(self.estimatedValue, c_index, c_new)
	    
	else:
	    c_new = [self.stepSize*(reward-tf.gather(self.estimatedValue, action))] 
            self.estimatedValue = tf.scatter_add(self.estimatedValue, c_action, c_new)
	    
	return reward
    
if __name__ == '__main__':
    with tf.Session() as sess:
	bandits = []
	nbandit = 10
	times = 100
	epsilon = [0.05, 0.1, 0.15]
	for eps in epsilon:
	    bandit = [
                       ArmedBandit(10, sess, idx=idx, epsilon=eps, sampleAverage=True, softmax=False) \
		       for idx in range(nbandit) \
		     ]
	    #for ab in bandit:
        	#sess.run(ab.initial_op)
	    bandits.append(bandit)
	
	ave_rewards = [np.zeros(times, dtype='float') for _ in range(len(epsilon))]
	best_actions = [np.zeros(times, dtype='float') for _ in range(len(epsilon))]

        sess.run(tf.global_variables_initializer())

	t_start = time.time()
	for ind, bandit in enumerate(bandits):
	    for i in range(nbandit):
		for t in range(times):
		    t1 = time.time()
		    action = bandit[i].getBestAction()
		    t2 = time.time()
		    reward = bandit[i].takeBestAction(action)
		    t3 = time.time()
		    #queue_rewards.enqueue([reward]) 
		    ave_rewards[ind][t] += reward.eval()
		    t4 = time.time()
		    optimal = tf.equal(tf.cast(action,tf.int32),bandit[i].bestAction)
		    optimal = tf.cast(optimal,tf.int32)
		    #queue_actions.enqueue([optimal])
		    best_actions[ind][t] += optimal.eval()
		    t5 = time.time()
		    print 't2-t1',t2-t1
		    print 't3-t2',t3-t2
		    print 't4-t3',t4-t3
		    print 't5-t4',t5-t4
	    best_actions[ind] /= nbandit
	    ave_rewards[ind] /= nbandit
	    

	t_end = time.time()
	print 'consume time:',t_end-t_start
		    
	for eps, ba in zip(epsilon, best_actions):
	    plt.plot(ba, label='epsilon='+str(eps))
	plt.xlabel('steps')
	plt.ylabel('optimal action %')
	plt.legend()
	plt.show()

	for eps, ar in zip(epsilon, ave_rewards):
	    plt.plot(ar, label='epsilon='+str(eps))
	plt.xlabel('steps')
	plt.ylabel('average reward')
	plt.legend()
	plt.show()
