# -*- coding:utf-8 -*-


'''
  n-Armed Bandit problem implemented in TensorFlow.

zzw922cn
2017-2-3
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
	self.stepSize = stepSize
        self.sampleAverage = sampleAverage
	self.softmax = softmax
	self.sess = session
	self.epoch = 0
	self.meanReward = 0
	self.actionCount = [0]*self.num_arms
	
	idx = str(idx)+str(epsilon)+str(stepSize)+str(sampleAverage)+str(softmax)

        self.armsIndex = tf.get_variable(str(idx)+'/armsIndex', shape=[self.num_arms], initializer=tf.constant_initializer(np.arange(self.num_arms)), dtype=tf.int32)

        self.estimatedValue = tf.get_variable(str(idx)+'/estimatedValue', shape=[self.num_arms], initializer=tf.constant_initializer([self.init_value]*self.num_arms), dtype=tf.float32)

        self.trueValue = tf.get_variable(str(idx)+'/trueValue', shape=[self.num_arms], initializer=tf.random_normal_initializer(), dtype=tf.float32)

	self.bestAction = tf.argmax(self.trueValue, axis=0)

        #self.value_estimated = subset_update(self.value_estimated, [0,1,2], [1,3,5], mode='sub', scope='init')

        self.initial_op = tf.global_variables_initializer()

    def getBestAction(self):
	# return the shuffled array

	## 1.exploration
	if self.epsilon > 0:
	    if np.random.binomial(1, self.epsilon) == 1:
	        return tf.random_shuffle(self.armsIndex)
	## 2.softmax exploration
	if self.softmax:
	    self.softmaxProb = tf.nn.softmax(self.estimatedValue) 
	    samples = tf.multinomial(tf.reshape(self.softmaxProb,[1,-1]), self.num_arms) 
	    samples = tf.cast(samples, tf.int32)
	    samples = tf.reshape(samples, [self.num_arms])
	    return samples
	return tf.reshape(tf.argmax(self.estimatedValue, axis=0),[1]) 

    def takeBestAction(self, action):
	# reward is generated with a random added by true value
	rand_reward = np.random.rand()
	reward = rand_reward + self.trueValue.eval()[action]
	self.epoch += 1
	self.meanReward = (self.epoch-1.0)/self.epoch*self.meanReward + reward/self.epoch
	self.actionCount[action] += 1
        c_action = tf.constant([action])
	
	if self.sampleAverage:
	    c_new = tf.constant([1.0/self.actionCount[action]*(reward-self.estimatedValue.eval()[action])])
            self.estimatedValue = tf.scatter_add(self.estimatedValue, c_action, c_new)

	elif self.softmax:
	    ones = [0.0]*self.num_arms
	    ones[action] = 1.0
	    ones = ones - self.softmaxProb.eval()
	    c_new = tf.constant(self.stepSize*(reward-self.meanReward)*ones,dtype=tf.float32)
	    c_index = tf.constant(range(10))
            self.estimatedValue = tf.scatter_add(self.estimatedValue, c_index, c_new)
	    
	else:
	    c_new = tf.constant([self.stepSize*(reward-self.estimatedValue.eval()[action])]) 
            self.estimatedValue = tf.scatter_add(self.estimatedValue, c_action, c_new)
	    
	return reward
    
if __name__ == '__main__':
    with tf.Session() as sess:
	bandits = []
	nbandit = 20
	times = 100
	epsilon = [0, 0.01, 0.1, 0.15]
	for eps in epsilon:
	    bandit = [
                       ArmedBandit(10, sess, idx=idx, epsilon=eps, sampleAverage=True, softmax=False) \
		       for idx in range(nbandit) \
		     ]
	    for ab in bandit:
        	sess.run(ab.initial_op)
	    bandits.append(bandit)
	
	ave_rewards = [np.zeros(times, dtype='float') for _ in range(len(epsilon))]
	best_actions = [np.zeros(times, dtype='float') for _ in range(len(epsilon))]
	for ind, bandit in enumerate(bandits):
	    for i in range(nbandit):
		for t in range(times):
		    action = bandit[i].getBestAction().eval()[0]
		    reward = bandit[i].takeBestAction(action)
		    print action, reward
		    ave_rewards[ind][t] += reward
		    if action == bandit[i].bestAction.eval():
			best_actions[ind][t] += 1
	    best_actions[ind] /= nbandit
	    ave_rewards[ind] /= nbandit
		    
	for eps, ba in zip(epsilon, best_actions):
	    plt.plot(ba, label='epsilon='+str(eps))
	plt.xlabel('steps')
	plt.ylabel('best action')
	plt.show()
