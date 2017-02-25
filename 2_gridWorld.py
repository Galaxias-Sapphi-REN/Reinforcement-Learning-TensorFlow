#coding=utf-8
import numpy as np
import argparse
from utils import printboard
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'

# 命令行接收参数
parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=int, default=5,
                       help='set size of grid stateValue')
parser.add_argument('--discount', type=float, default=0.9,
                       help='set discount factor of future expected reward')
parser.add_argument('--posA', type=list, default=[0, 1],
                       help='set position of A')
parser.add_argument('--posB', type=list, default=[0, 3],
                       help='set position of B')
parser.add_argument('--primeA', type=list, default=[4, 1],
                       help='set prime position of A')
parser.add_argument('--primeB', type=list, default=[2, 3],
                       help='set prime position of B')
args = parser.parse_args()

grid_size = args.grid_size
posA = args.posA
primeA = args.primeA
posB = args.posB
primeB = args.primeB
discount = args.discount




# 四种可能的决策
# left, up, right, down
actions = ['L', 'U', 'R', 'D']

actionProb = []

# 定义好每个状态处的行动概率，即均匀随机决策
for i in range(0, grid_size):
    actionProb.append([])
    for j in range(0, grid_size):
        actionProb[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))

# 定义好每个状态处的状态转移，以及回报值
successorState = []
actionReward = []
for i in range(0, grid_size):
    successorState.append([])
    actionReward.append([])
    for j in range(0, grid_size):
        next = dict()
        reward = dict()

	# 其他元胞上下左右以及注意边缘处行动要保持不变
        if i == 0:
            next['U'] = [i, j]
            reward['U'] = -1.0
        else:
            next['U'] = [i - 1, j]
            reward['U'] = 0.0

        if i == grid_size - 1:
            next['D'] = [i, j]
            reward['D'] = -1.0
        else:
            next['D'] = [i + 1, j]
            reward['D'] = 0.0

        if j == 0:
            next['L'] = [i, j]
            reward['L'] = -1.0
        else:
            next['L'] = [i, j - 1]
            reward['L'] = 0.0

        if j == grid_size - 1:
            next['R'] = [i, j]
            reward['R'] = -1.0
        else:
            next['R'] = [i, j + 1]
            reward['R'] = 0.0

	# 定义A、B处的状态转移以及回报值
	# 同时，将上面边缘处的限制覆盖
	if [i, j] == posA:
            next['L'] = next['R'] = next['D'] = next['U'] = primeA
            reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0

        if [i, j] == posB:
            next['L'] = next['R'] = next['D'] = next['U'] = primeB
            reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0

        successorState[i].append(next)
        actionReward[i].append(reward)

stateValue = np.zeros((grid_size, grid_size))

while True:
    newStateValue = np.zeros((grid_size, grid_size))
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            for action in actions:
                newPosition = successorState[i][j][action]
                newStateValue[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * stateValue[newPosition[0], newPosition[1]])
    if np.sum(np.abs(stateValue - newStateValue)) < 1e-4:
        print('均匀随机策略:状态-价值')
        print(newStateValue)
        break
    stateValue = newStateValue

plt.matshow(newStateValue, cmap=plt.cm.Greys)
plt.colorbar()
plt.title('state value')
plt.show()


stateValue = np.zeros((grid_size, grid_size))
while True:
    newStateValue = np.zeros((grid_size, grid_size))
    a = ''
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            values = []
            for action in actions:
                newPosition = successorState[i][j][action]
                values.append(actionReward[i][j][action] + discount * stateValue[newPosition[0], newPosition[1]])
            newStateValue[i][j] = np.max(values)
	    # 找出所有最优状态-价值的行动
	    indexes = np.argwhere(values == np.max(values))
	    indexes = indexes.flatten().tolist()
	    policyOfState = ''.join([actions[index] for index in indexes])
	    a += policyOfState+' '

    if np.sum(np.abs(stateValue - newStateValue)) < 1e-4:
        a = a.strip().split(' ')
        policy = np.reshape(a,[grid_size, grid_size])
        print('均匀随机策略:最优状态-价值')
	print(newStateValue)
	printboard(policy)
        break
    stateValue = newStateValue

plt.matshow(newStateValue, cmap=plt.cm.Greys)
plt.colorbar()
plt.title('optimal state value')
plt.show()
