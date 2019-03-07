import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easy21 import Easy21

GAMMA = 0.9
BATCH_SIZE = 128
LR = 0.01		# Learning rate
EPSILON = 0.1	# Greedy
N_STATES = 2
N_ACTIONS = 2
MEMORY_CAPACITY = 5000
TARGET_REPLACE_ITER = 100   # target update frequency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(N_STATES, 20)
		self.fc1.weight.data.normal_(0, 0.1)
		self.fc2 = nn.Linear(20, 50)
		self.fc2.weight.data.normal_(0, 0.1)
		self.out = nn.Linear(50, N_ACTIONS)
		self.out.weight.data.normal_(0, 0.1)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.out(x)
		x = 2 * F.softmax(x, dim=1) - 1
		return x

class DQN(object):
	def __init__(self):
		self.eval_net, self.target_net = Net().to(device), Net().to(device)

		self.learn_step_counter = 0                                     
		self.memory_counter = 0                                         
		self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))    
		self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=LR)
		self.loss_func = nn.MSELoss()

	def choose_action(self, s):
		s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
		if np.random.random() < EPSILON:
			# Exploration
			return np.random.randint(0, 2)
		# Greedy, exploitation
		a_values = self.eval_net.forward(s)
		a = torch.argmax(torch.squeeze(a_values))
		return a

	def store_transition(self, s, a, r, s_):
		transition = np.hstack((s, [a, r], s_))
		#print(s, a, r, s_, transition)
		# replace the old memory with new memory
		index = self.memory_counter % MEMORY_CAPACITY
		self.memory[index, :] = transition
		self.memory_counter += 1

	def learn(self):
		# target parameter update
		if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1

		# sample batch transitions
		sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
		b_memory = self.memory[sample_index, :]
		b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
		b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
		b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
		b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

		# q_eval w.r.t the action in experience
		q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
		q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
		q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
		loss = self.loss_func(q_eval, q_target)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def save_model(self):
		torch.save(self.eval_net.state_dict(), 'params.pkl')

def DQN_learning(env):
	num_episodes = 100000
	realQvalue = np.load("q.npy")
	dqn = DQN()
	wins = 0
	draws = 0
	for i_episode in range(num_episodes + 1):
		s, _, _ = env.reset()
		ep_r = 0
		while True:

			a = dqn.choose_action(s)

			# take action
			s_, r, done = env.step(a)

			dqn.store_transition(s, a, r, s_)

			ep_r += r
			if dqn.memory_counter > MEMORY_CAPACITY:
				dqn.learn()
				if done:
					if ep_r == 1:
						wins += 1
					elif ep_r == 0:
						draws += 1
					if i_episode % 1000 == 0:
						print('Ep: ', i_episode,
							'| Ep_r: ', round(ep_r, 2), 'win:draw:lose={}:{}:{}'.format(wins, draws, 1000 - wins - draws))
						wins = draws = 0

			if done:
				if i_episode % 10000 == 0:
					M = torch.zeros((21, 10, 2)).to(device)
					for i in range(21):
						for j in range(10):
							s = [i + 1, j + 1]
							s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
							a_values = dqn.eval_net.forward(s).detach()
							M[i][j] = torch.squeeze(a_values)
					QvalueM = M.detach().to('cpu').numpy()
					print('MSE: ', ((QvalueM - realQvalue) ** 2).mean())
					if i_episode % 20000 == 0:
						X = np.arange(1, 22)
						Y = np.arange(1, 11)
						XX, YY = X, Y = np.meshgrid(X, Y)
						fig = plt.figure(dpi=200)
						ax = fig.gca(projection='3d')
						res = np.max(QvalueM, axis=2)
						ax.plot_surface(YY, XX, res.T, cmap='hot')

						ax.set_zlabel('Value')
						ax.set_xlabel('Dealer showing')
						ax.set_ylabel('Player sum')
						ax.set_ylim(1, 21)
						ax.set_yticks(list(range(1, 22)))
						ax.set_xlim(1, 10)
						ax.view_init(azim=-30)
						if i_episode == num_episodes:
							plt.savefig('DQN.png')
						plt.show()
				break
			s = s_
	dqn.save_model()
	#MSEs.append(((QvalueM - realQvalue) ** 2).mean())


if __name__ == '__main__':
	matplotlib.use('Agg')
	env = Easy21()
	DQN_learning(env)