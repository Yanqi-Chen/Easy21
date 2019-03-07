import numpy as np
from itertools import count
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easy21 import Easy21

EPSILON = 0.05
GAMMA = 1

def feature_id(s, a):
	sf = []
	for i, (l, r) in enumerate(zip(range(1, 8, 3), range(4, 11, 3))):
		if s[1] >= l and s[1] <= r:
			sf.append(i)
			break

	for i, (l, r) in enumerate(zip(range(1, 17, 3), range(6, 22, 3))):
		if s[0] >= l and s[0] <= r:
			sf.append(i)
			break

	for i, j in enumerate(range(2)):
		if a == j:
			sf.append(i)
			break
	return tuple(sf)


def feature(s, a):
	sf = feature_id(s, a)
	m = np.zeros((3, 6, 2))
	m[tuple(sf)] = 1
	return m


def Qvalue(s, a, theta):
	return np.sum(feature(s, a) * theta)


def choose_action(s, theta):
	# epsilon greedy exploration
	if np.random.random_sample() < EPSILON:
		if np.random.random_sample() < 0.5:
			return 1
		else:
			return 0
	lst = list(map(lambda a: Qvalue(s, a, theta), [0, 1]))
	return np.argmax(lst)


def TD_learning(env):

	LAMBDAS = list(np.arange(0, 1.1, 0.1))
	#LAMBDA = list(range(0, 1.1, 0.1))
	#N = 100.0
	num_episodes = 10000
	realQvalue = np.load("q.npy")
	MSEs = []

	for LAMBDA in LAMBDAS:
		theta = np.zeros((3, 6, 2))
		QvalueM = np.zeros((21, 10, 2))
		#Qvalue = np.zeros((21, 10, 2))
		#cnt_state = np.zeros((21, 10, 2))

		mean_G = 0
		wins = 0
		record = LAMBDA == 0 or LAMBDA == 1
		record_MSEs = []
		for i_episode in range(num_episodes):
			s, _, done = env.reset()
			a = choose_action(s, theta)
			G = 0
			while not done:
				#cnt_state[tuple(sa)] += 1

				s_, r, done = env.step(a)

				if not done:
					a_ = choose_action(s_, theta)
					TD_error = r + GAMMA * Qvalue(s_, a_, theta) - Qvalue(s, a, theta)
				else:
					TD_error = r - Qvalue(s, a, theta)
				#cnt_state[tuple(sa)] += 1

				G += r

				#alpha = 1.0 / cnt_state[sa0]
				F = feature(s, a)
				alpha = 0.01
				theta += alpha * TD_error * F

				if not done:
					s, a = s_, a_

			#print("done")
			mean_G += (G - mean_G) / (i_episode + 1)
			if G == 1:
				wins += 1
			if i_episode % 1000 == 0:
				print("%d/%d, mean reward: %.3f, win: %.3f" % (i_episode // 1000,
                                                   num_episodes // 1000, mean_G, wins / (i_episode + 1)))
				print(theta)
			if record:
				for i in range(21):
					for j in range(10):
						for k in range(2):
							QvalueM[i][j][k] = Qvalue([i + 1, j + 1], k, theta)
				record_MSEs.append(((QvalueM - realQvalue) ** 2).mean())

		# MSE
		for i in range(21):
			for j in range(10):
				for k in range(2):
					QvalueM[i][j][k] = Qvalue([i + 1, j + 1], k, theta)
		MSEs.append(((QvalueM - realQvalue) ** 2).mean())

		# plot
		X = np.arange(1, 22)
		Y = np.arange(1, 11)
		XX, YY = X, Y = np.meshgrid(X, Y)
		fig = plt.figure(dpi=200)
		ax = fig.gca(projection='3d')
		res = np.max(QvalueM, axis=2)
		ax.plot_surface(XX, YY, res.T, cmap='hot')
		ax.set_zlabel('Value')
		ax.set_xlabel('Player sum')
		ax.set_ylabel('Dealer showing')
		ax.view_init(azim=225)
		ax.set_xlim(1, 21)
		ax.set_ylim(10, 1)
		plt.show()

		print(QvalueM)

		if record:
			fig2 = plt.figure(dpi=200)
			plt.plot(list(range(num_episodes)), record_MSEs)
			plt.savefig('MSE-linear-Lambda={}.png'.format(LAMBDA))
			plt.show()

	fig1 = plt.figure(dpi=200)
	plt.plot(LAMBDAS, MSEs)
	plt.savefig('MSE-linear-Lambda-{}'.format(num_episodes))
	plt.show()


if __name__ == '__main__':
	matplotlib.use('Agg')
	env = Easy21()
	TD_learning(env)
