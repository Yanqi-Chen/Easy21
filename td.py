import numpy as np
from itertools import count
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easy21 import Easy21


def choose_action(Qvalue, N, s):
	# epsilon greedy exploration
	ss = tuple(s - 1)
	e = N / (N + np.sum(Qvalue[ss]))
	if np.random.random_sample() < e:
		if np.random.random_sample() < 0.5:
			return 1
		else:
			return 0
	return np.argmax(Qvalue[ss])


def TD_learning(env):

	GAMMA = 1
	LAMBDAS = list(np.arange(0, 1.1, 0.1))
	#LAMBDA = list(range(0, 1.1, 0.1))
	N = 100.0
	num_episodes = 10000
	realQvalue = np.load("q.npy")
	MSEs = []

	for LAMBDA in LAMBDAS:
		Qvalue = np.zeros((21, 10, 2))
		cnt_state = np.zeros((21, 10, 2))

		mean_G = 0
		wins = 0
		record = LAMBDA == 0 or LAMBDA == 1
		record_MSEs = []
		for i_episode in range(num_episodes):
			s, _, done = env.reset()
			a = choose_action(Qvalue, N, s)
			G = 0
			E = np.zeros((21, 10, 2))
			sa_set = set()
			while not done:
				sa = np.append(s - 1, a)
				cnt_state[tuple(sa)] += 1

				s_, r, done = env.step(a)

				if not done:
					a_ = choose_action(Qvalue, N, s_)
					s_a_ = np.append(s_ - 1, a_)
					TD_error = r + GAMMA * Qvalue[tuple(s_a_)] - Qvalue[tuple(sa)]
				else:
					TD_error = r - Qvalue[tuple(sa)]
				E[tuple(sa)] += 1
				cnt_state[tuple(sa)] += 1
				sa_set.add(tuple(sa))

				G += r
				for sa0 in sa_set:
					alpha = 1.0 / cnt_state[sa0]
					Qvalue[sa0] += alpha * TD_error * E[sa0]
					E[sa0] = GAMMA * LAMBDA * E[sa0]
				if not done:
					s, a = s_, a_

			#print("done")
			mean_G += (G - mean_G) / (i_episode + 1)
			if G == 1:
				wins += 1
			if i_episode % 10000 == 0:
				print("%d/%d, mean reward: %.3f, win: %.3f" % (i_episode // 10000,
                                                   num_episodes // 10000, mean_G, wins / (i_episode + 1)))
			if record:
				record_MSEs.append(((Qvalue - realQvalue) ** 2).mean())

		# MSE
		MSEs.append(((Qvalue - realQvalue) ** 2).mean())

		print(Qvalue)
		# X = np.arange(1, 22)
		# Y = np.arange(1, 11)
		# XX, YY = X, Y = np.meshgrid(X, Y)
		# fig = plt.figure(dpi=200)
		# ax = fig.gca(projection='3d')

		# res = np.max(Qvalue, axis=2)
		# ax.plot_surface(XX, YY, res.T, cmap='hot')

		# ax.set_zlabel('Value')
		# ax.set_xlabel('Player sum')
		# ax.set_ylabel('Dealer showing')
		# ax.view_init(azim=225)
		# ax.set_xlim(1, 21)
		# ax.set_ylim(10, 1)
		# plt.savefig('MC')
		# plt.show()
		if record:
			fig2 = plt.figure(dpi=200)
			plt.plot(list(range(num_episodes)), record_MSEs)
			plt.savefig('MSE-Lambda={}.png'.format(LAMBDA))
			plt.show()

	fig1 = plt.figure(dpi=200)
	plt.plot(LAMBDAS, MSEs)
	plt.savefig('MSE-Lambda-{}'.format(num_episodes))
	plt.show()


if __name__ == '__main__':
	matplotlib.use('Agg')
	env = Easy21()
	TD_learning(env)
