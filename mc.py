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


def MC_control(env):
	Qvalue = np.zeros((21, 10, 2))
	cnt_state = np.zeros((21, 10, 2))
	N = 100.0

	num_episodes = 1000000
	mean_G = 0
	wins = 0
	for i_episode in range(num_episodes):
		s, _, done = env.reset()
		G = 0
		sar_list = []
		while not done:
			a = choose_action(Qvalue, N, s)

			sa = np.append(s - 1, a)
			cnt_state[tuple(sa)] += 1

			s_, r, done = env.step(a)
			sar_list.append([s, a, r])
			#print(env.player_sum, env.dealer_sum, a, r)
			G += r
			s = s_

		#print("done")
		mean_G += (G - mean_G) / (i_episode + 1)
		if G == 1:
			wins += 1
		if i_episode % 10000 == 0:
			print("%d/%d, mean reward: %.3f, win: %.3f" % (i_episode // 10000,
                                                  num_episodes // 10000, mean_G, wins / (i_episode + 1)))

		for s0, a0, r0 in sar_list:
			sa = tuple(np.append(s0 - 1, a0))
			alpha = 1.0 / cnt_state[sa]
			# Q(s, a) = Q(s, a) + 1 / N(s, a) * (G - Q(s, a))
			Qvalue[sa] += alpha * (G - Qvalue[sa])

	# save Qvalue
	np.save("q.npy", Qvalue)
	
	print(Qvalue)
	X = np.arange(1, 22)
	Y = np.arange(1, 11)
	XX, YY = X, Y = np.meshgrid(X, Y)
	fig = plt.figure(figsize=(8, 4), dpi=200)
	ax = fig.gca(projection='3d')

	res = np.max(Qvalue, axis=2)
	ax.plot_surface(YY, XX, res.T, cmap='hot')

	ax.set_zlabel('Value')
	ax.set_xlabel('Dealer showing')
	ax.set_ylabel('Player sum')
	ax.set_ylim(1, 21)
	ax.set_yticks(list(range(1, 22)))
	ax.set_xlim(1, 10)
	ax.view_init(azim=-30)
	plt.savefig('MC')
	plt.show()


if __name__ == '__main__':
	matplotlib.use('Agg')
	env = Easy21()
	MC_control(env)
