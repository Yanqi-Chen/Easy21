import numpy as np
from itertools import count
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easy21 import Easy21


def MC_control():
	stick_Qvalue = np.zeros((21, 10))
	hit_Qvalue = np.zeros((21, 10))
	cnt_stick = np.zeros((21, 10))
	cnt_hit = np.zeros((21, 10))
	cnt_state = np.zeros((21, 10))
	N = 100.0

	num_episodes = 100000

	for i_episode in range(num_episodes):
		s, _, done = env.reset()
		R = 0
		for t in count():
			ss = tuple(s - 1)
			cnt_state[ss] += 1
			# epsilon exploration
			e = N / (N + cnt_state[ss])
			if np.random.random_sample() < e:
				if np.random.random_sample() < 0.5:
					a = 'hit'
				else:
					a = 'stick'
			elif stick_Qvalue[ss] > hit_Qvalue[ss]:
				a = 'stick'
			else:
				a = 'hit'
			s_, r, done = env.step(a)
			R += r
			if a == 'hit':
				cnt_hit[ss] += 1
				hit_Qvalue[ss] += 1.0 / cnt_hit[ss] * (R - hit_Qvalue[ss])
			else:
				cnt_stick[ss] += 1
				stick_Qvalue[ss] += 1.0 / cnt_stick[ss] * (R - stick_Qvalue[ss])

			s = s_
			if done:
				break

	#print(stick_Qvalue, '\n',
		#hit_Qvalue)
	X = np.arange(1, 22)
	Y = np.arange(1, 11)
	XX, YY = X, Y = np.meshgrid(X, Y)
	fig = plt.figure(dpi=200)
	ax = fig.gca(projection='3d')
	#print(stick_Qvalue.ndim)

	#ax.plot_wireframe(XX, YY, stick_Qvalue.T,colors='r')
	#ax.plot_wireframe(XX, YY, hit_Qvalue.T,colors='b')
	res = np.max(np.array([stick_Qvalue, hit_Qvalue]), axis=0)
	ax.plot_surface(XX, YY, res.T, cmap='rainbow')

	ax.set_zlabel('Value')  # 坐标轴
	ax.set_xlabel('Player sum')
	ax.set_ylabel('Dealer showing')
	ax.view_init(azim=225)
	ax.set_xlim(1, 21)
	ax.set_ylim(1, 10)
	plt.savefig('MC')
	plt.show()


if __name__ == '__main__':
	matplotlib.use('Agg')
	env = Easy21()
	MC_control()
