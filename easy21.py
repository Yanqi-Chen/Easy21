import numpy as np


def isBust(p_sum):
	return p_sum > 21 or p_sum < 1

def drawCard():
	black_flag = 1 if np.random.random_sample() > (1.0 / 3) else -1
	return np.random.randint(1, 11) * black_flag

class Easy21(object):

	def __init__(self):
		self.states = np.arange(-9, 31)
		self.actions = { 'stick': 0, 'hit': 1 }

	def reset(self):
		self.done = False
		self.player_sum = np.random.randint(1, 11)
		self.dealer_card = np.random.randint(1, 11)
		self.dealer_sum = self.dealer_card
		self.state = np.array([self.player_sum, self.dealer_card])
		self.player_stick_flag = False
		self.dealer_stick_flag = False
		return self.state, 0, False

	def step(self, action):
		reward = 0
		if action == 1:
			self.player_sum += drawCard()
		elif action == 0:
			self.player_stick_flag = True
		# dealer AI
		if not self.dealer_stick_flag:
			if self.dealer_sum >= 17:
				self.dealer_stick_flag = True
			else:
				self.dealer_sum += drawCard()

		inner_state = [self.player_sum, self.dealer_sum]

		# result
		player_bust, dealer_bust = map(isBust, inner_state)
		self.done = True
		if player_bust and dealer_bust:
			reward = 0
		elif player_bust and not dealer_bust:
			reward = -1
		elif not player_bust and dealer_bust:
			reward = 1
		else:
			if self.player_stick_flag and self.dealer_stick_flag:
				reward = np.sign(self.player_sum - self.dealer_sum)
			elif self.player_stick_flag and not self.dealer_stick_flag:
				while self.dealer_sum < 17 and not isBust(self.dealer_sum):
					self.dealer_sum += drawCard()
				if isBust(self.dealer_sum):
					reward = 1
				else:
					reward = np.sign(self.player_sum - self.dealer_sum)
			else:
				self.done = False

		self.state = np.array([self.player_sum, self.dealer_card])
		return self.state, reward, self.done
