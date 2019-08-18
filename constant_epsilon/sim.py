# contains the environment in which the game is played
import numpy as np


class Arms():
    def __init__(self, number_of_arms, seed=0):
        self.seed = seed
        self.size = number_of_arms
        self.arms = []
        for i in range(number_of_arms):
            value = round(np.random.normal(), 3)
            self.arms.append(value)

    def __repr__(self):
        out = ['Environment with {} arms:\n'.format(self.size)]
        out.append("True Environment: {}".format(self.arms))
        return "".join(out)

    def get_reward(self, state_selected):
        # return action value + noise
        return self.arms[state_selected] + np.random.normal()
