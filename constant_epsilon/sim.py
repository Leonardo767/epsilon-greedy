# contains the environment in which the game is played
import numpy as np


class Arms():
    def __init__(self, number_of_arms, seed=0):
        self.seed = seed
        self.size = number_of_arms
        self.arms = []
        for i in range(number_of_arms):
            self.arms.append(round(np.random.normal(0.5, 0.1), 3))

    def __repr__(self):
        out = ['Environment with {} arms:\n'.format(self.size)]
        out.append("True Environment: {}".format(self.arms))
        return "".join(out)

    def get_reward(self, state_selected):
        # gather info about selection from environment
        prob_success = self.arms[state_selected]
        # simulate expected value
        if np.random.uniform() < prob_success:
            return 1
        return 0
