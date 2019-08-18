# contains the agent which plays the game
import numpy as np


class Agent():
    def __init__(self, observed_size, epsilon=0.03, Q_0=0):
        # agent characteristics:
        self.epsilon = epsilon
        # -----------------------------------
        # memory:
        self.k = 0  # internal memory of iteration
        # list of expected values corresponding with arm id:
        self.model = [Q_0]*observed_size
        # list of tallied rewards corresponding with arm id:
        self.tally = [0]*observed_size
        # -----------------------------------
        # model logger
        self.model_hist = [self.model]

    def __repr__(self):
        out = ["Agent where epsilon = {}\n".format(self.epsilon)]
        model_to_display = []
        for i in range(len(self.model)):
            model_to_display.append(round(self.model[i], 3))
        out.append("Current model: {}".format(model_to_display))
        return "".join(out)

    def choose_action(self):
        if np.random.uniform() < self.epsilon or self.k == 0:
            # explore
            a_t = np.random.randint(0, len(self.model))
        else:
            # exploit
            a_t = np.argmax(self.model)
        # remember the action that the agent took
        self.a_t = a_t
        return a_t

    def update_model(self, r_t):
        self.k += 1  # clock always ticks up
        self.tally[self.a_t] += r_t  # update reward tally
        # gather info to update model incrementally
        k = self.k
        a_t = self.a_t
        Q_k = self.model[a_t]
        Q_k_new = Q_k + 1/k*(r_t - Q_k)
        # print('updating from {} to {} for arm {}'.format(Q_k, Q_k_new, a_t))
        self.model[a_t] = Q_k_new
        # log new model
        self.model_hist.append(self.model.copy())
