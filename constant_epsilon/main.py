import numpy as np
from sim import Arms
from agent import Agent


def compute_agent_error(agent_model, true_model):
    m = len(agent_model)
    total = 0
    for Q_i, q_i in zip(agent_model, true_model):
        total += (q_i - Q_i)**2
    return round(total/m, 6)


np.random.seed(0)

number_arms = 3
arms = Arms(number_arms)
agent = Agent(number_arms)
print(arms)

iters = 100000
r_total = 0
for t in range(iters):
    a_t = agent.choose_action()
    r_t = arms.get_reward(a_t)
    agent.update_model(r_t)
    r_total += r_t
    # print('r({}): {}'.format(t, r_t))

print('\n...LEARNING COMPLETE...')
print(agent)
print('PERFORMANCE: {}'.format(r_total/iters))
print('ERROR: {}'.format(compute_agent_error(agent.model, arms.arms)))
print()
