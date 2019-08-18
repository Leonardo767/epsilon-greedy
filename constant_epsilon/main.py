import numpy as np
from sim import Arms
from agent import Agent


def compute_agent_error(agent_model, true_model):
    m = len(agent_model)
    total = 0
    for Q_i, q_i in zip(agent_model, true_model):
        total += (q_i - Q_i)**2
    return round(total/m, 6)


def test_agent(testbed, agent, iters):
    avg_reward_hist = []
    iters = 20
    r_total = 0
    for t in range(1, iters):
        a_t = agent.choose_action()
        r_t = testbed.get_reward(a_t)
        agent.update_model(r_t)
        r_total += r_t
        avg_reward_hist.append(r_total/t)
    print(agent)
    return avg_reward_hist


np.random.seed(0)

number_arms = 10
testbed = Arms(number_arms)
iters_test = 100
print(testbed)

print('\nMODELS OF VARIOUS AGENTS:')
agent1 = Agent(number_arms, epsilon=0)
test_agent(testbed, agent1, iters_test)

agent2 = Agent(number_arms, epsilon=0.01)
test_agent(testbed, agent2, iters_test)

agent3 = Agent(number_arms, epsilon=0.1)
test_agent(testbed, agent3, iters_test)
