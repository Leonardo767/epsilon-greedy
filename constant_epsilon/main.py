import numpy as np
from sim import Arms
from agent import Agent
from plot import (plot_avg_performance, compare_avg_performance, plot_model)


def compute_agent_error(agent_model, true_model):
    m = len(agent_model)
    total = 0
    for Q_i, q_i in zip(agent_model, true_model):
        total += (q_i - Q_i)**2
    return round(total/m, 6)


def test_agent(testbed, agent, iters):
    avg_reward_hist = []
    model_log = []
    r_total = 0
    for t in range(1, iters):
        a_t = agent.choose_action()
        r_t = testbed.get_reward(a_t)
        agent.update_model(r_t)
        r_total += r_t
        avg_reward_hist.append(r_total/t)
        model_log.append(agent.model)
    return avg_reward_hist, agent.model_hist


np.random.seed(10)

number_arms = 10
testbed = Arms(number_arms)
iters_test = 10000
print(testbed)

print('\nMODELS OF VARIOUS AGENTS:')
agent1 = Agent(number_arms, epsilon=0, Q_0=-3)  # -1 epsilon will never explore
reward1, model1 = test_agent(testbed, agent1, iters_test)

agent2 = Agent(number_arms, epsilon=0.0, Q_0=-3)
reward2, model2 = test_agent(testbed, agent2, iters_test)

agent3 = Agent(number_arms, epsilon=0.1)
reward3, model3 = test_agent(testbed, agent3, iters_test)
# plot_model(model3, iters_test, testbed.arms)

agent4 = Agent(number_arms, epsilon=1)
reward4, model4 = test_agent(testbed, agent4, iters_test)

compare_avg_performance([agent1, agent2, agent3, agent4], iters_test,
                        [reward1, reward2, reward3, reward4],
                        ['red', 'blue', 'green', 'black'])
