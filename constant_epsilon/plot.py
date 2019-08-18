import numpy as np
from bokeh.plotting import figure, output_file, show


def plot_model(agent_model_hist, iters, true_model, name='agent1'):
    p = figure(title="Internal model evolution for {}".format(name))
    t = np.arange(0, iters)
    agent_model_hist = np.array(agent_model_hist)
    agent_model_hist = np.transpose(agent_model_hist)
    arm_id = 0
    for arm in agent_model_hist:
        p.line(t, arm, legend=str(arm_id))
        arm_id += 1
    output_file("plot_model_{}.html".format(name))
    show(p)
    return


def plot_avg_performance(agent, iters, reward_hist):
    p = figure(title='Epsilon = {}'.format(agent.epsilon))
    t = np.arange(0, iters - 1)
    r_avg = np.array(reward_hist)
    p.line(t, r_avg)
    output_file("plot_avg_perf.html")
    show(p)
    return


def compare_avg_performance(agent_list, iters, reward_hist_list, colors):
    p = figure(title='Comparison of Performance')
    t = np.arange(0, iters - 1)
    for i in range(len(agent_list)):
        r_avg = np.array(reward_hist_list[i])
        name = 'epsilon = ' + \
            str(agent_list[i].epsilon) + ';\tQ_0 = ' + str(agent_list[i].Q_0)
        p.line(
            t, r_avg, line_color=colors[i], legend=name)
    p.legend.location = 'bottom_right'
    output_file("compare_avg_perf.html")
    show(p)
    return
