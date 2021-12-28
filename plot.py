from rlberry.experiment import load_experiment_results
from rlberry.stats import plot_fit_info, plot_episode_rewards, mc_policy_evaluation
from rlberry.agents.dynprog import ValueIterationAgent
import matplotlib.pyplot as plt

import argparse
import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 14
matplotlib.rcParams.update({'errorbar.capsize': 0})


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default='experiment',
    help="Experiment name, e.g. 'experiment' if you used the 'config/experiment.yaml' file.")

# ------------------------------------------
# Load results
# ------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()
    EXPERIMENT_NAME = args.experiment_name


    PLOT_TITLES = {
        'optql': 'OptQL',
        'ucbmq': 'UCBMQ',
        'ucbvi': 'UCBVI',
        'ucbvi_rtdp': 'Greedy-UCBVI',
    }

    output_data = load_experiment_results('results', EXPERIMENT_NAME)


    # Get list of AgentStats
    _stats_list = list(output_data['stats'].values())
    stats_list = []
    for stats in _stats_list:
        if stats.agent_name in PLOT_TITLES:
            stats.agent_name = PLOT_TITLES[stats.agent_name]
            stats_list.append(stats)


    # Get value of optimal agent
    env = stats_list[0].eval_env
    horizon = stats_list[0].fitted_agents[0].horizon
    vi_agent = ValueIterationAgent(env, gamma=1.0, horizon=horizon)
    vi_agent.fit()
    rewards = mc_policy_evaluation(vi_agent, env, horizon, n_sim=20, stationary_policy=False)
    max_value = rewards.mean()
    print("max value = ", max_value)

    # -------------------------------
    # Plot and save
    # -------------------------------
    plot_fit_info(stats_list, "episode_rewards", show=False)
    plot_episode_rewards(stats_list, cumulative=True, show=False)


    # matplotlib.rcParams['text.usetex'] = True
    plot_episode_rewards(stats_list,
                        cumulative=False,
                        show=False,
                        max_value=max_value,
                        plot_regret=True,
                        grid=False)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel("episode", labelpad=0)

    # show save all figs
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for ii, fig in enumerate(figs):
        fname = output_data['experiment_dirs'][0] / 'fig_{}.pdf'.format(ii)
        fig.savefig(fname, format='pdf', bbox_inches='tight')

    plt.show()
