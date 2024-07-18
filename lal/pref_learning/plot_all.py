import numpy as np
from matplotlib import rcParams
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Palatino Linotype']
# rcParams['font.size'] = 40
rcParams["legend.loc"] = 'lower right'
import matplotlib.pyplot as plt

def plot_curve_and_std(ax, mean, std, label, color='#E48B10'):
    ax.plot(
        mean,
        color=color,
        linewidth=4,
        label=label)
    ax.fill_between(np.arange(0, len(mean), 1),
                    mean - 0.5 * std,
                    mean + 0.5 * std,
                    alpha=0.2,
                    color=color)

num_data = 4
active = 1
reward = 4
lang = 1
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
# colors = ['#AA3377', '#CCBB44', '#4477AA', '#228833', 'red', 'yellow']
# colors = ['#004488', '#DDAA33', '#BB5566', 'black']
# colors = ['#4477AA', '#228833', '#EE6677', '#BBBBBB']

mh_mh = [np.load(f"lal/pref_learning/true_rewards/1/pref_learning/{active}_{1}_{1}_noisy_{i+1}.npz") for i in range(num_data)]
gibbs_mh = [np.load(f"lal/pref_learning/true_rewards/1/pref_learning/{active}_{2}_{1}_noisy_{i+1}.npz") for i in range(num_data)]
laplace_mh = [np.load(f"lal/pref_learning/true_rewards/1/pref_learning/{active}_{3}_{1}_noisy_{i+1}.npz") for i in range(num_data)]
ep_mh = [np.load(f"lal/pref_learning/true_rewards/1/pref_learning/{active}_{4}_{1}_noisy_{i+1}.npz") for i in range(num_data)]
rs_comp = [np.load(f"lal/pref_learning/true_rewards/1/pref_learning/comp_noisy_{i}.npz") for i in range(3)]
rs_lang = [np.load(f"lal/pref_learning/true_rewards/1/pref_learning/lang_noisy_{i}.npz") for i in range(3)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

mh_mh_ce = [mh_mh[i]["eval_cross_entropies"] for i in range(num_data)]
gibbs_mh_ce = [gibbs_mh[i]["eval_cross_entropies"] for i in range(num_data)]
laplace_mh_ce = [laplace_mh[i]["eval_cross_entropies"] for i in range(num_data)]
ep_mh_ce = [ep_mh[i]["eval_cross_entropies"] for i in range(num_data)]
test_ce = [mh_mh[i]["test_ce"] for i in range(num_data)]
rs_comp_ce = [rs_comp[i]["eval_cross_entropies"][:25] for i in range(3)]
rs_lang_ce = [rs_lang[i]["eval_cross_entropies"][:25] for i in range(3)]

ax1.plot([0, len(mh_mh_ce[0])], [np.mean(test_ce), np.mean(test_ce)], 'k--', label='GT')
plot_curve_and_std(ax1, np.mean(mh_mh_ce, axis=0),
                np.std(mh_mh_ce, axis=0), "MH-MH",
                color=colors[0])
plot_curve_and_std(ax1, np.mean(gibbs_mh_ce, axis=0),
                np.std(gibbs_mh_ce, axis=0), "Gibbs-MH",
                color=colors[1])
plot_curve_and_std(ax1, np.mean(laplace_mh_ce, axis=0),
                np.std(laplace_mh_ce, axis=0), "LA-MH",
                color=colors[2])
plot_curve_and_std(ax1, np.mean(ep_mh_ce, axis=0),
                np.std(ep_mh_ce, axis=0), "EP-MH",
                color=colors[3])
plot_curve_and_std(ax1, np.mean(rs_comp_ce, axis=0),
                np.std(rs_comp_ce, axis=0), "Comp",
                color=colors[4])
plot_curve_and_std(ax1, np.mean(rs_lang_ce, axis=0),
                np.std(rs_lang_ce, axis=0), "Lang",
                color=colors[5])
ax1.set_xlabel("Number of Queries")
ax1.set_ylabel("Cross-Entropy")
ax1.set_title("Feedback, True Dist: Softmax")
ax1.legend()

mh_mh_rew = [mh_mh[i]["optimal_learned_rewards"]/mh_mh[i]["optimal_true_rewards"] for i in range(num_data)]
gibbs_mh_rew = [gibbs_mh[i]["optimal_learned_rewards"]/gibbs_mh[i]["optimal_true_rewards"] for i in range(num_data)]
laplace_mh_rew = [laplace_mh[i]["optimal_learned_rewards"]/laplace_mh[i]["optimal_true_rewards"] for i in range(num_data)]
ep_mh_rew = [ep_mh[i]["optimal_learned_rewards"]/ep_mh[i]["optimal_true_rewards"] for i in range(num_data)]
rs_comp_rew = [rs_comp[i]["optimal_learned_rewards"][:25]/rs_comp[i]["optimal_true_rewards"][:25] for i in range(3)]
rs_lang_rew = [rs_lang[i]["optimal_learned_rewards"][:25]/rs_lang[i]["optimal_true_rewards"][:25] for i in range(3)]

ax2.plot([0, len(mh_mh_rew[0])], [1.0, 1.0], 'k--', label='GT')
plot_curve_and_std(ax2, np.mean(mh_mh_rew, axis=0),
                np.std(mh_mh_rew, axis=0), "MH-MH",
                color=colors[0])
plot_curve_and_std(ax2, np.mean(gibbs_mh_rew, axis=0),
                np.std(gibbs_mh_rew, axis=0), "Gibbs-MH",
                color=colors[1])
plot_curve_and_std(ax2, np.mean(laplace_mh_rew, axis=0),
                np.std(laplace_mh_rew, axis=0), "LA-MH",
                color=colors[2])
plot_curve_and_std(ax2, np.mean(ep_mh_rew, axis=0),
                np.std(ep_mh_rew, axis=0), "EP-MH",
                color=colors[3])
plot_curve_and_std(ax2, np.mean(rs_comp_rew, axis=0),
                np.std(rs_comp_rew, axis=0), "Comp",
                color=colors[4])
plot_curve_and_std(ax2, np.mean(rs_lang_rew, axis=0),
                np.std(rs_lang_rew, axis=0), "Lang",
                color=colors[5])
ax2.set_xlabel("Number of Queries")
ax2.set_ylabel("Reward Value")
ax2.set_title("True Reward of Optimal Trajectory")
ax2.legend()

plt.tight_layout()
plt.savefig(f"lal/pref_learning/true_rewards/1/pref_learning/{active}_all_noisy_plots.png")