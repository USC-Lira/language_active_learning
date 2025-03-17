import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Palatino Linotype']
rcParams['font.size'] = 12
rcParams['legend.loc'] = 'best'

print(f"Current font family: {rcParams['font.family']}")
print(f"Current serif font: {rcParams['font.serif']}")

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

sim = [0, 1, 2, 3]
# sim = [0, 1, 2, 3, 4]
# sim = [0]
num_data = len(sim)
num_seeds = [0, 1, 2]
# num_seeds = [0]
# colors = ['#082a54', '#f0c571', '#59a89c', '#a559aa', '#cecece', '#e02b35', 'blue', 'red']
colors = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#000000']

# test1.png, looks good except active does worse than lang, uses argmax for info gain
# name = "rs_new_t5_base-both-norm_general_ALL_active"
la_la = [np.load(f"true_rewards_rs_t5-base-test/{i}/pref_learning/1_3_3_dupe_traj_-1_seed_{j}.npz") for i in sim for j in num_seeds] # this method uses softmax
nn_la_la = [np.load(f"true_rewards_rs_t5-base-test/{i}/pref_learning/nn_active_3_3_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
qbc = [np.load(f"true_rewards_rs_t5-base-active/{i}/pref_learning/qbc_lr_0.005_num_models_50_other_feedback_20_seed_{j}_temp_1.0_lc_1.0.npz") for i in sim for j in num_seeds]
bald = [np.load(f"true_rewards_rs_t5-base-active/{i}/pref_learning/bald_lr_0.005_num_models_50_lang_3_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]

# test1.png, looks good except active does worse than lang, uses argmax for info gain
name = "rs_main_figure"
# name = "rs_new_t5_base-both-norm_general_lang_vs_active_lr_5e-3_test1" <- original name
nn_la_la = [np.load(f"true_rewards_rs_t5-base-test/{i}/pref_learning/nn_active_3_3_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
lang = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/lang_noisy_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
comp = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/comp_noisy_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)

# CE
best = range(len(nn_la_la))
# main figure
nn_la_la_ce = [nn_la_la[i]["eval_cross_entropies"] for i in best]
lang_ce = [lang[i]["eval_cross_entropies"] for i in best]
comp_ce = [comp[i]["eval_cross_entropies"] for i in best]

# active figure
# nn_la_la_ce = [nn_la_la[i]["eval_cross_entropies"] for i in best]
# la_la_ce = [la_la[i]["eval_cross_entropies"] for i in best]
# qbc_ce = [qbc[i]["eval_cross_entropies"] for i in best]
# bald_ce = [bald[i]["eval_cross_entropies"] for i in best] # aka, Mutual Information

plot_curve_and_std(ax1, np.mean(nn_la_la_ce, axis=0),
                np.std(nn_la_la_ce, axis=0), "NN-LA-LA",
                color=colors[1])
# plot_curve_and_std(ax1, np.mean(la_la_ce, axis=0),
#                 np.std(la_la_ce, axis=0), "LA-LA",
#                 color=colors[2])
# plot_curve_and_std(ax1, np.mean(qbc_ce, axis=0),
#                 np.std(qbc_ce, axis=0), "QbC",
#                 color=colors[3])
# plot_curve_and_std(ax1, np.mean(bald_ce, axis=0),
#                 np.std(bald_ce, axis=0), "MI",
#                 color=colors[4])
plot_curve_and_std(ax1, np.mean(lang_ce, axis=0),
                np.std(lang_ce, axis=0), "Lang",
                color=colors[0])
plot_curve_and_std(ax1, np.mean(comp_ce, axis=0),
                np.std(comp_ce, axis=0), "Comparative",
                color=colors[5])
ax1.set_ylim(0.60, 0.70)
ax1.set_xlabel("Number of Queries")
ax1.set_ylabel("Cross-Entropy")
ax1.set_title("Robosuite")
ax1.legend(frameon=False)

# Rewards of ABI
# main figure
nn_la_la_rew = [nn_la_la[i]["optimal_learned_rewards"]/nn_la_la[i]["optimal_true_rewards"] for i in range(num_data)]
lang_rew = [lang[i]["optimal_learned_rewards"]/lang[i]["optimal_true_rewards"] for i in best]
comp_rew = [comp[i]["optimal_learned_rewards"]/comp[i]["optimal_true_rewards"] for i in best]

# active
# nn_la_la_rew = [nn_la_la[i]["optimal_learned_rewards"]/nn_la_la[i]["optimal_true_rewards"] for i in range(num_data)]
# la_la_rew = [la_la[i]["optimal_learned_rewards"]/la_la[i]["optimal_true_rewards"] for i in range(num_data)]
# qbc_rew = [qbc[i]["optimal_learned_rewards"]/qbc[i]["optimal_true_rewards"] for i in range(num_data)]
# bald_rew = [bald[i]["optimal_learned_rewards"]/bald[i]["optimal_true_rewards"] for i in range(num_data)]

plot_curve_and_std(ax2, np.mean(nn_la_la_rew, axis=0),
                np.std(nn_la_la_rew, axis=0), "NN-LA-LA",
                color=colors[1])
# plot_curve_and_std(ax2, np.mean(la_la_rew, axis=0),
#                 np.std(la_la_rew, axis=0), "LA-LA",
#                 color=colors[2])
# plot_curve_and_std(ax2, np.mean(qbc_rew, axis=0),
#                 np.std(qbc_rew, axis=0), "QbC",
#                 color=colors[3])
# plot_curve_and_std(ax2, np.mean(bald_rew, axis=0),
#                 np.std(bald_rew, axis=0), "MI",
#                 color=colors[4])
plot_curve_and_std(ax2, np.mean(lang_rew, axis=0),
                np.std(lang_rew, axis=0), "Lang",
                color=colors[0])
plot_curve_and_std(ax2, np.mean(comp_rew, axis=0),
                np.std(comp_rew, axis=0), "Comp",
                color=colors[5])
ax2.set_xlabel("Number of Queries")
ax2.set_ylabel("Reward Value")
ax2.set_title("Robosuite")
ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig(name)

'''
# argmin.png, uses argmin for info gain
# la_la = [np.load(f"true_rewards_rs_t5-base-argmin/{i}/pref_learning/1_3_3_dupe_traj_-1_seed_{j}.npz") for i in sim for j in num_seeds] # this method uses softmax
# nn_la_la = [np.load(f"true_rewards_rs_t5-base-argmin/{i}/pref_learning/nn_active_3_3_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
# lang = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/lang_noisy_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
# comp = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/comp_noisy_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]

# the ideal results
# la_la = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/1_3_3_dupe_traj_-1_seed_{j}.npz") for i in sim for j in num_seeds] # this method uses softmax
# nn_la_la = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/nn_active_3_3_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
# lang = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/lang_noisy_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
# comp = [np.load(f"true_rewards_rs_t5-base-both-norm/{i}/pref_learning/comp_noisy_lr_0.005_dupe_traj_-1_num_iter_30_other_feedback_20_seed_{j}_temp_1.0_lc_1.0_wd_0.2.npz") for i in sim for j in num_seeds]
'''