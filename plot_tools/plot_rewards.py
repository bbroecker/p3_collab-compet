import json
import numpy as np
import matplotlib.pyplot as plt

json_folder = "../tennis_json/"

figure_folder = "/home/broecker/src/udacity_rl/report_p3/figures/"

data_ddpg = {
    "file_name": "run-MADDPG_crh1_128_crh2_128_actor_lr_0.001_critic_lr_0.001_batch_size_128_repeat_3_update_every_1_noise_start_0.3_noise_end _0.05-tag-Avg_window_Reward.json",
    "label": "DDPG (decentralized critic)"}

data_madddpg = {
    "file_name": "run-MADDPG_crh1_128_crh2_128_actor_lr_0.001_critic_lr_0.001_batch_size_128_repeat_4_update_every_1_noise_start_5.0_noise_end _0.0_central_critic-tag-Avg_window_Reward.json",
    "label": "MADDPG (central critic)"}

def plot_reward_per_step(data_set, folders, colors, save_name, x_limit=1000, y_limit=1., aspect=500):
    labels = []
    for data_frame, color, folder in zip(data_set, colors, folders):
        labels.append(data_frame["label"])
        f = open(folder + data_frame["file_name"], 'r')

        data = np.array(json.load(f))
        xs = data[:, 1]
        ys = data[:, 2]
        firsts = [x for x, y in zip(xs, ys) if y > 13.]
        first = 'n/a' if not firsts else firsts[0]
        print("{} & {} &{:.2f} \\\\".format(data_frame["label"], first, max(ys)))
        print("\\hline")
        plt.plot(xs, ys, color=color, label=data_frame["label"], linewidth=1)
        f.close()
    plt.grid(color='#7f7f7f', linestyle='-', linewidth=1)
    plt.hlines(0.5, 0, x_limit, linestyles='dashed', linewidth=2.5)
    plt.xlim([0., x_limit])
    plt.ylim([0., y_limit])
    plt.xlabel("Episodes")
    plt.ylabel("Reward Avg (100 Episode )")
    plt.legend()
    plt.axes().set_aspect(aspect=aspect)
    plt.savefig(figure_folder + save_name, format="pdf", pad_inches=0, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_reward_per_step([data_ddpg, data_madddpg], [json_folder] * 2, ['r', 'b'],
                         'maddpg.pdf')


    # plot_reward_per_step([data_double_dqn, data_double_dqn_skip], ['r', 'b'], '../figures/priory.pdf')
