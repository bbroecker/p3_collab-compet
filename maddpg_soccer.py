import copy

from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from madddpg_utils import DDPGConfig, MADDDPGConfig, train_agent, EnvironmentWrapper
from maddpg_trainer import MADDPGManager

NUM_SUB_POLICIES = 1

def default_cfg():
    config = DDPGConfig()
    config.actor_lr = 1e-3
    config.critic_lr_lr = 1e-3
    config.noise_end = 0.05
    config.noise_steps = 600 * 15
    config.fc_1_hidden_actor = 800
    config.fc_2_hidden_actor = 600
    config.fc_1_hidden_critic = 800
    config.fc_2_hidden_critic = 600
    config.repeat_learn = 3
    config.ou_eps_decay_episodes = 80
    config.ou_eps_start = 0.95
    config.ou_eps_final = .05
    config.ou_mu = 0
    config.ou_theta = 0.15
    config.ou_sigma = 0.2
    config.ou_steps_per_episode_estimate = 601
    config.continues_actions = False

    return config


def generate_grid_config():
    configs = []
    batch_sizes = [128]
    central = [True]
    update_every = [5]
    noise_start = [0.4]
    critic_lr = [1e-3]
    actor_lr = [1e-3]
    for b in batch_sizes:
        for cen in central:
            for a in actor_lr:
                for c in critic_lr:
                    for n in noise_start:
                        for u in update_every:
                            tmp = default_cfg()
                            tmp.batch_size = b
                            tmp.central_critic = cen
                            tmp.noise_start = n
                            tmp.actor_lr = a
                            tmp.critic_lr = c
                            tmp.update_every = u
                            configs.append(tmp)
    # tmp.n_steps = n
    return configs



if __name__ == "__main__":
    # env = UnityEnvironment(file_name='Reacher_20/Reacher.x86_64')
    env = UnityEnvironment(file_name='environments/Soccer_Linux/Soccer.x86_64')
    env = EnvironmentWrapper(env, continues_actions=False)

    for config in generate_grid_config():

        config.multi_agent_actions = env.action_sizes
        config.multi_agent_states = env.state_sizes
        multi_cfg = []
        for i in range(env.num_agents):
            agent_cfg = copy.deepcopy(config)
            maddpg_cfg = MADDDPGConfig()
            agent_cfg.state_size = agent_cfg.multi_agent_states[i]
            agent_cfg.action_size = agent_cfg.multi_agent_actions[i]
            maddpg_cfg.subpolicy_configs = [agent_cfg] * NUM_SUB_POLICIES
            multi_cfg.append(maddpg_cfg)

        agent = MADDPGManager(maddpg_agents_configs=multi_cfg, update_every=config.update_every, warm_up_episodes=1,
                              summary_writer=SummaryWriter("madddpg_soccer_logs/{}".format(config)))
        train_agent(env, agent, main_weight_folder="madddpg_soccer_weight/{}/".format(config),
                    n_episodes=5000, evaluation_freq=50)
        # torch.save(agent.actor_critic.state_dict(), 'weights/{}.pth'.format(agent.ddpg_config))
        # agent.actor_critic.load_state_dict(torch.load('weights/{}.pth'.format(agent.ddpg_config)))
        # test_agent(env, agent, brain_name, num_agents)
