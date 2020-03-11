import copy

from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from madddpg_utils import DDPGConfig, MADDDPGConfig, train_agent, EnvironmentWrapper, evaluate_current_weights, \
    ExploreStrategy
from maddpg_trainer import MADDPGManager

NUM_SUB_POLICIES = 1

def default_cfg():
    config = DDPGConfig()
    config.actor_lr = 1e-3
    config.critic_lr_lr = 1e-3
    config.fc_1_hidden_actor = 128
    config.fc_2_hidden_actor = 128
    config.fc_1_hidden_critic = 128
    config.fc_2_hidden_critic = 128
    config.buffer_size = int(1e6)
    config.batch_size = 128
    config.gamma = 0.95
    config.tau = 8e-3
    config.repeat_learn = 5
    config.update_every = 1
    config.ou_eps_decay_episodes = 1200
    config.ou_eps_start = 5.0
    config.ou_eps_final = .0
    config.ou_mu = 0
    config.ou_theta = 0.15
    config.ou_sigma = 0.2
    config.noise_start = 0.30
    config.ou_steps_per_episode_estimate = 15
    config.continues_actions = True
    config.explore_strategy = ExploreStrategy.NOU_NOISE
    config.central_critic = True
    return config


def generate_grid_config():
    configs = []
    update = [1]
    central = [True]
    repeat = [5]
    noise_start = [0.30]
    critic_lr = [1e-3]
    actor_lr = [1e-3]
    for u in update:
        for cen in central:
            for a in actor_lr:
                for c in critic_lr:
                    for n in noise_start:
                        for r in repeat:
                            tmp = default_cfg()
                            tmp.update_every = u
                            tmp.central_critic = cen
                            tmp.noise_start = n
                            tmp.actor_lr = a
                            tmp.critic_lr = c
                            tmp.repeat_learn = r
                            configs.append(tmp)
    # tmp.n_steps = n
    return configs



if __name__ == "__main__":
    # env = UnityEnvironment(file_name='Reacher_20/Reacher.x86_64')
    env = UnityEnvironment(file_name='environments/Tennis_Linux/Tennis.x86_64')
    env = EnvironmentWrapper(env)

    for config in generate_grid_config():
        config.multi_agent_actions = env.action_sizes
        config.multi_agent_states = env.state_sizes
        multi_cfg = []
        for i in range(env.num_agents):
            agent_cfg = copy.deepcopy(config)
            agent_cfg.state_size = config.multi_agent_states[i]
            agent_cfg.action_size = config.multi_agent_actions[i]
            maddpg_cfg = MADDDPGConfig()
            maddpg_cfg.subpolicy_configs = [agent_cfg] * NUM_SUB_POLICIES
            multi_cfg.append(maddpg_cfg)

        agent = MADDPGManager(maddpg_agents_configs=multi_cfg, update_every=config.update_every,
                              summary_writer=SummaryWriter("madddpg_tennis_logs/{}".format(config)), warm_up_episodes=0)
        # train_agent(env, agent, main_weight_folder="madddpg_tennis_weight/{}/".format(config),
        #             n_episodes=4000, evaluation_freq=50)
        agent.load_weights(main_folder="best_weights/")
        evaluate_current_weights(env, agent, env.num_agents, train_mode=False)
        # torch.save(agent.actor_critic.state_dict(), 'weights/{}.pth'.format(agent.ddpg_config))
        # agent.actor_critic.load_state_dict(torch.load('weights/{}.pth'.format(agent.ddpg_config)))
        # test_agent(env, agent, brain_name, num_agents)
