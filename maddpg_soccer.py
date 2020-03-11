import copy

from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

from madddpg_utils import DDPGConfig, MADDDPGConfig, train_agent, EnvironmentWrapper, ExploreStrategy
from maddpg_trainer import MADDPGManager

NUM_SUB_POLICIES = 1

def default_cfg():
    config = DDPGConfig()
    config.actor_lr = 1e-3
    config.critic_lr_lr = 1e-3
    config.batch_size = 128
    config.noise_end = 0.05
    config.noise_steps = 600 * 15
    config.fc_1_hidden_actor = 512
    config.fc_2_hidden_actor = 128
    config.fc_1_hidden_critic = 512
    config.fc_2_hidden_critic = 128
    config.repeat_learn = 4
    config.ou_eps_decay_episodes = 120
    config.ou_eps_start = 1.
    config.ou_eps_final = .1
    config.ou_mu = 0
    config.ou_theta = 0.15
    config.ou_sigma = 0.2
    config.ou_steps_per_episode_estimate = 601
    config.continues_actions = False
    config.explore_strategy = ExploreStrategy.EPSILON
    config.low_action = -1
    config.high_action = 1
    config.central_critic = True

    return config


def generate_grid_config():
    configs = []
    update_every = [1, 3]
    hidden_1 = [128, 256, 512]
    for h in hidden_1:
        for u in update_every:
            tmp = default_cfg()
            tmp.update_every = u
            tmp.fc_1_hidden_critic = h
            tmp.fc_1_hidden_actor = h
            configs.append(tmp)
    # tmp.n_steps = n
    return configs


def slow_epsilon(config):
    config.explore_strategy = ExploreStrategy.EPSILON
    config.ou_eps_start = 1.
    config.ou_eps_final = 0.95
    config.central_critic = False
    config.update_every = 40
    config.repeat_learn = 1
    return config


if __name__ == "__main__":
    # env = UnityEnvironment(file_name='Reacher_20/Reacher.x86_64')
    env = UnityEnvironment(file_name='environments/Soccer_Linux/Soccer.x86_64')
    env = EnvironmentWrapper(env, discrete_actions=True, skip_frames=0, action_min=-1, action_max=1.)

    for config in generate_grid_config():

        config.multi_agent_actions = env.action_sizes
        config.multi_agent_states = env.state_sizes
        multi_cfg = []
        for i in range(env.num_agents):
            agent_cfg = copy.deepcopy(config)
            if i in [0, 2]:
                agent_cfg = slow_epsilon(agent_cfg)
            maddpg_cfg = MADDDPGConfig()
            agent_cfg.state_size = agent_cfg.multi_agent_states[i]
            agent_cfg.action_size = agent_cfg.multi_agent_actions[i]

            maddpg_cfg.subpolicy_configs = [agent_cfg] * NUM_SUB_POLICIES


            multi_cfg.append(maddpg_cfg)

        agent = MADDPGManager(maddpg_agents_configs=multi_cfg, update_every=config.update_every, warm_up_episodes=50,
                              summary_writer=SummaryWriter("madddpg_soccer_logs/{}".format(config)))
        train_agent(env, agent, main_weight_folder="madddpg_soccer_weight/{}/".format(config),
                    n_episodes=250, evaluation_freq=50)
        # torch.save(agent.actor_critic.state_dict(), 'weights/{}.pth'.format(agent.ddpg_config))
        # agent.actor_critic.load_state_dict(torch.load('weights/{}.pth'.format(agent.ddpg_config)))
        # test_agent(env, agent, brain_name, num_agents)
