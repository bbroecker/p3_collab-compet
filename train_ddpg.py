from collections import deque

import torch
from unityagents import UnityEnvironment
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from ddpg_agent import DDPGAgent
from xp_buffers import BufferType

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_current_weights(env, agent, brain_name, num_agents, n_episodes=5, train_mode=True):
    total_score = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        dones = [False] * num_agents
        while not np.any(dones):
            actions = agent.act(states, noise=False)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            scores += rewards
        print("Evaluation episode {}, score {}".format(i_episode, np.mean(scores)))
        total_score += np.mean(scores) / n_episodes

    return total_score


def train_agent(env, agent, brain_name, num_agents, actor_weight_dir, critic_weight_dir, n_episodes=200,
                evaluation_freq=10, summary_writer=None):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    # save weight when the score is higher that current max
    max_score = 0.

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        critic_losses = []
        actor_losses = []
        dones = [False] * num_agents
        while not np.any(dones):
            actions = agent.act(states, noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            critic_loss, actor_loss = agent.step(states, actions, rewards, next_states, dones)
            if critic_loss is not None and actor_loss is not None:
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
            states = next_states
            scores += rewards


        scores_window.append(np.mean(scores))  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f} '
              'Actor Losses: {:.4f} Critic Losses {:.4f}'.format(i_episode, np.mean(scores),
                                                                 np.mean(actor_losses), np.mean(critic_losses)))
        if i_episode % int(100 / num_agents) == 0:
            print('\r100 Episode {}\tAverage Score: {:.2f} '
                  'Actor Losses: {:.4f} Critic Losses {:.4f}'.format(i_episode, np.mean(scores_window),
                                                                     np.mean(actor_losses),
                                                                     np.mean(critic_losses)))

        # evaluate current weights
        if i_episode % evaluation_freq == 0:
            print("Evaluating current weights.")
            current_score = evaluate_current_weights(env, agent, brain_name, num_agents, 5)
            print('\nScore with current weights and no noise. Episode {} evaluation score {}'.format(i_episode,
                                                                                                     current_score))
            if current_score > max_score:
                max_score = current_score
                torch.save(agent.ddpg_critic_local.state_dict(), critic_weight_dir)
                torch.save(agent.ddpg_actor_local.state_dict(), actor_weight_dir)
                max_score_episode = i_episode
                print('\nNew max score. Weights saved {:d} episodes!\tAverage Score: {:.2f}'.format(max_score_episode,
                                                                                                    max_score))
            if summary_writer is not None:
                summary_writer.add_scalar('Eval_score', current_score, i_episode)

        if summary_writer is not None:
            summary_writer.add_scalar('Critic_Avg_Loss', np.mean(critic_losses), i_episode)
            summary_writer.add_scalar('Actor_Avg_Loss', np.mean(actor_losses), i_episode)
            summary_writer.add_scalar('Avg_Reward', np.mean(scores_window), i_episode)

    return scores


class DDPGConfig:
    def __init__(self):
        self.critic_lr = 1e-4
        self.actor_lr = 1e-3
        self.buffer_type = BufferType.NORMAL
        self.buffer_size = int(1e5)
        self.batch_size = 128
        self.gamma = 0.95
        self.tau = 1e-3
        self.seed = 999
        self.update_every = 1
        self.warmup_steps = 2000
        self.low_action = -1
        self.high_action = 1
        self.noise_start = 0.4
        self.noise_end = 0.1
        self.noise_steps = 2000 * 10
        self.multi_agent = False
        self.fc_1_hidden_actor = 128
        self.fc_2_hidden_actor = 128
        self.fc_1_hidden_critic = 128
        self.fc_2_hidden_critic = 128
        # self.std = nn.Parameter(torch.zeros(action_dim))

    def __str__(self):
        return "DDPG_actor_lr_{}_critic_lr_{}_batch_size_{}_batch_size_{}_noise_start_{}_noise_end _{}".format(
            self.actor_lr, self.critic_lr, self.batch_size, self.update_every, self.noise_start, self.noise_end)


def generate_grid_config():
    configs = []
    batch_sizes = [128]
    update_every = [1]
    noise_start = [0.3, 0.4, 0.5]
    critic_lr = [1e-3, 1e-4, 1e-5]
    actor_lr = [1e-3, 1e-4, 1e-5]
    for b in batch_sizes:
        for u in update_every:
            for n in noise_start:
                for a in actor_lr:
                    for c in critic_lr:
                        tmp = DDPGConfig()
                        tmp.batch_size = b
                        tmp.update_every = u
                        tmp.noise_start = n
                        tmp.actor_lr = a
                        tmp.critic_lr = c
                        configs.append(tmp)
    # tmp.n_steps = n
    return configs


def load_weights(agent, critic_dir, actor_dir):
    agent.ddpg_actor_local.load_state_dict(torch.load(actor_dir))
    agent.ddpg_critic_local.load_state_dict(torch.load(critic_dir))


if __name__ == "__main__":
    # env = UnityEnvironment(file_name='Reacher_20/Reacher.x86_64')
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    for config in generate_grid_config():
        config.state_size = state_size
        config.action_size = action_size
        agent = DDPGAgent(config, num_agents=num_agents)
        critic_weight_dir = 'ddpg_weights/critic_{}.pth'.format(config)
        actor_weight_dir = 'ddpg_weights/actor_{}.pth'.format(config)
        train_agent(env, agent, brain_name, num_agents, actor_weight_dir=actor_weight_dir,
                    critic_weight_dir=critic_weight_dir, summary_writer=SummaryWriter("ddpg_logs/{}".format(config)),
                    n_episodes=3000, evaluation_freq=50)
        # torch.save(agent.actor_critic.state_dict(), 'weights/{}.pth'.format(agent.ddpg_config))
        # agent.actor_critic.load_state_dict(torch.load('weights/{}.pth'.format(agent.ddpg_config)))
        # test_agent(env, agent, brain_name, num_agents)
