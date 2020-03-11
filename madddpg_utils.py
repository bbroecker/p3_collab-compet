from collections import deque, namedtuple
from enum import Enum

import torch
import numpy as np
from xp_buffers import BufferType

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EnvInfo = namedtuple("EnvInfo", field_names=["vector_observations", "rewards", "local_done"])

class ExploreStrategy(Enum):
    EPSILON = 0
    NOU_NOISE = 1
    DUMP = 2

class EnvironmentWrapper:
    def __init__(self, unity_env, discrete_actions=False, action_min=-1., action_max=1.):
        self.unity_env = unity_env
        self.discrete_actions = discrete_actions
        self.brain_names = unity_env.brain_names
        self.brains = [unity_env.brains[brain_name] for brain_name in self.brain_names]
        self.state_sizes = []
        self.action_sizes = []
        self.discrete_action_size = []
        self.discrete_action_size_brain_name = {}
        self.action_space_brain_name = {}
        self.brain_num_agents = {}
        self.num_agents = 0
        self.action_min = action_min
        self.action_max = action_max
        self.score = [0] * 2
        self._get_state_action_space()

    def _get_state_action_space(self):
        self.num_agents = 0
        for brain, brain_name in zip(self.brains, self.brain_names):
            env_info = self.unity_env.reset(train_mode=True)[brain_name]

            print('brain_name:', brain_name)
            # number of agents
            num_agents = len(env_info.agents)
            print('Number of agents:', num_agents)
            self.num_agents += num_agents

            self.brain_num_agents[brain_name] = num_agents

            # size of each action
            action_size = brain.vector_action_space_size
            print('Size of each action:', action_size)

            # examine the state space
            states = env_info.vector_observations
            state_size = states.shape[1]
            self.state_sizes.extend([state_size] * num_agents)
            if self.discrete_actions:
                self.action_sizes.extend([1] * num_agents)
                self.action_space_brain_name[brain_name] = 1
                if brain_name == "StrikerBrain":
                    action_size = 4
                self.discrete_action_size.extend([action_size] * num_agents)
                self.discrete_action_size_brain_name[brain_name] = action_size
            else:
                self.action_sizes.extend([action_size] * num_agents)
                self.action_space_brain_name[brain_name] = action_size
            print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
            print('The state for the first agent looks like:', states[0])

    def reset(self, train_mode=True):
        states_all = []
        rewards_all = []
        dones_all = []
        agents = 0
        for brain, brain_name in zip(self.brains, self.brain_names):
            env_info = self.unity_env.reset(train_mode=train_mode)[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            states_all.extend([state for state in states])
            rewards_all.extend([reward for reward in rewards])
            dones_all.extend([done for done in dones])
        env_info = EnvInfo(np.array(states_all), np.array(rewards_all), np.array(dones_all))

        return env_info

    def close(self):
        self.unity_env.close()


    @staticmethod
    def _continues_to_discrete(value, min_value, max_value, discrete_actions, brain_name):
        value_range = max_value - min_value
        value = np.clip(value, min_value, max_value - 0.001)
        step = value_range / discrete_actions
        action = int((value - min_value) / float(step))
        if brain_name == "StrikerBrain":
            if action > 1:
                action += 2
        return action

    def _discrete_actions(self, brain_name, brain_actions):
        actions = []
        start_idx = 0
        action_sizes = self.action_space_brain_name[brain_name]
        for idx in range(self.brain_num_agents[brain_name]):
            end_idx = start_idx+action_sizes
            # probs = brain_actions[start_idx: end_idx]
            # a = [i for i in range(action_sizes)]
            # action = np.random.choice(a, p=probs)
            action = self._continues_to_discrete(brain_actions[idx], self.action_min, self.action_max, self.discrete_action_size_brain_name[brain_name], brain_name)
            actions.append(action)
            start_idx = end_idx
        return np.array(actions)

    def step(self, actions):
        states_all = []
        rewards_all = []
        dones_all = []
        action_start = 0
        action_dict = {}
        for brain, brain_name in zip(self.brains, self.brain_names):
            action_size = self.action_space_brain_name[brain_name]
            action_end = action_start + self.brain_num_agents[brain_name] * action_size
            brain_actions = actions[action_start:action_end]
            if self.discrete_actions:
                brain_actions = self._discrete_actions(brain_name, brain_actions)
                # if brain_name == "GoalieBrain":
                #     print(brain_actions[0])
            action_dict[brain_name] = brain_actions
            action_start = action_end

        env_info = self.unity_env.step(action_dict)
        for brain_name in self.brain_names:
            info = env_info[brain_name]
            states = info.vector_observations
            rewards = info.rewards
            dones = info.local_done
            states_all.extend([state for state in states])
            rewards_all.extend([reward for reward in rewards])
            dones_all.extend([done for done in dones])

        env_info = EnvInfo(np.array(states_all), np.array(rewards_all), np.array(dones_all))
        return env_info



def evaluate_current_weights(env, agent_manager, num_agents, n_episodes=5, train_mode=True):
    total_score = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode) # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        dones = [False] * num_agents
        while not np.any(dones):
            actions = agent_manager.act_all(states, noise=False)
            env_info = env.step(actions)
            states = env_info.vector_observations
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            scores += rewards

        print("Evaluation episode {}, max agent scrore score {}".format(i_episode, np.max(scores)))
        total_score += np.max(scores) / n_episodes

    return total_score


def train_agent(env, agent_manager, main_weight_folder, n_episodes=2000,
                evaluation_freq=20):
    num_agents = env.num_agents
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=int(100 / num_agents))  # last 100 scores

    # save weight when the score is higher that current max
    max_score = 0.

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True) # reset the environment
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        dones = [False] * num_agents
        while not np.any(dones):
            actions = agent_manager.act_all(states, noise=True)
            env_info = env.step(actions)
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            agent_manager.step_all_agents(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards

        scores_window.append(np.mean(scores))  # save most recent score
        agent_manager.episode_done(i_episode)

        if i_episode % evaluation_freq == 0:
            print("Evaluating current weights.")
            current_score = evaluate_current_weights(env, agent_manager, num_agents, 5)
            print('\nScore with current weights and no noise. Episode {} evaluation score {}'.format(i_episode,
                                                                                                     current_score))
            if current_score > max_score:
                max_score = current_score
                agent_manager.save_weights(main_weight_folder)
                max_score_episode = i_episode
                print('\nNew max score. Weights saved {:d} episodes!\tAverage Score: {:.2f}'.format(max_score_episode,
                                                                                                    max_score))
            if agent_manager.summary_writer is not None:
                agent_manager.summary_writer.add_scalar('Eval_score', current_score, i_episode)

    return scores


class MADDDPGConfig:
    def __init__(self):
        self.subpolicy_configs = []


class DDPGConfig:
    def __init__(self):
        self.critic_lr = 1e-4
        self.actor_lr = 1e-3
        self.buffer_type = BufferType.MULTI_AGENT
        self.buffer_size = int(1e6)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 8e-3
        self.seed = 999
        self.update_every = 1
        self.warmup_steps = 200
        self.low_action = -1.
        self.high_action = 1.
        self.central_critic = True
        self.multi_agent_actions = None
        self.fc_1_hidden_actor = 128
        self.fc_2_hidden_actor = 128
        self.fc_1_hidden_critic = 128
        self.fc_2_hidden_critic = 128
        self.ou_eps_decay_episodes = 300
        self.repeat_learn = 5
        self.ou_eps_start = 5.0
        self.ou_eps_final = 0.
        self.ou_mu = 0
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_steps_per_episode_estimate = None
        self.continues_actions = True
        # self.std = nn.Parameter(torch.zeros(action_dim))

    def __str__(self):
        central = "" if not self.central_critic else "_central_critic"
        return "MADDPG_crh1_{}_crh2_{}_actor_lr_{}_critic_lr_{}_batch_size_{}_repeat_{}_update_every_{}_noise_start_{}_noise_end _{}{}".format(
            self.fc_1_hidden_critic,
            self.fc_2_hidden_critic,
            self.actor_lr, self.critic_lr, self.batch_size, self.repeat_learn, self.update_every, self.ou_eps_start, self.ou_eps_final, central)

# def load_weights(agents, main_dir)
#
#
# def load_agent_weights(agent, critic_dir, actor_dir):
#     agent.ddpg_actor_local.load_state_dict(torch.load(actor_dir))
#     agent.ddpg_critic_local.load_state_dict(torch.load(critic_dir))


