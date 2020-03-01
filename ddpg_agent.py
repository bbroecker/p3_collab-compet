import copy
from collections import deque
from enum import Enum

import numpy as np
import random

from models import DDPGActorModel, DDPGCriticModel

import torch
import torch.optim as optim

from xp_buffers import BufferType, StandardBuffer, PriorityBuffer, MultiAgentBuffer
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class SoftCategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return torch.softmax(self.logits, axis=-1)
#     def logp(self, x):
#         return -tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
#     def kl(self, other):
#         a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
#         a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
#         ea0 = a0.exp()
#         ea1 = a1.exp()
#         z0 = ea0.sum(axis=1, keepdims=True)
#         z1 = ea1.sum(axis=1, keepdims=True)
#         p0 = ea0 / z0
#         return torch.sum(p0 * (a0 - z0.log() - a1 + z1.log()), axis=1)
#
#     def entropy(self):
#         a0 = self.logits - self.logits.max(axis=1, keepdims=True)
#         ea0 = torch.exp(a0)
#         z0 = torch.sum(ea0, axis=1, keepdims=True)
#         p0 = ea0 / z0
#         return torch.sum(p0 * (z0.log() - a0), axis=1)
#
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.softmax(self.logits - tf.log(-tf.log(u)), axis=-1)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, ddpg_config, num_agents=1):
        """Initialize an Agent object."""

        self.num_agents = num_agents
        self.state_size = ddpg_config.state_size
        self.action_size = ddpg_config.action_size
        self.seed = random.seed(ddpg_config.seed)
        self.buffer_type = ddpg_config.buffer_type
        self.ddpg_config = ddpg_config
        seed = ddpg_config.seed
        self.central_critic = self.ddpg_config.central_critic
        # self.skip_frames = ddpg_config.skip_frames
        if self.ddpg_config.central_critic:
            self.critic_action_size = sum(
                [agent_action_size for agent_action_size in self.ddpg_config.multi_agent_actions])
            self.critic_state_size = sum([state_size for state_size in self.ddpg_config.multi_agent_states])
        else:
            self.critic_action_size = self.action_size
            self.critic_state_size = self.state_size

        fc_1_hidden_actor = ddpg_config.fc_1_hidden_actor
        fc_2_hidden_actor = ddpg_config.fc_2_hidden_actor
        fc_1_hidden_critic = ddpg_config.fc_1_hidden_critic
        fc_2_hidden_critic = ddpg_config.fc_2_hidden_critic

        output_activation = F.tanh if self.ddpg_config.continues_actions else F.softmax

        self.ddpg_actor_local = DDPGActorModel(state_dim=self.state_size, action_dim=self.action_size, seed=seed,
                                               hidden_actor_fc_1=fc_1_hidden_actor,
                                               hidden_actor_fc_2=fc_2_hidden_actor,
                                               output_activation=output_activation).to(device)
        self.ddpg_actor_target = DDPGActorModel(state_dim=self.state_size, action_dim=self.action_size, seed=seed,
                                                hidden_actor_fc_1=fc_1_hidden_actor,
                                                hidden_actor_fc_2=fc_2_hidden_actor,
                                                output_activation=output_activation).to(device)

        self.ddpg_critic_local = DDPGCriticModel(state_dim=self.critic_state_size, action_dim=self.critic_action_size,
                                                 seed=seed, hidden_critic_fc_1=fc_1_hidden_critic,
                                                 hidden_critic_fc_2=fc_2_hidden_critic).to(device)
        self.ddpg_critic_target = DDPGCriticModel(state_dim=self.critic_state_size, action_dim=self.critic_action_size,
                                                  seed=seed, hidden_critic_fc_1=fc_1_hidden_critic,
                                                  hidden_critic_fc_2=fc_2_hidden_critic).to(device)

        self.ou_noise = OrnsteinUhlenbeckProcess(size=(self.action_size,), seed=ddpg_config.seed, mu=ddpg_config.ou_mu,
                                                 theta=ddpg_config.ou_theta, sigma=ddpg_config.ou_sigma)

        self.critic_optimizer = optim.Adam(self.ddpg_critic_local.parameters(), lr=ddpg_config.critic_lr)
        self.actor_optimizer = optim.Adam(self.ddpg_actor_local.parameters(), lr=ddpg_config.actor_lr)

        self.eps = self.ddpg_config.ou_eps_start
        self.eps_steps_end = self.ddpg_config.ou_eps_decay_episodes
        self.eps_decay = (self.ddpg_config.ou_eps_start - self.ddpg_config.ou_eps_final) / (
                self.ddpg_config.ou_eps_decay_episodes * self.ddpg_config.repeat_learn * (self.ddpg_config.ou_steps_per_episode_estimate / self.ddpg_config.update_every))
        self.eps_final = self.ddpg_config.ou_eps_final

        self.learning_steps = 0

        # self.criterion = nn.MSELoss()
        # Replay memory
        if ddpg_config.buffer_type == BufferType.NORMAL:
            self.buffer = StandardBuffer(self.action_size, ddpg_config)
        elif ddpg_config.buffer_type == BufferType.PRIORITY:
            self.buffer = PriorityBuffer(self.action_size, ddpg_config)
        elif ddpg_config.buffer_type == BufferType.MULTI_AGENT:
            self.buffer = MultiAgentBuffer(self.action_size, ddpg_config)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.gamma = ddpg_config.gamma
        self.tau = ddpg_config.tau
        self.update_every = ddpg_config.update_every
        self.batch_size = ddpg_config.batch_size
        self.noise_summery = deque(maxlen=500)

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience per agent in the replay buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            if self.buffer_type == BufferType.PRIORITY:
                action_dev = torch.from_numpy(action).float().unsqueeze(0).to(device)
                state_dev = torch.from_numpy(state).float().unsqueeze(0).to(device)
                next_state_dev = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                next_action = self.ddpg_actor_target.forward(next_state_dev).detach().to(device)

                state_action_value = self.ddpg_critic_target.forward(next_state_dev, next_action).detach()
                q_new = reward + self.gamma * state_action_value * (1 - done)

                q_old = self.ddpg_critic_local.forward(state_dev, action_dev).data.cpu().numpy()[0]
                error = abs(q_new.cpu().detach().numpy() - q_old)
            else:
                error = 0.

            self.buffer.add((state, action, reward, next_state, done), error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.buffer) > self.batch_size:
                return self.learn()
        return None, None

    def buffer_size(self):
        return len(self.buffer)

    def reset(self):
        self.learning_steps = 0

    # def sample_action(self, state):
    #     act_pd = act_pdtype_n[p_index].pdfromflat(state)
    #     act_sample = act_pd.sample()
    #     p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
    #
    #     act_input_n = act_ph_n + []
    #     return act_pd.sample()

    def sample_action(self, state):
        actions = self.ddpg_actor_local.forward(state)
        if self.ddpg_config.continues_actions:
            return actions
        else:
            # logits = actions
            # u = torch.rand(logits.shape).to(device)
            # return torch.softmax(logits - torch.log(-torch.log(u)), dim=-1)
            return actions

    def _continues_action(self, state, noise):
        state = torch.from_numpy(state).float().to(device)
        action = self.sample_action(state).detach().cpu().detach().numpy()

        if not noise:
            return action
        elif self.learning_steps < self.ddpg_config.warmup_steps:
            action = np.random.uniform(self.ddpg_config.low_action, self.ddpg_config.high_action,
                                       (self.num_agents, self.action_size))
        else:
            noise_sample = self.ou_noise.sample() * self.eps
            self.noise_summery.append(noise_sample)
            action += noise_sample

            # action = action.detach().cpu().numpy()
        self.learning_steps += 1
        return np.clip(action, self.ddpg_config.low_action, self.ddpg_config.high_action)

    def _discrete_actions(self, state, noise):
        state = torch.from_numpy(state).float().to(device)
        # return self.sample_action(state).cpu().detach().numpy()
        if noise and np.random.rand() < self.eps:
            action_idx = np.random.randint(0, self.action_size)
            actions = np.zeros((1, self.action_size))
            actions[0][action_idx] = 1.
            return actions

        return self.sample_action(state).cpu().detach().numpy()


    def act(self, state, noise=False):
        return self._continues_action(state, noise) if self.ddpg_config.continues_actions else self._discrete_actions(
            state, noise)

    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        experiences, idxs, is_weights = self.buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.sample_action(next_states).detach()
        q_next = self.ddpg_critic_target.forward(next_states, next_actions).detach()

        q_targets = rewards + (self.gamma * q_next * (1 - dones))
        q_expected = self.ddpg_critic_local.forward(states, actions)

        errors = torch.abs(q_targets - q_expected).data.cpu().numpy()
        for idx, error in zip(idxs, errors):
            self.buffer.update(idx, error)

        self.critic_optimizer.zero_grad()
        # if prioritised buffer is active the weights will effect the update
        # loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(q_expected, q_targets).squeeze())
        # loss = loss.mean()
        critic_loss = (q_targets - q_expected).pow(2).mul(0.5).mean()
        critic_loss.backward()
        critic_loss_output = critic_loss.cpu().detach().numpy()
        self.critic_optimizer.step()

        actions = self.sample_action(states)
        # policy_loss = -(torch.FloatTensor(is_weights).to(device) * self.ddpg_critic_local.forward(states, actions))
        policy_loss = -self.ddpg_critic_local.forward(states, actions).mean()
        # policy_loss = policy_loss.mean()
        self.actor_optimizer.zero_grad()
        # if prioritised buffer is active the weights will effect the update

        policy_loss.backward()
        self.actor_optimizer.step()
        actor_loss_output = policy_loss.cpu().detach().numpy()

        # ------------------- update target network ------------------- #
        self.soft_update()
        self.update_eps()

        return critic_loss_output, actor_loss_output

    def update_eps(self):
        self.eps -= self.eps_decay
        self.eps = max(self.eps, self.eps_final)
        self.ou_noise.reset()

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.ddpg_critic_target.parameters(), self.ddpg_critic_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.ddpg_actor_target.parameters(), self.ddpg_actor_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


OU_SIGMA = 0.2
OU_THETA = 0.15


class OrnsteinUhlenbeckProcess:
    """
    Implementation inspiration by: https://github.com/ShangtongZhang/DeepRL
    """

    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu (float)    : long-running mean
            theta (float) : speed of mean reversion
            sigma (float) : volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class LinearSchedule:
    """
    Implementation inspiration by: https://github.com/ShangtongZhang/DeepRL
    Class to automatically increase/decrease the noise std-variation
    """

    def __init__(self, ddpp_config):
        start = ddpp_config.noise_start
        end = ddpp_config.noise_end
        steps = ddpp_config.noise_steps
        if end is None:
            end = ddpp_config.start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
