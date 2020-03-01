import os
import time
from collections import deque
import numpy as np
import torch

from maddpg_agent import MADDPGAgent
from xp_buffers import BufferType

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGManager:
    def __init__(self, maddpg_agents_configs, update_every, warm_up_episodes=0, score_window_size=100, summary_writer=None):
        self.maddpg_agents = [MADDPGAgent(cfg, idx) for idx, cfg in enumerate(maddpg_agents_configs)]
        self.max_batch_size = max(
            [sub.batch_size for mad_cfg in maddpg_agents_configs for sub in mad_cfg.subpolicy_configs])
        self.total_steps = 0
        self.current_episode = 0
        self.warm_up_episodes = warm_up_episodes
        self.update_every = update_every
        self.summary_writer = summary_writer
        self.score_window_size = score_window_size
        self.episode_reward = {i: [] for i in range(len(self.maddpg_agents))}
        self.episode_reward_window = deque(maxlen=score_window_size)
        self.actor_lost = {i: [] for i in range(len(self.maddpg_agents))}
        self.critic_lost = {i: [] for i in range(len(self.maddpg_agents))}

    def act_all(self, states, noise=True):
        all_actions = np.array([])
        for idx, state in enumerate(states):
            agent = self.maddpg_agents[idx % len(self.maddpg_agents)]
            action = agent.current_policy.act(state, noise).flatten()
            all_actions = np.concatenate((all_actions, action), axis=0)
        # all_actions = all_actions.reshape((2, 2))
        return all_actions

    def load_weights(self, main_folder):
        for agent in self.maddpg_agents:
            current_policy = agent.current_policy
            critic_file = "agent_{}_critic.pth".format(agent.agent_idx)
            actor_file = "agent_{}_actor.pth".format(agent.agent_idx)
            current_policy.ddpg_actor_local.load_state_dict(torch.load(os.path.join(main_folder, actor_file)))
            current_policy.ddpg_critic_local.load_state_dict(torch.load(os.path.join(main_folder, critic_file)))
            current_policy.ddpg_actor_target.load_state_dict(torch.load(os.path.join(main_folder, actor_file)))
            current_policy.ddpg_critic_target.load_state_dict(torch.load(os.path.join(main_folder, critic_file)))

    def save_weights(self, main_folder):
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        for agent in self.maddpg_agents:
            current_policy = agent.current_policy
            critic_file = "agent_{}_critic.pth".format(agent.agent_idx)
            actor_file = "agent_{}_actor.pth".format(agent.agent_idx)
            torch.save(current_policy.ddpg_critic_local.state_dict(), os.path.join(main_folder, critic_file))
            torch.save(current_policy.ddpg_actor_local.state_dict(), os.path.join(main_folder, actor_file))

    def episode_done(self, i_episode):
        self._log_print_episode_sum(i_episode)
        self._select_sub_policies()
        self.current_episode = i_episode

    def _select_sub_policies(self):
        for agent in self.maddpg_agents:
            agent.select_new_policy()

    def _log_print_episode_sum(self, i_episode):
        reward_sum = [np.sum(r) for r in self.episode_reward.values()]
        self.episode_reward_window.append(max(reward_sum))
        print("Episode {}: agent rewards {}".format(i_episode, reward_sum))

        if self.summary_writer:
            for idx, agent in enumerate(self.maddpg_agents):
                self.summary_writer.add_scalar('Agent_{}_Critic_Avg_Loss'.format(idx), np.mean(self.critic_lost[idx]),
                                               i_episode)
                self.summary_writer.add_scalar('Agent_{}_Actor_Avg_Loss'.format(idx), np.mean(self.actor_lost[idx]),
                                               i_episode)
                self.summary_writer.add_scalar('Agent_{}_Reward_per_episode'.format(idx), np.sum(self.episode_reward[idx]),
                                               i_episode)
                self.summary_writer.add_scalar('Agent_{}_epsilion'.format(idx), agent.current_policy.eps,
                                               i_episode)
                if self.maddpg_agents[idx].current_policy.noise_summery:
                    self.summary_writer.add_histogram('Agent_{}_Noise_Histogram'.format(idx),
                                                      np.array(self.maddpg_agents[idx].current_policy.noise_summery),
                                                      i_episode)
            self.summary_writer.add_scalar('Avg_window_Reward', np.mean(self.episode_reward_window), i_episode)
        self.clear_episode_buffer()
        return np.mean(self.episode_reward_window)

    def clear_episode_buffer(self):
        self.episode_reward = {i: [] for i in range(len(self.maddpg_agents))}
        self.actor_lost = {i: [] for i in range(len(self.maddpg_agents))}
        self.critic_lost = {i: [] for i in range(len(self.maddpg_agents))}

    def step_all_agents(self, states, actions, rewards, next_states, dones):
        # Save experience per agent in the replay buffer
        next_actions = []
        for agent, next_state in zip(self.maddpg_agents, next_states):
            self.episode_reward[agent.agent_idx].append(rewards[agent.agent_idx])
            if agent.current_policy.buffer_type == BufferType.PRIORITY:
                next_state_dev = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                next_actions.append(agent.current_policy.sample_action(next_state_dev).detach().to(device))

        if agent.current_policy.buffer_type == BufferType.PRIORITY:
            for agent, state, action, reward, next_state, done in zip(self.maddpg_agents, states, actions, rewards,
                                                                      next_states, dones):
                actions_dev = torch.from_numpy(actions).float().unsqueeze(0).to(device)
                state_dev = torch.from_numpy(state).float().unsqueeze(0).to(device)
                next_state_dev = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                # next_action = agent.ddpg_actor_target.forward(next_state_dev).detach().to(device)

                state_action_value = agent.current_policy.ddpg_critic_target.forward(next_state_dev,
                                                                                     next_actions).detach()
                q_new = reward + self.gamma * state_action_value * (1 - done)

                q_old = agent.current_policy.ddpg_critic_local.forward(state_dev, actions_dev).data.cpu().numpy()[0]
                error = (q_new.cpu().detach().numpy() - q_old) ** 2

                agent.current_policy.buffer.add((states, actions, rewards, next_states, dones), error)
        else:
            for agent in self.maddpg_agents:
                agent.current_policy.buffer.add((states, actions, rewards, next_states, dones), 1)

        self.total_steps += 1

        if self.total_steps % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            all_buff_size = [len(policy.buffer) for agent in self.maddpg_agents for policy in agent.subpolicies]
            if min(all_buff_size) > self.max_batch_size and self.current_episode > self.warm_up_episodes:
                self.train_all()

    def train_all(self):
        for idx, agent in enumerate(self.maddpg_agents):
            for _ in range(agent.current_policy.ddpg_config.repeat_learn):
                critic_loss_output, actor_loss_output = self.train(idx)
                self.critic_lost[idx].append(critic_loss_output)
                self.actor_lost[idx].append(actor_loss_output)

    def prepare_data(self, agent_idx, current_policy, states, actions, rewards, next_states, dones):
        my_dones = dones[:, agent_idx].unsqueeze(1).clone()
        my_rewards = rewards[:, agent_idx].unsqueeze(1).clone()

        if current_policy.central_critic:
            batch_size = current_policy.batch_size
            critic_state_size = current_policy.critic_state_size
            critic_action_size = current_policy.critic_action_size
            next_actions = torch.tensor([]).to(device)
            next_states_all = next_states.reshape(batch_size, critic_state_size).clone().detach()
            states_all = states.reshape(batch_size, critic_state_size).clone()
            actions_all = actions.reshape(batch_size, critic_action_size).clone()
            for agent in self.maddpg_agents:
                next_action = agent.current_policy.sample_action(next_states[:, agent.agent_idx]).detach()
                next_actions = torch.cat((next_actions, next_action), dim=1)
            current_actions = torch.tensor([]).to(device)
            for agent in self.maddpg_agents:
                current_action = agent.current_policy.ddpg_actor_local.forward(states[:, agent.agent_idx])
                current_actions = torch.cat((current_actions, current_action), dim=1)
        else:
            next_states_all = next_states[:, agent_idx].clone().detach()
            next_actions = current_policy.sample_action(next_states_all).clone().detach()
            states = states[:, agent_idx].clone()
            states_all = states
            actions_all = actions[:, agent_idx].clone()
            current_actions = current_policy.sample_action(states)

        return next_actions, current_actions, states_all, next_states_all, actions_all, my_dones, my_rewards

    def train(self, agent_idx):
        current_agent = self.maddpg_agents[agent_idx]
        current_policy = current_agent.current_policy
        experiences, idxs, is_weights = current_agent.current_policy.buffer.sample()
        states, actions, rewards, next_states, dones = experiences
        next_actions_policies, current_actions_policies, states_all_memory, next_states_all_memory, \
        actions_all_memory, my_dones_memory, my_rewards_memory = self.prepare_data(
            agent_idx, current_policy, states, actions, rewards, next_states, dones)

        q_next = current_policy.ddpg_critic_target.forward(next_states_all_memory, next_actions_policies).detach()

        # toDo check if max if good here (rewards[:, agent_idx])
        q_targets = rewards + (current_policy.gamma * q_next * (1 - dones))
        q_expected = current_policy.ddpg_critic_local.forward(states_all_memory, actions_all_memory)

        errors = torch.abs(q_targets - q_expected).data.cpu().numpy()
        for idx, error in zip(idxs, errors):
            current_policy.buffer.update(idx, error)

        current_policy.critic_optimizer.zero_grad()
        # if prioritised buffer is active the weights will effect the update
        # loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(q_expected, q_targets).squeeze())
        # loss = loss.mean()
        critic_loss = (q_targets - q_expected).pow(2).mul(0.5).mean()
        # critic_loss = (q_targets - q_expected).pow(2).mul(1).mean()
        critic_loss.backward()
        critic_loss_output = critic_loss.cpu().detach().numpy()
        # torch.nn.utils.clip_grad_norm_(current_policy.ddpg_critic_local.parameters(), 1)
        current_policy.critic_optimizer.step()

        # actions = current_policy.ddpg_actor_local.forward(states)
        # policy_loss = -(torch.FloatTensor(is_weights).to(device) * self.ddpg_critic_local.forward(states, actions))

        # policy_loss = policy_loss.mean()

        # if prioritised buffer is active the weights will effect the update
        policy_loss = -current_policy.ddpg_critic_local.forward(states_all_memory,
                                                                current_actions_policies).mean()
        current_policy.actor_optimizer.zero_grad()
        policy_loss.backward()
        current_policy.actor_optimizer.step()
        actor_loss_output = policy_loss.cpu().detach().numpy()

        # ------------------- update target network ------------------- #
        current_policy.soft_update()
        current_policy.update_eps()
        return critic_loss_output, actor_loss_output
