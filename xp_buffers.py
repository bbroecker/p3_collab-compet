from enum import Enum

import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sum_tree import SumTree

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BufferType(Enum):
    NORMAL = 0
    PRIORITY = 1
    MULTI_AGENT = 2

class PriorityBuffer:
    # Inspired by implementation from: https://github.com/rlcode/per/blob/master/prioritized_memory.py

    def __init__(self, action_size, agent_config):
        """Initialize a PriorityBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a (float): amount of uniformity in the sampling (0 == uniform, 1. == priority only)
            beta_start (float): start of beta value for prioritised buffer
            beta_max_steps (int): max number of steps to reach beta value of 1.
        """
        self.action_size = action_size
        self.tree = SumTree(capacity=agent_config.buffer_size)
        self.batch_size = agent_config.batch_size
        # self.seed = random.seed(buffer_config.seed)
        self.epsilon = agent_config.buffer_epsilon
        # how much randomness we require a = 0 (pure random) a = 1 (only priority)
        self.alpha = agent_config.alpha
        self.beta = agent_config.beta_start
        self.beta_start = agent_config.beta_start
        self.beta_end = agent_config.beta_end
        self.beta_increment_per_sampling = (self.beta_end - self.beta_start) / agent_config.beta_max_steps

    def add(self, sample, error):
        """Add a new experience to memory."""
        p = self._get_priority(error)
        state, action, reward, next_state, done = sample
        e = Experience(state, action, reward, next_state, done)
        self.tree.add(p, e)

    def _get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def sample(self):
        experiences = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []


        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if isinstance(data, Experience):
                priorities.append(p)
                experiences.append(data)
                idxs.append(idx)
            else:
                print("WHAT THE HECK !!!")

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        self.beta = np.min([self.beta_end, self.beta + self.beta_increment_per_sampling])
        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idx, error):
        # Not required in normal ReplayBuffer
        self.tree.update(idx, self._get_priority(error))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)


class StandardBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_config):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_config.buffer_size)
        self.batch_size = buffer_config.batch_size

        # self.seed = random.seed(buffer_config.seed)

    def add(self, sample, error):
        """Add a new experience to memory."""
        state, action, reward, next_state, done = sample
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().detach().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().detach().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().detach().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().detach().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().detach().to(
            device)

        # since buffer is not able update samples, indices are irrelevant
        idxs = [0] * self.batch_size
        # uniform weights
        weights = [1.] * self.batch_size
        return (states, actions, rewards, next_states, dones), idxs, weights

    def update(self, idx, error):
        # Not required in normal ReplayBuffer
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MultiAgentBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_config):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_config.buffer_size)
        self.batch_size = buffer_config.batch_size

        # self.seed = random.seed(buffer_config.seed)

    def add(self, sample, error):
        """Add a new experience to memory."""
        state, action, reward, next_state, done = sample
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().detach().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().detach().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().detach().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().detach().to(
            device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().detach().to(
            device)

        # since buffer is not able update samples, indices are irrelevant
        idxs = [0] * self.batch_size
        # uniform weights
        weights = [1.] * self.batch_size
        return (states, actions, rewards, next_states, dones), idxs, weights

    def update(self, idx, error):
        # Not required in normal ReplayBuffer
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class RollOutMemory:

    def __init__(self):
        self._experiences = []

    def clear(self):
        del self._experiences[:]

    def add(self, actions, rewards, log_prob, entropy, dones, state_values):
        self._experiences.append([actions, rewards, log_prob, entropy, dones, state_values])

    @property
    def experiences(self):
        # _action, _reward, _log_prob, _entropy, _done, _value
        return self._experiences

    def size(self):
        return len(self._experiences)


