import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):

    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorCriticModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim, seed, hidden_actor, hidden_critic):
        super(ActorCriticModel, self).__init__()

        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.input_fc = nn.Linear(state_dim, hidden_actor[0])

        self.actor_fc_1 = nn.Linear(hidden_actor[0], hidden_actor[1])
        self.actor_output = nn.Linear(hidden_actor[1], action_dim)

        self.critic_fc_1 = nn.Linear(hidden_critic[0], hidden_critic[1])
        self.critic_output = nn.Linear(hidden_critic[1], 1)

        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.reset_parameters()
        self.to(device)

    def forward(self, state):

        input_ = F.relu(self.input_fc(state))

        mean = self.actor_output(F.relu(self.actor_fc_1(input_)))
        action_dist = torch.distributions.Normal(mean, self.std)

        state_value = self.critic_output(F.relu(self.critic_fc_1(input_)))
        actions = action_dist.sample()
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return torch.clamp(actions, -1, 1), log_prob, entropy, state_value

    def reset_parameters(self):

        self.input_fc.weight.data.uniform_(*hidden_init(self.input_fc))
        self.actor_fc_1.weight.data.uniform_(*hidden_init(self.actor_fc_1))
        self.critic_fc_1.weight.data.uniform_(*hidden_init(self.critic_fc_1))
        self.actor_output.weight.data.uniform_(-3e-3, 3e-3)
        self.critic_output.weight.data.uniform_(-3e-3, 3e-3)


class DDPGActorModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim, seed, hidden_actor_fc_1=128, hidden_actor_fc_2=128, min_action=-1, max_action=1, output_activation=F.tanh):
        super(DDPGActorModel, self).__init__()
        self.min_action = min_action
        self.max_action = max_action
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.output_activation = output_activation
        self.actor_fc_1 = nn.Linear(state_dim, hidden_actor_fc_1)

        self.actor_fc_2 = nn.Linear(hidden_actor_fc_1, hidden_actor_fc_2)
        self.actor_output = nn.Linear(hidden_actor_fc_2, action_dim)

        self.reset_parameters()
        self.to(device)

    def forward(self, state):
        x = F.relu(self.actor_fc_1(state))
        x = F.relu(self.actor_fc_2(x))
        if self.output_activation is not None:
            x = self.output_activation(self.actor_output(x))
        else:
            x = self.actor_output(x)

        return torch.clamp(x, self.min_action, self.max_action)

    def reset_parameters(self):

        self.actor_fc_1.weight.data.uniform_(*hidden_init(self.actor_fc_1))
        self.actor_fc_2.weight.data.uniform_(*hidden_init(self.actor_fc_2))
        # self.actor_output.weight.data.uniform_(-3e-3, 3e-3)
        self.actor_output.weight.data.uniform_(-3e-3, 3e-3)


class DDPGCriticModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim, seed, hidden_critic_fc_1=128, hidden_critic_fc_2=128):
        super(DDPGCriticModel, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.critic_fc_1 = nn.Linear(state_dim + action_dim, hidden_critic_fc_1)

        self.critic_fc_2 = nn.Linear(hidden_critic_fc_1, hidden_critic_fc_2)
        self.critic_output = nn.Linear(hidden_critic_fc_2, 1)

        self.reset_parameters()
        self.to(device)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.critic_fc_1(x))
        x = F.relu(self.critic_fc_2(x))
        return self.critic_output(x)

    def reset_parameters(self):

        self.critic_fc_1.weight.data.uniform_(*hidden_init(self.critic_fc_1))
        self.critic_fc_2.weight.data.uniform_(*hidden_init(self.critic_fc_2))
        self.critic_output.weight.data.uniform_(-3e-3, 3e-3)
