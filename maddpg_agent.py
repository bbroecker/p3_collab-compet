import numpy as np
from ddpg_agent import DDPGAgent


class MADDPGAgent:

    def __init__(self, maddpg_config, agent_idx):
        self.agent_idx = agent_idx
        self.subpolicies = []
        for ddpg_config in maddpg_config.subpolicy_configs:
            self.subpolicies.append(DDPGAgent(ddpg_config))

        self.current_policy = self._random_subpolicy()

    def _random_subpolicy(self):
        return self.subpolicies[np.random.randint(0, len(self.subpolicies))]

    def select_new_policy(self):
        self.current_policy = self._random_subpolicy()

    def get_current_policy(self):
        return self.current_policy
