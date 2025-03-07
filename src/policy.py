import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SharedPolicy(nn.Module):
"""
A simple Multi-Layer Perceptron (MLP) policy for a multi-agent driving simulator.
Each agent receives an observation (e.g., position, velocity, and additional info)
and the network outputs logits for two discrete action categories:
- Acceleration (e.g., decelerate, maintain speed, accelerate)
- Steering (e.g., left, straight, right)

This shared policy is designed to work with batched observations from all agents.
"""

def __init__(self, observation_dim, hidden_dim=128, action_bins=3):
    """
    Initialize the network.
    :param observation_dim: Dimensionality of the input observation per agent.
    :param hidden_dim: Number of neurons in the hidden layers.
    :param action_bins: Number of discrete actions for each head (default=3).
    """
    super(SharedPolicy, self).__init__()
    self.fc1 = nn.Linear(observation_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    # Two output heads for discrete actions:
    self.accel_head = nn.Linear(hidden_dim, action_bins)  # for throttle/brake actions
    self.steer_head = nn.Linear(hidden_dim, action_bins)  # for steering actions

def forward(self, x):
    """
    Forward pass through the network.
    :param x: Input tensor of shape [batch_size, observation_dim].
    :return: Tuple (logits_accel, logits_steer) where each is of shape [batch_size, action_bins].
    """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits_accel = self.accel_head(x)
    logits_steer = self.steer_head(x)
    return logits_accel, logits_steer

def get_action(self, observation):
    """
    Sample actions from the policy given an observation.
    :param observation: A numpy array or torch.Tensor of shape [batch_size, observation_dim].
                        For single-agent operation, the shape can be [observation_dim],
                        but it will be unsqueezed to include a batch dimension.
    :return:
      - actions: Tensor of shape [batch_size, 2] with discrete actions for each agent.
      - log_probs: Tensor of shape [batch_size, 2] of the log probabilities corresponding to each action.
    """
    # Convert input to tensor if necessary and ensure batch dimension exists.
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)
    if observation.dim() == 1:
        observation = observation.unsqueeze(0)

    logits_accel, logits_steer = self.forward(observation)
    distrib_accel = Categorical(logits=logits_accel)
    distrib_steer = Categorical(logits=logits_steer)

    # Sample discrete actions from each distribution.
    action_accel = distrib_accel.sample()
    action_steer = distrib_steer.sample()

    # Calculate the log probabilities of the sampled actions.
    log_prob_accel = distrib_accel.log_prob(action_accel)
    log_prob_steer = distrib_steer.log_prob(action_steer)

    # Stack actions and log probabilities along the last dimension.
    actions = torch.stack([action_accel, action_steer], dim=-1)
    log_probs = torch.stack([log_prob_accel, log_prob_steer], dim=-1)
    return actions, log_probs

def evaluate_actions(self, observations, actions):
    """
    Evaluate provided actions given a batch of observations.
    Useful for computing log probabilities and entropy in policy gradient algorithms (e.g., PPO).
    :param observations: Tensor of shape [batch_size, observation_dim].
    :param actions: Tensor of shape [batch_size, 2] where each row is (acceleration, steering) action.
    :return:
      - log_probs: Tensor of shape [batch_size] containing the summed log probs for the two actions.
      - entropies: Tensor of shape [batch_size] containing the summed entropy of both distributions.
    """
    logits_accel, logits_steer = self.forward(observations)
    distrib_accel = Categorical(logits=logits_accel)
    distrib_steer = Categorical(logits=logits_steer)
    
    # Separate the actions.
    action_accel = actions[:, 0]
    action_steer = actions[:, 1]

    # Calculate log probabilities for each action.
    log_prob_accel = distrib_accel.log_prob(action_accel)
    log_prob_steer = distrib_steer.log_prob(action_steer)
    
    # Sum log probabilities (or return them separately based on your training objective).
    log_probs = log_prob_accel + log_prob_steer

    # Calculate distribution entropies (used for encouraging exploration in some RL algorithms).
    entropy_accel = distrib_accel.entropy()
    entropy_steer = distrib_steer.entropy()
    entropies = entropy_accel + entropy_steer

    return log_probs, entropies

