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
