import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

class SharedPPOPolicy(nn.Module):
    """Shared Actor-Critic Network for Multi-Agent PPO"""
    
    def __init__(self, 
                 input_dim: int = 29,
                 hidden_dim: int = 256,
                 throttle_bins: int = 3,
                 steer_bins: int = 5):
        super().__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor heads
        self.throttle_head = nn.Linear(hidden_dim, throttle_bins)
        self.steer_head = nn.Linear(hidden_dim, steer_bins)
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
        
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return state value"""
        features = self.shared_net(x)
        return self.critic(features)
    
    def get_action_and_value(self, 
                           x: torch.Tensor,
                           throttle_action: torch.Tensor = None,
                           steer_action: torch.Tensor = None):
        """Sample actions and compute logprobs/entropy/values"""
        features = self.shared_net(x)
        
        # Action distributions
        throttle_logits = self.throttle_head(features)
        steer_logits = self.steer_head(features)
        throttle_dist = Categorical(logits=throttle_logits)
        steer_dist = Categorical(logits=steer_logits)
        
        # Sample actions
        if throttle_action is None:
            throttle_action = throttle_dist.sample()
            steer_action = steer_dist.sample()
            
        # Compute log probabilities and entropy
        throttle_logprob = throttle_dist.log_prob(throttle_action)
        steer_logprob = steer_dist.log_prob(steer_action)
        logprob = throttle_logprob + steer_logprob  # Total logprob
        entropy = (throttle_dist.entropy() + steer_dist.entropy()).mean()
        
        return (
            torch.stack([throttle_action, steer_action], dim=1),
            logprob,
            entropy,
            self.critic(features),
            (throttle_logits, steer_logits)
        )

class PPOTrainer:
    """Handles PPO training logic for multi-agent environment"""
    
    def __init__(self, 
                 env,
                 policy: SharedPPOPolicy,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_coef: float = 0.2,
                 ent_coef: float = 0.01,
                 batch_size: int = 512,
                 minibatches: int = 4,
                 epochs: int = 4):
        
        self.env = env
        self.policy = policy
        self.optimizer = Adam(policy.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.minibatches = minibatches
        self.epochs = epochs
        
    def collect_rollouts(self, num_steps: int) -> dict:
        """Collect experiences from environment"""
        obs = self.env.reset()
        all_obs = []
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_dones = []
        all_values = []
        
        for _ in range(num_steps):
            # Convert to tensor
            obs_tensor = self._obs_to_tensor(obs)
            
            # Get actions and values
            with torch.no_grad():
                actions, logprobs, _, values, _ = self.policy.get_action_and_value(obs_tensor)
            
            # Step environment
            next_obs, rewards, dones, _ = self.env.step(actions.cpu().numpy())
            
            # Store transition
            all_obs.append(obs)
            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_rewards.append(torch.tensor(rewards))
            all_dones.append(torch.tensor(dones))
            all_values.append(values.flatten())
            
            obs = next_obs
        
        # Process rollouts
        batch = {
            "observations": self._stack_obs(all_obs),
            "actions": torch.cat(all_actions),
            "logprobs": torch.cat(all_logprobs),
            "rewards": torch.cat(all_rewards),
            "dones": torch.cat(all_dones),
            "values": torch.cat(all_values)
        }
        
        # Compute advantages and returns
        batch["advantages"], batch["returns"] = self._compute_gae(batch)
        return batch
    
    def _compute_gae(self, batch: dict) -> tuple:
        """Calculate Generalized Advantage Estimation"""
        rewards = batch["rewards"]
        values = batch["values"]
        dones = batch["dones"]
        
        advantages = []
        last_advantage = 0
        
        # Reverse computation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage
        
        advantages = torch.tensor(advantages)
        returns = advantages + values
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8), returns
    
    def train(self, batch: dict) -> None:
        """Perform PPO update"""
        # Convert numpy arrays to tensors
        obs_tensor = self._obs_to_tensor(batch["observations"])
        
        # Mini-batch updates
        indices = np.arange(len(batch["observations"]))
        minibatch_size = len(indices) // self.minibatches
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]
                
                # Get minibatch
                mb_obs = obs_tensor[idx]
                mb_actions = batch["actions"][idx]
                mb_logprobs = batch["logprobs"][idx]
                mb_advantages = batch["advantages"][idx]
                mb_returns = batch["returns"][idx]
                
                # Get new policy values
                _, new_logprobs, entropy, new_values, (throttle_logits, steer_logits) = \
                    self.policy.get_action_and_value(mb_obs, 
                                                   mb_actions[:, 0], 
                                                   mb_actions[:, 1])
                
                # Value loss
                value_loss = 0.5 * (new_values.flatten() - mb_returns).pow(2).mean()
                
                # Policy loss
                ratio = (new_logprobs - mb_logprobs).exp()
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean() * self.ent_coef
                
                # Total loss
                loss = policy_loss + value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        """Convert observation dict to tensor"""
        # Flatten all observation components
        return torch.FloatTensor(np.concatenate([
            obs["position"],
            obs["velocity"],
            obs["goal_vec"],
            obs["nearby_agents"].reshape(obs["nearby_agents"].shape[0], -1),
            obs["lane_info"]
        ], axis=1))
    
    def _stack_obs(self, obs_list: list) -> np.ndarray:
        """Stack observations from multiple steps"""
        return np.concatenate([
            np.concatenate([
                o["position"],
                o["velocity"],
                o["goal_vec"],
                o["nearby_agents"].reshape(o["nearby_agents"].shape[0], -1),
                o["lane_info"]
            ], axis=1) for o in obs_list
        ])
