import numpy as np
import json
from typing import Dict, Tuple, Optional
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

class DrivingEnv:
    """2D Multi-Agent Driving Environment with Self-Play"""
    
    def __init__(self, 
                 map_config: str = "configs/intersection.json",
                 num_agents: int = 100,
                 max_steps: int = 500,
                 render_mode: Optional[str] = None):
        # Configuration
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Load map
        self._load_map(map_config)
        
        # State storage (batched)
        self.positions = np.zeros((num_agents, 2))  # (x, y)
        self.velocities = np.zeros((num_agents, 2))  # (vx, vy)
        self.headings = np.zeros(num_agents)  # radians
        self.goals = np.zeros((num_agents, 2))  # target positions
        
        # Spaces
        self.action_space = spaces.MultiDiscrete([3, 5])  # throttle (0: brake, 1: coast, 2: gas), steering bins
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "velocity": spaces.Box(low=-10, high=10, shape=(2,)),
            "goal_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "nearby_agents": spaces.Box(low=-10, high=10, shape=(5, 4)),  # 5 nearest agents (rel_pos, rel_vel)
            "lane_info": spaces.Box(low=-1, high=1, shape=(3,))  # (distance_to_center, lane_angle_diff, is_intersection)
        })

        # Reset environment
        self.reset()

    def _load_map(self, config_path: str) -> None:
        """Load lanes and intersections from JSON config"""
        with open(config_path) as f:
            self.map_data = json.load(f)
        
        # Precompute lane boundaries
        self.lanes = [
            {
                "start": np.array(lane["start"]),
                "end": np.array(lane["end"]),
                "width": lane["width"]
            } for lane in self.map_data["lanes"]
        ]

    def reset(self) -> Dict[str, ObsType]:
        """Reset all agents to initial positions"""
        # Random spawn positions in lanes
        lane = np.random.choice(self.lanes, self.num_agents)
        t = np.random.uniform(0, 1, self.num_agents)
        self.positions = np.stack([l["start"] * (1 - ti) + l["end"] * ti for l, ti in zip(lane, t)])
        
        # Initial velocities (forward direction)
        self.headings = np.arctan2(
            [l["end"][1] - l["start"][1] for l in lane],
            [l["end"][0] - l["start"][0] for l in lane]
        )
        speed = np.random.uniform(1, 3, self.num_agents)
        self.velocities = np.stack([
            speed * np.cos(self.headings),
            speed * np.sin(self.headings)
        ]).T

        # Random goals in opposite lanes
        self.goals = np.stack([np.random.choice([l["start"], l["end"]]) 
                             for l in np.random.choice(self.lanes, self.num_agents)])
        
        self.steps = 0
        return self._get_observations()

    def step(self, actions: ActType) -> Tuple[Dict[str, ObsType], np.ndarray, np.ndarray, Dict]:
        """Step all agents simultaneously"""
        # Parse actions
        throttle = actions[:, 0]  # 0-2
        steer_bin = actions[:, 1]  # 0-4
        
        # Convert discrete actions to continuous
        steer = np.deg2rad(steer_bin * 15 - 30)  # [-30°, 30°]
        accel = np.where(throttle == 2, 0.5, 
                       np.where(throttle == 1, 0.0, -1.0))
        
        # Update dynamics
        self._update_physics(accel, steer)
        
        # Get observations
        obs = self._get_observations()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check terminations
        dones = self._check_terminations()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()

        self.steps += 1
        return obs, rewards, dones, {"collisions": self._check_collisions()}

    def _update_physics(self, accel: np.ndarray, steer: np.ndarray) -> None:
        """Vectorized physics update"""
        # Update heading
        self.headings += steer * 0.1  # steering sensitivity
        
        # Update velocity
        direction = np.stack([np.cos(self.headings), np.sin(self.headings)]).T
        self.velocities += accel[:, None] * direction * 0.1
        
        # Clip velocity
        speed = np.linalg.norm(self.velocities, axis=1)
        self.velocities = np.clip(self.velocities, -10, 10)
        
        # Update position
        self.positions += self.velocities * 0.1  # dt=0.1

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Build observations for all agents"""
        # Relative positions and velocities
        diffs = self.positions[:, None] - self.positions  # (N, N, 2)
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)  # ignore self
        
        # For each agent, get 5 nearest neighbors
        nearby = np.argpartition(dists, 5, axis=1)[:, :5]
        nearby_obs = np.stack([
            diffs[np.arange(self.num_agents)[:, None], nearby],
            self.velocities[nearby] - self.velocities[:, None]
        ], axis=-1).reshape(self.num_agents, 5, 4)
        
        # Lane features
        lane_feats = np.array([self._get_lane_features(i) for i in range(self.num_agents)])
        
        return {
            "position": self.positions,
            "velocity": self.velocities,
            "goal_vec": self.goals - self.positions,
            "nearby_agents": nearby_obs,
            "lane_info": lane_feats
        }

    def _get_lane_features(self, agent_id: int) -> np.ndarray:
        """Calculate lane-related features for one agent"""
        # Project position onto nearest lane
        # (Implementation simplified for example)
        return np.array([0.0, 0.0, 0.0])  # [distance_to_center, angle_diff, is_intersection]

    def _calculate_rewards(self) -> np.ndarray:
        """Batched reward calculation"""
        # Progress reward
        progress = np.linalg.norm(self.goals - self.positions, axis=1)
        progress_reward = (self.prev_progress - progress) / 100
        self.prev_progress = progress.copy()
        
        # Collision penalty
        collision = self._check_collisions().astype(float) * -10
        
        # Lane keeping
        lane_dev = np.abs(self._get_lane_features()[:, 0]) * -0.1
        
        return progress_reward + collision + lane_dev

    def _check_collisions(self) -> np.ndarray:
        """Vectorized collision detection"""
        dists = np.linalg.norm(self.positions[:, None] - self.positions, axis=2)
        collisions = (dists < 2.0) & (dists > 0)  # 2m collision radius
        return np.any(collisions, axis=1)

    def _check_terminations(self) -> np.ndarray:
        """Check termination conditions"""
        out_of_bounds = np.any((self.positions < -100) | (self.positions > 100), axis=1)
        reached_goal = np.linalg.norm(self.goals - self.positions, axis=1) < 3.0
        timeout = (self.steps >= self.max_steps)
        return out_of_bounds | reached_goal | timeout

    def render(self) -> None:
        """Render using Pygame"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 800))
            self.clock = pygame.time.Clock()
        
        self.screen.fill((255, 255, 255))
        
        # Draw lanes
        for lane in self.lanes:
            pygame.draw.line(self.screen, (0, 0, 0),
                           lane["start"] * 8 + 400,
                           lane["end"] * 8 + 400,
                           int(lane["width"] * 8))
        
        # Draw agents
        for pos, vel in zip(self.positions, self.velocities):
            color = (255 * min(1, np.linalg.norm(vel)/5), 0, 0)  # red intensity = speed
            pygame.draw.circle(self.screen, color,
                             pos * 8 + 400, 5)
        
        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
