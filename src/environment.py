import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class DrivingEnv(gym.Env):
    """
    A simple 2D multi-agent driving environment.
    Each agent is represented with a state vector: [x, y, vx, vy].
    Actions are represented as (acceleration, steering) pairs.
    """
    def __init__(self, config=None):
        super(DrivingEnv, self).__init__()
        # Load configuration with default parameters
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 10)
        self.map_size = self.config.get("map_size", (100, 100))
        self.collision_threshold = self.config.get("collision_threshold", 2.0)
        
        # Observation: For each agent [x_position, y_position, velocity_x, velocity_y]
        obs_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf])
        # The observation space is a Box containing the state vectors of all agents.
        self.observation_space = spaces.Box(low=obs_low, high=obs_high,
                                            shape=(self.num_agents, 4),
                                            dtype=np.float32)
        
        # Action: Each agent takes a tuple of discrete actions.
        # First component for acceleration: 0 (decelerate), 1 (steady), 2 (accelerate)
        # Second component for steering: 0 (left), 1 (straight), 2 (right)
        # For simplicity, assume a uniform action space for all agents.
        self.individual_action_space = spaces.MultiDiscrete([3, 3])
        # We expect the provided actions to be a list of (acceleration, steering) for each agent.
        self.action_space = spaces.Tuple([self.individual_action_space] * self.num_agents)
        
        # Initialize environment state.
        self.state = None
        self.dt = self.config.get("dt", 0.1)
        self.reset()
        
    def reset(self):
        """
        Reset the environment to an initial state.
        Each agent is initialized at a random position within the map with zero velocity.
        """
        # State for each agent is [x, y, vx, vy]
        self.state = np.zeros((self.num_agents, 4), dtype=np.float32)
        for i in range(self.num_agents):
            self.state[i, 0] = np.random.uniform(0, self.map_size[0])  # x position
            self.state[i, 1] = np.random.uniform(0, self.map_size[1])  # y position
            # Start with zero velocity
            self.state[i, 2] = 0.0  # vx
            self.state[i, 3] = 0.0  # vy
        return self.state

    def step(self, actions):
        """
        Execute one time step within the environment.
        :param actions: List of action tuples [(acceleration_action, steering_action), ...] for each agent.
        :return: (state, rewards, done, info)
        """
        # Update each agent with a simple kinematic model.
        for i in range(self.num_agents):
            acc, steer = actions[i]
            # Map discrete acceleration (0, 1, 2) to actual acceleration values (-1, 0, 1)
            acc_value = (acc - 1) * 1.0  
            # Map discrete steering (0, 1, 2) to a small angle change in radians (-0.1, 0, 0.1)
            steer_angle = (steer - 1) * 0.1  
            
            # Compute the current speed magnitude.
            curr_v = np.linalg.norm(self.state[i, 2:4])
            # Simple physics update: new speed = current speed + acceleration * dt
            new_v = max(curr_v + acc_value * self.dt, 0)
            
            # Use the current velocity vector to define a heading.
            if curr_v > 0:
                heading = np.arctan2(self.state[i, 3], self.state[i, 2])
            else:
                heading = 0.0
            # Adjust heading based on steering input.
            heading += steer_angle
            
            # Update velocity components.
            self.state[i, 2] = new_v * np.cos(heading)
            self.state[i, 3] = new_v * np.sin(heading)
        
        # Update positions using vectorized operations.
        self.state[:, 0:2] = self.state[:, 0:2] + self.state[:, 2:4] * self.dt
        
        # Perform vectorized collision checking.
        collision_flags = self._check_collisions()
        
        # Compute rewards: a simple approach assigns a penalty on collision and rewards progress.
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            if collision_flags[i]:
                rewards[i] = -10.0  # Collision penalty
            else:
                rewards[i] = np.linalg.norm(self.state[i, 2:4])  # Reward for moving (can be refined)
        
        # Determine if the episode is done.
        # Here, we end the episode if any agent moves out of the defined map boundaries.
        out_of_bounds = (self.state[:, 0] < 0) | (self.state[:, 0] > self.map_size[0]) | \
                        (self.state[:, 1] < 0) | (self.state[:, 1] > self.map_size[1])
        done = {"__all__": bool(np.any(out_of_bounds))}
        info = {}  # Additional information can be added as needed.
        
        return self.state, rewards, done, info

    def _check_collisions(self):
        """
        Perform collision checking using a fully vectorized approach.
        Returns a boolean array indicating which agents are in collision.
        """
        positions = self.state[:, 0:2]
        # Compute pairwise distances between agents using broadcasting.
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Prevent an agent from colliding with itself.
        np.fill_diagonal(distances, np.inf)
        # Each agent is flagged if any other agent is closer than the collision threshold.
        collision_flags = np.any(distances < self.collision_threshold, axis=1)
        return collision_flags

    def render(self, mode="human"):
        """
        Render the current state of the environment using a simple matplotlib plot.
        """
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        plt.scatter(self.state[:, 0], self.state[:, 1], c="blue", s=50)
        plt.title("Multi-Agent Driving Environment")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()
