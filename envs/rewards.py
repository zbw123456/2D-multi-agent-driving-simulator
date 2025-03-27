import numpy as np
from typing import Dict, Optional

class RewardCalculator:
    """Modular reward system with batched calculations"""
    
    def __init__(self, 
                 env,
                 weights: Optional[Dict[str, float]] = None,
                 collision_penalty: float = -10.0,
                 goal_bonus: float = 50.0):
        self.env = env
        self.weights = weights or {
            "progress": 1.0,
            "collision": 1.0,
            "lane_center": 0.5,
            "heading_alignment": 0.3,
            "speed_limit": 0.2
        }
        self.collision_penalty = collision_penalty
        self.goal_bonus = goal_bonus
        self.last_positions = None

    def compute(self, 
                positions: np.ndarray,
                velocities: np.ndarray,
                headings: np.ndarray,
                goals: np.ndarray) -> np.ndarray:
        """Calculate total rewards for all agents (batched)"""
        self.last_positions = positions.copy()
        
        rewards = np.zeros(positions.shape[0])
        
        # Add individual components
        rewards += self._progress_reward(positions, goals)
        rewards += self._collision_penalty()
        rewards += self._lane_center_reward(positions)
        rewards += self._heading_alignment_reward(headings)
        rewards += self._speed_limit_reward(velocities)
        rewards += self._goal_bonus(positions, goals)
        
        return rewards

    def _progress_reward(self, 
                       current_pos: np.ndarray,
                       goals: np.ndarray) -> np.ndarray:
        """Reward for making progress toward goal"""
        if self.last_positions is None:
            return np.zeros(current_pos.shape[0])
            
        prev_dist = np.linalg.norm(goals - self.last_positions, axis=1)
        curr_dist = np.linalg.norm(goals - current_pos, axis=1)
        return (prev_dist - curr_dist) * self.weights["progress"]

    def _collision_penalty(self) -> np.ndarray:
        """Penalty for collisions"""
        collisions = self.env._check_collisions()
        return collisions.astype(float) * self.collision_penalty * self.weights["collision"]

    def _lane_center_reward(self, positions: np.ndarray) -> np.ndarray:
        """Reward for staying centered in lane"""
        # Simplified example - should integrate with map data
        lane_offsets = np.random.random(positions.shape[0])  # Mock data
        return -np.abs(lane_offsets) * self.weights["lane_center"]

    def _heading_alignment_reward(self, headings: np.ndarray) -> np.ndarray:
        """Reward for maintaining lane-appropriate heading"""
        # Simplified example - should use actual lane directions
        heading_diff = np.random.random(headings.shape[0])  # Mock data
        return -heading_diff * self.weights["heading_alignment"]

    def _speed_limit_reward(self, velocities: np.ndarray) -> np.ndarray:
        """Penalty/reward for speed limit adherence"""
        speeds = np.linalg.norm(velocities, axis=1)
        return -np.maximum(speeds - 5.0, 0) * self.weights["speed_limit"]

    def _goal_bonus(self, 
                  positions: np.ndarray,
                  goals: np.ndarray) -> np.ndarray:
        """Large bonus for reaching goal"""
        reached = np.linalg.norm(goals - positions, axis=1) < 3.0
        return reached.astype(float) * self.goal_bonus

    def add_custom_reward(self,
                        name: str,
                        weight: float,
                        reward_fn: callable):
        """Add new reward component dynamically"""
        self.weights[name] = weight
        setattr(self, f"_{name}_reward", reward_fn)
