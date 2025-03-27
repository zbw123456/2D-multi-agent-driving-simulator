import pygame
import numpy as np
from typing import Optional

class DrivingVisualizer:
    """Handles 2D visualization of the driving environment"""
    
    def __init__(self, env, screen_width: int = 800, screen_height: int = 800):
        self.env = env
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = None
        self.clock = None
        self.scale = 8  # Pixels per meter
        self.offset = np.array([screen_width // 2, screen_height // 2])

    def _init_pygame(self) -> None:
        """Initialize Pygame resources"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Multi-Agent Driving Environment")

    def render(self, agent_states: dict) -> None:
        """Render current frame"""
        if self.screen is None:
            self._init_pygame()
        
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw lanes
        for lane in self.env.map_data["lanes"]:
            start = lane["start"] * self.scale + self.offset
            end = lane["end"] * self.scale + self.offset
            pygame.draw.line(
                self.screen, 
                (0, 0, 0),  # Black
                start.astype(int),
                end.astype(int),
                int(lane["width"] * self.scale)
            )
        
        # Draw agents
        for pos, vel, _ in zip(agent_states["positions"], agent_states["velocities"], agent_states["headings"]):
            color = (
                min(255, int(255 * np.linalg.norm(vel) / 5)),  # Speed-based red
                0,
                0
            )
            screen_pos = pos * self.scale + self.offset
            pygame.draw.circle(
                self.screen,
                color,
                screen_pos.astype(int),
                5  # Radius
            )
        
        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        """Cleanup resources"""
        if self.screen is not None:
            pygame.quit()
