import numpy as np
import pygame
import random

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 10
AGENT_SIZE = 10
SPEED = 2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Create game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Multi-Agent Driving Simulator")

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = random.choice([0, 90, 180, 270])  # Random start direction

    def move(self):
        if self.direction == 0:
            self.y -= SPEED
        elif self.direction == 90:
            self.x += SPEED
        elif self.direction == 180:
            self.y += SPEED
        elif self.direction == 270:
            self.x -= SPEED
        
        # Keep within bounds
        self.x = max(0, min(WIDTH - AGENT_SIZE, self.x))
        self.y = max(0, min(HEIGHT - AGENT_SIZE, self.y))

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.x, self.y, AGENT_SIZE, AGENT_SIZE))

# Create agents
agents = [Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(AGENT_COUNT)]

# Game loop
running = True
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    for agent in agents:
        agent.move()
        agent.draw(screen)
    
    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
