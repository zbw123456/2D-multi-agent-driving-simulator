import time
import numpy as np
import torch
from driving_env import DrivingEnv
from shared_ppo import SharedPPOPolicy, PPOTrainer
from visualization import DrivingVisualizer
from utils.logger import Logger  # Assume a simple logger exists

def main():
    # Configuration
    NUM_AGENTS = 100
    TOTAL_TIMESTEPS = 100_000
    LOG_INTERVAL = 10
    
    # Initialize components
    env = DrivingEnv(num_agents=NUM_AGENTS)
    policy = SharedPPOPolicy(input_dim=29)  # 29D observation space
    trainer = PPOTrainer(env, policy)
    visualizer = DrivingVisualizer(env)
    logger = Logger()
    
    # Training loop
    start_time = time.time()
    for update in range(TOTAL_TIMESTEPS // trainer.batch_size):
        # Collect experiences and update policy
        batch = trainer.collect_rollouts(trainer.batch_size)
        trainer.train(1)  # One training iteration
        
        # Log metrics
        if update % LOG_INTERVAL == 0:
            avg_reward = batch["rewards"].mean().item()
            collision_rate = (batch["dones"].sum() / NUM_AGENTS).item()
            logger.log({
                "timestep": update * trainer.batch_size,
                "avg_reward": avg_reward,
                "collision_rate": collision_rate,
                "entropy": batch["entropy"].item()
            })
            
            # Render sample episode
            visualizer.render({
                "positions": env.positions,
                "velocities": env.velocities,
                "headings": env.headings
            })
    
    # Save final policy
    torch.save(policy.state_dict(), "models/shared_ppo_final.pt")
    print(f"Training completed in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()
