import time
import matplotlib.pyplot as plt

class Logger:
    """Simple metric logger with visualization"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        
    def log(self, metrics_dict: dict) -> None:
        """Store metrics and print summary"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Print summary
        print(f"\nStep {metrics_dict['timestep']} | "
              f"Avg Reward: {metrics_dict['avg_reward']:.2f} | "
              f"Collisions: {metrics_dict['collision_rate']:.2%} | "
              f"Entropy: {metrics_dict['entropy']:.2f}")
        
    def plot(self) -> None:
        """Generate training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.plot(self.metrics['timestep'], self.metrics['avg_reward'])
        plt.title("Average Reward")
        
        plt.subplot(132)
        plt.plot(self.metrics['timestep'], self.metrics['collision_rate'])
        plt.title("Collision Rate")
        
        plt.subplot(133)
        plt.plot(self.metrics['timestep'], self.metrics['entropy'])
        plt.title("Policy Entropy")
        
        plt.tight_layout()
        plt.savefig("training_curves.png")
