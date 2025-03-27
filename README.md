Bosch thesis project: 2D multi-agent driving simulator

This project is aimed at building a 2D multi-agent driving simulator where multiple vehicles are controlled via a single shared policy using self-play, inspired by large‐scale multi-agent reinforcement learning methods as exemplified in the Gigaflow paper[1]. 

Overview

The assignment challenges you to design and implement a multi-agent environment that supports realistic self-play scenarios. The focus is on creating an efficient, scalable simulation that can manage large batches of agents while integrating with a reinforcement learning (RL) method such as PPO (Proximal Policy Optimization)[1].

Project Objectives

- Multi-Agent Simulation: Develop a 2D driving environment where multiple vehicles interact and learn through self-play using a shared RL policy.
- RL Integration: Implement and demonstrate partial or full integration with an RL algorithm (e.g., PPO) to assess learning efficiency and performance in a multi-agent setting.
- Rapid Prototyping: Emphasize quick iteration and adaptation in environment building, showcasing your ability to prototype robust multi-agent systems rapidly[1].

Key Components

| Component                 | Details                                                                                                                                                                 |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Environment Efficiency  | Ensure large-batch scalability (e.g., handling 1000+ agents) using vectorized or batched operations for transformations, collision detection, etc.                    |
| Map Representation      | Encode roads, intersections, and lane boundaries in an easily queryable format that supports extension to different scenarios or custom map layouts.                     |
| Reward Calculation      | Design a flexible and efficient reward computation system that easily allows the addition of new reward components, including collision penalties, lane adherence, etc. |
| Policy Training         | Integrate the simulation with an RL method, handle multi-agent data batching, and monitor training progress through metrics like collision and success rates.            |
| Policy Design           | Implement a straightforward model architecture (e.g., a basic multilayer perceptron) with discrete actions (throttle/brake and steering bins) and allow for reward weighting. |

Each of these components is intended to test your ability to design a system that not only simulates realistic driving scenarios but also scales well when subjected to high computational loads[1].

Design & Implementation Guidelines

- Scenario Choice: Focus on a single driving scenario, such as an intersection, multi-lane highway, or traffic light, that best demonstrates the environment's capability to manage self-play. The environment should be modular to ease the addition of new scenarios without major re-engineering.
- Visualization & Debugging: Include a visualization tool (even a simple 2D rendering or console-based view) to monitor and debug agent behaviors.
- Integration with RL Libraries: Demonstrate how your environment integrates with standard RL libraries, ensuring correct handling of multi-agent batch data during policy updates.
- Code Organization: Prepare your repository on GitHub with a clear folder structure and include thorough documentation (README or Wiki) that explains design choices, performance challenges, and areas for future improvement.
- Partial Solutions are Acceptable: The project is designed with a limited timeframe in mind, so demonstrating a basic working solution with partial training results and logs is acceptable[1].

Evaluation Criteria

- Efficiency & Scalability: Ability of the simulation to handle large-scale agent interactions without significant performance degradation.
- Extensibility: How easily new environmental scenarios, map configurations, or reward components can be integrated.
- RL Integration: Correct and effective integration with RL training processes, including batching of rollouts and policy updates.
- Visualization & Debug Tools: Availability of mechanisms to monitor agent behavior and assess training performance.
- Documentation and Code Quality: Clear, well-structured code hosted on GitHub with adequate documentation explaining design decisions, encountered challenges, and future directions for improvement[1].
  
Installation
pip install -r requirements.txt

replace torch with CUDA-enabled version
torch>=2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121


Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/56721543/97bdddc5-f82a-451d-905a-c2f5798c5abd/20250306-Take-Home-Project_-Multi-Agent-Intersection-with-a-Shared-RL-Policy.docx
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/56721543/97bdddc5-f82a-451d-905a-c2f5798c5abd/20250306-Take-Home-Project_-Multi-Agent-Intersection-with-a-Shared-RL-Policy.docx
