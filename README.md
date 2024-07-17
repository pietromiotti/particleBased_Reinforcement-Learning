# Room Evacuation Particle-Based Model

## Summary

### Overview
The room evacuation particle-based model simulates the behavior of individuals (represented as particles) during an evacuation scenario. This model aims to understand and optimize the evacuation process, ensuring efficient and safe exit from a confined space.

### Model Description
In the particle-based model, individuals are represented as particles that move within a room towards an exit. The movement of these particles is influenced by various factors, including the distance to the exit, interactions with other particles, and obstacles within the room. The model employs principles from physics and social dynamics to realistically simulate the evacuation process.

### Key Features
- **Particle Representation:** Each individual is modeled as a particle with specific properties such as position, velocity, and preferred direction of movement.
- **Social Forces:** The model incorporates social force algorithms that account for the attraction to exits, repulsion from other particles, and avoidance of obstacles.
- **Crowd Dynamics:** Simulates the collective behavior of particles, including congestion and bottlenecks, which are common in real-life evacuation scenarios.
- **Customizable Parameters:** Allows adjustment of parameters such as room layout, number of exits, particle speed, and interaction forces to study different evacuation scenarios.

### Optimal Planning with Reinforcement Learning
In this project, I explored whether it was possible to achieve an optimal planning strategy for escaping from a room using Reinforcement Learning, reproducing the work done in [this paper](https://arxiv.org/abs/2012.00065). By training a neural network to identify the best action policy based on the state of the room, I designed an appropriate reward function to guide the learning process. This project was particularly enjoyable as it combined particle-based modeling with neural networks, demonstrating how a single agent can become "intelligent" and improve evacuation strategies.


https://github.com/user-attachments/assets/61a5157f-b87e-43af-a716-ae28da5e611c


### Applications
The room evacuation particle-based model has several practical applications:
- **Safety Planning:** Helps in designing safer building layouts and evacuation plans by simulating different emergency scenarios.
- **Event Management:** Assists in crowd management and planning for large events to ensure smooth and safe evacuations.
- **Policy Making:** Informs policies related to building codes and safety regulations by providing evidence-based insights into evacuation dynamics.

### Conclusion
The room evacuation particle-based model is a powerful tool for understanding and improving evacuation processes in confined spaces. By simulating the behavior of individuals during an evacuation, it helps identify potential issues and optimize strategies for safer and more efficient evacuations.

### Report
For a more deeper understanding, I suggest to read the project report "ParticleBased_RL". 


