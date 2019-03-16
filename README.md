# Udacity Deep Reinforcement Learning Reacher

![Double-jointed arms moving to target locations](reacher.gif)

## Background
This project involves training 20 agents (in parallel) to move their double-jointed arm to target locations and keep it there.

### State Space
There are 33 dimensions comprised of the agent's position, rotation, velocity, and angular velocities of the arm.

### Action Space
It is a continuous action space in a form of a vector of size 4 corresponding to torque applicable to two joints.

### Reward
+0.1 for each step that the agent's hand is in the goal location.

### Benchmark Mean Reward
The environment is considered solved for an average reward of +30 over 100 consecutive episodes and over all 20 agents.


## Installation
1. Follow this link to get started:

https://github.com/udacity/deep-reinforcement-learning#dependencies

2. Navigate to `deep-reinforcement-learning` directory in your `drlnd` environment
```
cd deep-reinforcement-learning
```

3. Clone the repo
```
git clone https://github.com/rtmink/udacity-drl-reacher.git
```

4. Navigate to `udacity-drl-reacher` folder
```
cd udacity-drl-reacher
```

5. Unzip the unity environment
```
unzip Reacher.app.zip
```

## Training & Report
Run the following command in the `udacity-drl-reacher` folder:
```
jupyter notebook
```

In the notebook, refer to `Report.ipynb` to see how the agents are implemented and trained. The implementation includes the model architecture of the neural networks. A plot of rewards per episode averaged over all 20 agents is also shown to show the number of episodes needed to solve the environment. Lastly, it highlights ideas for future work.

## Evaluation
Refer to `Reacher.ipynb` to see how the trained agents perform in Unity Reacher environment built for Udacity.
