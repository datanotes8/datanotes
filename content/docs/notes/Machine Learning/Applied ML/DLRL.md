---
title: "Reinforcement Learning"
draft: false
---
# Reinforcement Learning
An agent learns to make decisions by taking actions in an environment to maximize cumulative rewards. Unlike supervised learning, where the model learns from labeled data, RL relies on feedback from the environment in the form of rewards or penalties.


## Key Concepts
- **Agent**: The learner or decision-maker that interacts with the environment.
- **Environment**: The external system with which the agent interacts.
- **State**: A representation of the current situation of the agent within the environment.
- **Action**: A set of all possible moves the agent can make.
- **Reward**: A scalar feedback signal received after taking an action, indicating the immediate benefit of that action.
- **Policy**: A strategy used by the agent to determine the next action based on the current state.

## States and Observations
- State s is the complete description of the environment at a given time.
- Partially observed: agent can only see part of the state.
- Fully observed: agent can see the entire state.

## Action Space
- Discrete: finite set of actions for the agent (e.g., move left, right, up, down).
- Continuous: infinite possible actions (e.g., steering angle, speed).

## Policies
- Rule used by agent to decide actions based on states. It can be deterministic or stochastic.
- Deterministic Policy: Maps states to specific actions. The equation is:
```math
    a = \mu(s)
```
- Stochastic Policy: Maps states to a probability distribution over actions. The equation is \( \pi(a|s) = P(A=a|S=s) \).
```math
    a ~=~ \pi(.|s)
```