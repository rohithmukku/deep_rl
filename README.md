# Deep Reinforcement Learning Methods

This repository contains implementations of some popular DRL methods.

## Methods

- [x] Deep Q-Networks (DQN)
- [x] Vanilla Policy Gradient (VPG)
- [ ] Natural Policy Gradient (NPG)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed DDPG (TD3)
- [ ] Actor Critic (A2C)
- [ ] Asynchronous Actor Critic (A3C)

## Environments

- [x] Classic OpenAI Gym
- [x] MuJoCo
- [ ] Robotics

### Installation

For MuJoCo installation, refer to these links:

- https://github.com/openai/mujoco-py

## Usage

```shell
$ python main.py --help

main is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

agent: dqn, vpg


== Config ==
Override anything in the config (foo.bar=value)

agent:
  _target_: agent.dqn.DQNAgent
  obs_dim: ???
  act_dim: ???
  buffer_size: 10000
  min_size: 100
  batch_size: 32
  epsilon: 1
  epsilon_decay: 0.95
  min_epsilon: 0.01
  gamma: 0.99
  alpha: 0.001
  target_update: 100
env: CartPole-v0
num_episodes: 1000
device: cuda
plot: false


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```

## References

### Blogs/Tutorials

- https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
- https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

### Papers

- [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

### Code

- https://github.com/dongminlee94/deep_rl
