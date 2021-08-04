# Deep Reinforcement Learning Methods

This repository contains implementations of some of the popular DRL methods.

## Methods

- [x] Deep Q-Networks (DQN)
- [x] Vanilla Policy Gradient (VPG)
- [x] Vanilla Actor Critic (VAC)
- [x] Advantage Actor Critic (A2C)
- [ ] Natural Policy Gradient (NPG)
- [x] Proximal Policy Optimization (PPO)
- [x] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed DDPG (TD3)
- [ ] Asynchronous Advantage Actor Critic (A3C)
- [ ] Soft Actor Critic (SAC)

  > **NOTE:** PPO, DDPG aren't working properly, need to be improved.

## Environments

- Classic OpenAI Gym

  | Environment              | Observation Space | Action Space |
  | ------------------------ | ----------------- | ------------ |
  | CartPole-v1              | Box, 4            | Discrete, 2  |
  | Pendulum-v0              | Box, 3            | Box, 1       |
  | MountainCar-v0           | Box, 2            | Discrete, 3  |
  | MountainCarContinuous-v0 | Box, 2            | Box, 1       |
  | Acrobot-v1               | Box, 6            | Discrete, 3  |

- MuJoCo

  | Environment               | Observation Space | Action Space |
  | ------------------------- | ----------------- | ------------ |
  | Ant-v2                    | Box, 111          | Box, 8       |
  | HalfCheetah-v2            | Box, 17           | Box, 6       |
  | Hopper-v2                 | Box, 11           | Box, 3       |
  | Humanoid-v2               | Box, 376          | Box, 17      |
  | HumanoidStandup-v2        | Box, 376          | Box, 17      |
  | InvertedDoublePendulum-v2 | Box, 11           | Box, 1       |
  | InvertedPendulum-v2       | Box, 4            | Box, 1       |
  | Reacher-v2                | Box, 11           | Box, 2       |
  | Swimmer-v2                | Box, 8            | Box, 2       |
  | Walker2d-v2               | Box, 17           | Box, 6       |

- Robotics (Not implemented)

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
- [A2C](https://arxiv.org/pdf/1602.01783.pdf)
- [PPO](https://arxiv.org/pdf/1707.06347.pdf)
- [DDPG](https://arxiv.org/pdf/1509.02971.pdf)

### Code

- https://github.com/dongminlee94/deep_rl
- https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch
- https://github.com/nikhilbarhate99/PPO-PyTorch

## Todo

- Improve PPO, DDPG
- Unit testing
- Better logs, plots
