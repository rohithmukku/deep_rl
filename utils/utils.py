import gym
import torch
import matplotlib.pyplot as plt

env_type = {
    "classic": ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"],
    "mujoco": ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2"],
    "robotics": []
}

def make_env(env_name):
    env = gym.make(env_name)
    if env_name in env_type["classic"]:
        print("Classic Environment: " + env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
    elif env_name in env_type["mujoco"]:
        print("MuJoCo Environment: " + env_name)
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif env_name in env_type["robotics"]:
        print("Robotics Environment: " + env_name)
        env = gym.make(env_name)
        obs_dim = None
        act_dim = None

    return env, obs_dim, act_dim

def seed_torch(seed=100):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def plot_durations(episode_durations):
    fig = plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(durations_t.numpy(), label='Episode Reward')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='Average Reward')

    plt.legend()

    # plt.pause(0.001)
    fig.canvas.start_event_loop(0.001)