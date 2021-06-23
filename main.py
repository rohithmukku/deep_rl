import numpy as np
import hydra
from itertools import count
import matplotlib.pyplot as plt

from utils.utils import seed_torch, make_env, plot_durations
from agent.dqn import DQNAgent

from omegaconf import DictConfig, OmegaConf

class Environment(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        # pass
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = cfg.env
        self.num_episodes = cfg.num_episodes
        self.average_rewards_list = []
        self.cumulative_rewards_list = []
        self.episode_durations = []
        self.plot = cfg.plot

        if self.plot:
            plt.ion()

        self.env, self.obs_dim, self.act_dim = make_env(cfg.env)

        self.seed = 100
        np.random.seed(self.seed)
        seed_torch(self.seed)
        self.env.seed(self.seed)

        self.agent = hydra.utils.instantiate(cfg.agent, device=cfg.device, obs_dim=self.obs_dim, act_dim=self.act_dim)
    
    def run(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_rewards = 0
            # if episode % 10 == 0:
            #     print(episode)
            for t in count():
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                if done:
                    next_state = None
                self.agent.observe(state, action, next_state, reward)
                self.agent.train(done)
                self.agent.update(done)
                state = next_state

                total_rewards += reward

                if done:
                    self.episode_durations.append(t + 1)
                    if self.plot:
                        plot_durations(self.episode_durations)
                    break

            average_rewards = total_rewards / (t+1)
            self.average_rewards_list.append(average_rewards)
            self.cumulative_rewards_list.append(total_rewards)
    
    def eval(self):
        for i_episode in range(20):
            observation = self.env.reset()
            for t in range(100):
                self.env.render()
                print(observation)
                action = self.agent.act(observation)
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
    
    def close(self):
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()

@hydra.main(config_path="config", config_name="main")
def main(cfg):
    environment = Environment(cfg)
    environment.run()
    for i, reward in enumerate(environment.average_rewards_list):
        print("Episode: " + str(i) + "\t Total Reward: " + str(environment.cumulative_rewards_list[i]) + "\t Average Reward: " + str(reward))
    
    # # environment.eval()
    environment.close()

if __name__ == "__main__":
    main()