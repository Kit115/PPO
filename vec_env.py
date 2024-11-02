import gymnasium as gym
import numpy as np

class VecEnv:
    def __init__(self, envs, auto_reset=True):
        self.envs = envs
        self.num_envs = len(envs) 
        self.auto_reset = auto_reset

        self.stats = {
            "episode_returns": [0 for _ in range(self.num_envs)]
        }

    @property
    def observation_space(self):
        return self.envs[0].observation_space
    @property
    def action_space(self):
        return self.envs[0].action_space

    def _convert_infos(self, infos):
        return {key: [i[key] for i in infos] for key in infos[0]}

    def reset(self):

        self.stats = {
            "episode_returns": [0 for _ in range(self.num_envs)]
        }

        observations, infos = [], []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        observations = np.stack(observations) 
        infos = self._convert_infos(infos)

        return observations, infos


    def step(self, actions):
        assert actions.shape[0] == self.num_envs
        
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        final_observations = []
        final_returns = []

        for i in range(self.num_envs):
            observation, reward, terminated, truncated, info = self.envs[i].step(actions[i])
            self.stats["episode_returns"][i] += reward
            final_returns.append(self.stats["episode_returns"][i])

            final_observations.append(observation)
            if terminated or truncated:
                observation, info = self.envs[i].reset()
                self.stats["episode_returns"][i] = 0


            observations.append(observation)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        observations = np.stack(observations)
        rewards = np.array(rewards)
        terminateds = np.array(terminated)
        truncateds = np.array(truncateds)
        infos = self._convert_infos(infos)
        infos["final_observations"] = np.stack(final_observations)
        infos["episode_returns"] = np.stack(final_returns)

        return observations, rewards, terminateds, truncateds, infos


if __name__ == "__main__":
    def env_fn():
        env = gym.make("LunarLander-v3", continuous=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    envs = VecEnv([env_fn() for _ in range(8)])

    obs, info = envs.reset()
    actions = np.random.randn(8, 2)

    obs, rew, term, trunc, info = envs.step(actions)
    while not term.any() and not trunc.any():
        obs, rew, term, trunc, info = envs.step(actions)
    print(info)





