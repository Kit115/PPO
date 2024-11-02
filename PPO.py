import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim 
from collections import deque

from vec_env import VecEnv

class RolloutBuffer():
    def __init__(self, num_envs, n_steps, obs_dim, act_dim, gae_gamma=0.99, gae_lambda=0.95):
        self.obs_buf    = np.zeros((n_steps, num_envs, *obs_dim), dtype=np.float32)
        self.act_buf    = np.zeros((n_steps, num_envs, *act_dim), dtype=np.float32)
        self.logp_buf   = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.rew_buf    = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.done_buf   = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.val_buf    = np.zeros((n_steps, num_envs), dtype=np.float32)

        self.pointer = 0
        self.n_steps = n_steps

        self.ret_buf = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.adv_buf = np.zeros((n_steps, num_envs), dtype=np.float32)

        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda

    def append(self, *, obs, act, logp, rew, done, val):
        assert self.pointer < self.n_steps
        self.obs_buf[self.pointer]  = obs
        self.act_buf[self.pointer]  = act
        self.logp_buf[self.pointer] = logp
        self.rew_buf[self.pointer]  = rew
        self.done_buf[self.pointer] = done
        self.val_buf[self.pointer]  = val

        self.pointer += 1

    def calculate_advantage_and_returns(self, last_values, dones):
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - dones 
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - self.done_buf[t + 1]
                nextvalues = self.val_buf[t + 1]

            delta       = self.rew_buf[t] + self.gae_gamma * nextvalues * nextnonterminal - self.val_buf[t]
            lastgaelam  = delta + self.gae_gamma * self.gae_lambda * nextnonterminal * lastgaelam
            self.adv_buf[t] = lastgaelam
        self.ret_buf = self.adv_buf + self.val_buf

    def get_batch(self):
        assert self.pointer == self.n_steps
        self.pointer = 0
        return self.obs_buf, self.act_buf, self.logp_buf, self.ret_buf, self.adv_buf

    def is_full(self):
        return self.pointer == self.n_steps

    def is_empty(self):
        return self.pointer == 0


class PPOTrainer():
    def __init__(self,
        env_fn,
        agent,
        num_envs            = 8,
        num_steps           = 512, 
        gae_gamma           = 0.99,
        gae_lambda          = 0.95, 
        minibatch_size      = 64, 
        update_epochs       = 10, 
        learning_rate       = 3e-4,
        clip_coef           = 0.2, 
        ent_coef            = 0.0, 
        vf_coef             = 0.5,
        max_grad_norm       = 0.5, 
        step_limit          = None,
        num_test_episodes   = 128,
        stochastic_testing  = False,
        record_test_vids    = False,
        recording_interval  = 1,
        recording_path      = "recordings/",
        device              = "cpu",
    ):
        def make_env():
            env = env_fn("train")
            if step_limit is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=step_limit)
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env

        def make_test_env():
            env = env_fn("test")
            if step_limit is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=step_limit)
            if record_test_vids:
                env = gym.wrappers.RecordVideo(
                    env, 
                    recording_path,
                    episode_trigger = lambda x: x % int(recording_interval * num_test_episodes) == 0,
                    disable_logger = True
                )
            return env

        self.envs = VecEnv([make_env() for _ in range(num_envs)])
        self.test_env = make_test_env()
        self.num_envs       = num_envs

        self.agent          = agent.to(device)
        self.optimizer      = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(
            obs_dim     = self.envs.observation_space.shape,
            act_dim     = self.envs.action_space.shape,
            num_envs    = self.num_envs, 
            n_steps     = num_steps
        )

        self.num_steps      = num_steps
        self.gae_gamma      = gae_gamma
        self.gae_lambda     = gae_lambda
        self.batch_size     = num_steps * self.num_envs 
        self.minibatch_size = minibatch_size
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.update_epochs  = update_epochs
        self.clip_coef      = clip_coef
        self.ent_coef       = ent_coef
        self.vf_coef        = vf_coef
        self.max_grad_norm  = max_grad_norm
       
        self.episode_returns = deque(maxlen=100)

        self.num_test_episodes = num_test_episodes
        self.stochastic_testing = stochastic_testing

        self.device = device
 
    @torch.no_grad
    def collect_rollout(self):
        assert self.rollout_buffer.is_empty()

        while not self.rollout_buffer.is_full():
            action, logprob, _, value = self.agent.get_action_and_value(torch.from_numpy(self._last_obs).float().to(self.device))

            next_obs, reward, next_done, trunc, info = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(next_done, trunc)

            for idx in range(self.num_envs):
                if trunc[idx]: 
                    final_obs = torch.from_numpy(np.array(info['final_observations'][idx])).unsqueeze(0).float().to(self.device)
                    bootstrap_value = self.agent.get_value(final_obs).item()
                    reward[idx] += self.gae_gamma * bootstrap_value
                if next_done[idx]: 
                    episode_return = info['episode_returns'][idx].item()
                    self.episode_returns.append(episode_return)

            self.rollout_buffer.append(
                obs     = self._last_obs,
                act     = action.cpu().numpy(),
                logp    = logprob.cpu().numpy(),
                rew     = reward.reshape(-1),
                done    = self._last_done,
                val     = value.reshape(-1).cpu().numpy()
            )

            self._last_obs, self._last_done = next_obs, next_done

        last_values = self.agent.get_value(torch.from_numpy(self._last_obs).float().to(self.device)).reshape(1, -1).cpu().numpy()
        self.rollout_buffer.calculate_advantage_and_returns(last_values=last_values, dones=self._last_done)

    @torch.enable_grad
    def update(self):
        obs_buf, act_buf, logp_buf, ret_buf, adv_buf = self.rollout_buffer.get_batch()
        obs_batch   = torch.from_numpy(obs_buf).reshape((-1,) + self.envs.observation_space.shape).float().to(self.device)
        logp_batch  = torch.from_numpy(logp_buf).reshape(-1).float().to(self.device)
        act_batch   = torch.from_numpy(act_buf).reshape((-1,) + self.envs.action_space.shape).float().to(self.device)
        adv_batch   = torch.from_numpy(adv_buf).reshape(-1).float().to(self.device)
        ret_batch   = torch.from_numpy(ret_buf).reshape(-1).float().to(self.device)

        for epoch in range(self.update_epochs):
            random_idx = np.random.permutation(self.batch_size)
            for num_minibatch in range(self.num_minibatches):
                minibatch_idx = random_idx[self.minibatch_size*num_minibatch:self.minibatch_size*(num_minibatch+ 1)]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(obs_batch[minibatch_idx], act_batch[minibatch_idx])
                ratio = (newlogprob - logp_batch[minibatch_idx]).exp()

                adv_minibatch = adv_batch[minibatch_idx]
                adv_minibatch = (adv_minibatch - adv_minibatch.mean()) / (adv_minibatch.std() + 1e-8)

                pg_loss1 = -adv_minibatch * ratio
                pg_loss2 = -adv_minibatch * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = F.mse_loss(newvalue, ret_batch[minibatch_idx])

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

    @torch.no_grad()
    def evaluate_model(self):
        episode_returns = []
        for _ in range(self.num_test_episodes):
            obs, _ = self.test_env.reset()
            terminated, truncated = False, False
            ep_ret = 0
            
            while not terminated and not truncated:
                action = self.agent.get_action(
                    torch.from_numpy(obs).unsqueeze(0), 
                    deterministic = not self.stochastic_testing
                ).squeeze().cpu().numpy()

                obs, rew, terminated, truncated, _ = self.test_env.step(action)
                ep_ret += rew

            episode_returns.append(ep_ret)
        return sum(episode_returns) / len(episode_returns)

    def train(self, total_timesteps=1_000_000):
        self._last_obs, _   = self.envs.reset()
        self._last_done     = np.zeros(self.num_envs)

        global_step = 0

        while global_step < total_timesteps:
            global_step += self.num_envs * self.num_steps

            self.collect_rollout()
            self.update()
            
            evaluation_return = self.evaluate_model() 
            training_return = "N/A" if len(self.episode_returns) == 0 else f"{(sum(self.episode_returns) / len(self.episode_returns)):.2f}"
            print(f"Global Step: {global_step:06d}; Mean Training Return (Normalized): {training_return}; Mean Evaluation Return: {evaluation_return:.2f};", end="\n")
        
    def save(self, filename):
        torch.save(self.agent.state_dict(), filename)




