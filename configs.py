import gymnasium as gym
from models import ContinuousCNNActorCritic, ContinuousMLPActorCritic
from models import GaussianDistribution, SquashedGaussianDistribution

def get_config(env):
    assert env in ["LunarLander", "BipedalWalker", "CarRacing"]
    if env == "LunarLander":
        def env_fn(mode):
            env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array" if mode=="test" else None)
            env = gym.wrappers.ClipAction(env)
            return env

        agent = ContinuousMLPActorCritic(obs_dim=8, act_dim=2, distribution_cls=SquashedGaussianDistribution)

        trainer_kwargs = {
            "ent_coef": 0.01,
            "record_test_vids": True,
            "recording_path": "LunarLanderRecordings",
            "recording_interval": 4,
            "num_test_episodes": 64,
        }

        train_steps = 500_000

    elif env == "BipedalWalker":
        def env_fn(mode):
            env = gym.make(
                "BipedalWalker-v3", 
                render_mode="rgb_array" if mode=="test" else None
            )
            env = gym.wrappers.ClipAction(env)
            return env

        agent = ContinuousMLPActorCritic(obs_dim=24, act_dim=4, distribution_cls=GaussianDistribution)
        trainer_kwargs = {
            "num_envs": 16,
            "minibatch_size": 128,
            "ent_coef": 0.01, 
            "record_test_vids": True,
            "recording_path": "WalkerRecordings",
            "recording_interval": 4,
            "num_test_episodes": 64,
        }

        train_steps = 2_000_000

    elif env == "CarRacing":
        def env_fn(mode):
            env = gym.make(
                "CarRacing-v3", 
                continuous=True, 
                render_mode="rgb_array" if mode=="test" else None
            )
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
            env = gym.wrappers.FrameStackObservation(env, 8)
            return env

        agent = ContinuousCNNActorCritic(act_dim=3)

        trainer_kwargs = {
            "num_envs": 16,
            "minibatch_size": 128,
            "ent_coef": 0.01, 
            "record_test_vids": True,
            "recording_path": "RaceRecordings",
            "recording_interval": 4,
            "num_test_episodes": 64,
        }
        
        train_steps = 4_000_000

    return env_fn, agent, trainer_kwargs, train_steps




