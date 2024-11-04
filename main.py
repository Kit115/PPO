from PPO import PPOTrainer
from configs import get_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str, default="LunarLander")
args = parser.parse_args()

env_fn, agent, trainer_kwargs, train_steps = get_config(args.game)

trainer = PPOTrainer(
    env_fn, 
    agent, 
    **trainer_kwargs
)

trainer.train(train_steps)
print("Finished!")


