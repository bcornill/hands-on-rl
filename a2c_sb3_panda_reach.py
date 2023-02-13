import gym
import panda_gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 500000,
    "env_name": "PandaReachJointsDense-v2",
}

run = wandb.init(
    project="pandareach",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
model = A2C(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=10000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)

run.finish()