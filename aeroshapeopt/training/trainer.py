"""
RL training loop for airfoil shape optimization.

Orchestrates interaction between the PPO agent and the airfoil
environment, collecting rollouts and performing policy updates.

Author: Samarjith Biswas
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from aeroshapeopt.envs.airfoil_env import AirfoilOptEnv
from aeroshapeopt.models.ppo import PPOAgent, RolloutBuffer

logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    total_timesteps: int = 50000
    rollout_steps: int = 2048      # Steps per rollout before PPO update
    n_epochs: int = 10             # PPO epochs per update
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    hidden_dim: int = 256
    log_interval: int = 5          # Log every N rollouts
    checkpoint_dir: str = "checkpoints"
    env_config: dict = None

    def __post_init__(self):
        if self.env_config is None:
            self.env_config = {}
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


class AeroShapeTrainer:
    """Train PPO agent for airfoil optimization."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.env = AirfoilOptEnv(config.env_config)
        obs_dim = self.env.obs_dim
        act_dim = self.env.n_actions

        self.agent = PPOAgent(
            obs_dim=obs_dim, act_dim=act_dim,
            lr=config.learning_rate, gamma=config.gamma, lam=config.lam,
            clip_eps=config.clip_eps, entropy_coef=config.entropy_coef,
            n_epochs=config.n_epochs, batch_size=config.batch_size,
            hidden_dim=config.hidden_dim,
        )
        self.buffer = RolloutBuffer(config.rollout_steps, obs_dim, act_dim)
        self.history = {
            "episode_rewards": [], "episode_L_D": [], "episode_Cl": [],
            "policy_loss": [], "value_loss": [],
        }

    def train(self) -> dict:
        """Run the full training loop."""
        logger.info(f"Training PPO for {self.config.total_timesteps} timesteps")
        start = time.time()
        total_steps = 0
        n_rollouts = 0
        obs, info = self.env.reset()

        while total_steps < self.config.total_timesteps:
            self.buffer.reset()

            # Collect rollout
            for _ in range(self.config.rollout_steps):
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.buffer.store(obs, action, reward, value, log_prob, float(done))
                obs = next_obs
                total_steps += 1

                if done:
                    self.history["episode_rewards"].append(sum(self.env.episode_rewards))
                    self.history["episode_L_D"].append(info.get("L/D", 0))
                    self.history["episode_Cl"].append(info.get("Cl", 0))
                    obs, info = self.env.reset()

            # Compute advantages
            _, _, last_value = self.agent.select_action(obs)
            self.buffer.compute_gae(last_value, self.config.gamma, self.config.lam)

            # PPO update
            metrics = self.agent.update(self.buffer)
            self.history["policy_loss"].append(metrics["policy_loss"])
            self.history["value_loss"].append(metrics["value_loss"])

            n_rollouts += 1
            if n_rollouts % self.config.log_interval == 0:
                recent_rewards = self.history["episode_rewards"][-10:]
                recent_ld = self.history["episode_L_D"][-10:]
                avg_r = np.mean(recent_rewards) if recent_rewards else 0
                avg_ld = np.mean(recent_ld) if recent_ld else 0
                logger.info(
                    f"Steps: {total_steps}/{self.config.total_timesteps} | "
                    f"Avg Reward: {avg_r:.2f} | Avg L/D: {avg_ld:.2f} | "
                    f"PG Loss: {metrics['policy_loss']:.4f}"
                )

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed:.1f}s ({total_steps} steps)")
        return self.history

    def save(self, path: str = "checkpoints/ppo_agent.pt"):
        self.agent.save(path)

    def get_best_airfoil(self):
        """Return the best airfoil found during training."""
        return self.env.best_weights
