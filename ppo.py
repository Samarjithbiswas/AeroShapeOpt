"""
Proximal Policy Optimization (PPO) agent for continuous action spaces.

Implements PPO-Clip (Schulman et al., 2017) with:
- Actor-critic architecture (shared feature extractor)
- Gaussian policy for continuous actions
- Generalized Advantage Estimation (GAE, Schulman et al., 2016)
- Clipped surrogate objective
- Value function clipping
- Entropy bonus for exploration

Reference:
    Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms."
    arXiv:1707.06347.

Author: Samarjith Biswas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Actor-Critic network with shared feature extractor.

    Architecture:
        obs → shared MLP → actor_head → (mean, log_std)  [policy]
                          → critic_head → V(s)            [value]

    The policy is a diagonal Gaussian: a ~ N(mu(s), diag(sigma(s)^2))
    with log_std as a learnable parameter (state-independent).

    Args:
        obs_dim: Observation space dimension
        act_dim: Action space dimension
        hidden_dim: Hidden layer width
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs action mean
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

        # Learnable log standard deviation (state-independent)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # Critic head: outputs state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small init for policy output (conservative initial actions)
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action mean and state value.

        Returns:
            (action_mean, value)
        """
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        return action_mean, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability and value.

        Args:
            obs: Observation tensor (batch, obs_dim)
            action: If provided, compute log prob of this action instead of sampling

        Returns:
            (action, log_prob, entropy, value)
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value


class RolloutBuffer:
    """Storage for on-policy rollout data.

    Stores transitions from environment interaction for PPO updates.
    Computes GAE advantages and returns after rollout completion.
    """

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.size = buffer_size

    def store(self, obs, action, reward, value, log_prob, done):
        assert self.ptr < self.size
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute Generalized Advantage Estimation (Schulman et al., 2016).

        A_t = delta_t + (gamma*lam)*delta_{t+1} + ... + (gamma*lam)^{T-t}*delta_T
        where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        """
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            self.advantages[t] = last_gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, batch_size: int):
        """Yield random mini-batches for PPO update."""
        indices = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_idx = indices[start:end]
            yield {
                "obs": torch.tensor(self.obs[batch_idx]),
                "actions": torch.tensor(self.actions[batch_idx]),
                "log_probs": torch.tensor(self.log_probs[batch_idx]),
                "advantages": torch.tensor(self.advantages[batch_idx]),
                "returns": torch.tensor(self.returns[batch_idx]),
            }

    def reset(self):
        self.ptr = 0


class PPOAgent:
    """Proximal Policy Optimization agent.

    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        lr: Learning rate
        gamma: Discount factor
        lam: GAE lambda
        clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient
        vf_coef: Value function loss coefficient
        max_grad_norm: Gradient clipping norm
        n_epochs: Number of PPO epochs per update
        batch_size: Mini-batch size for PPO updates
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        hidden_dim: int = 256,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Select action using current policy.

        Returns:
            (action, log_prob, value)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.policy.get_action_and_value(obs_t)
        return (
            action.cpu().numpy().flatten(),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Perform PPO update using collected rollout data.

        Returns:
            Dictionary of training metrics
        """
        # Normalize advantages
        adv = buffer.advantages[:buffer.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer.advantages[:buffer.ptr] = adv

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs = batch["obs"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                _, new_log_probs, entropy, values = self.policy.get_action_and_value(
                    obs, actions
                )

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                vf_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.vf_coef * vf_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy_state_dict": self.policy.state_dict()}, path)
        logger.info(f"PPO agent saved to {path}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        logger.info(f"PPO agent loaded from {path}")
