"""
Gymnasium environment for airfoil shape optimization via RL.

The agent observes the current airfoil shape (CST weights) and its
aerodynamic performance, then takes actions that modify the CST
weights to improve the objective (maximize L/D, maximize Cl, etc.).

State space: [CST_weights_upper, CST_weights_lower, Cl, Cd, L/D, alpha, step/max_steps]
Action space: delta_CST_weights (continuous, bounded perturbations)

This follows the MDP formulation from Viquerat et al. (2021) and
the Gymnasium API standard (Towers et al., 2024).

Reference:
    Viquerat, J. et al. (2021). "Direct shape optimization through
    deep reinforcement learning." J. Comp. Physics.

Author: Samarjith Biswas
"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np

# Use gymnasium if available, fall back to a minimal interface
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from aeroshapeopt.physics.cst import (
    CSTAirfoil, generate_airfoil_coordinates, naca0012_cst_weights,
    compute_thickness,
)
from aeroshapeopt.physics.aero_solver import evaluate_airfoil, AeroCoefficients


# Default configuration
DEFAULT_CONFIG = {
    "cst_order": 6,                  # Bernstein polynomial order (7 weights per surface)
    "alpha_deg": 5.0,                # Design angle of attack
    "Re": 1e6,                       # Reynolds number
    "max_steps": 50,                 # Maximum optimization steps per episode
    "action_scale": 0.02,            # Max perturbation per step
    "objective": "L/D",              # 'L/D', 'Cl', '-Cd'
    "min_thickness": 0.06,           # Minimum thickness constraint (6% chord)
    "max_thickness": 0.25,           # Maximum thickness constraint
    "thickness_penalty": 5.0,        # Penalty weight for thickness violation
    "smoothness_penalty": 1.0,       # Penalty for non-smooth shapes
    "initial_airfoil": "naca0012",   # Starting shape
    "n_surface_points": 80,          # Points for airfoil discretization
}


class AirfoilOptEnv:
    """RL environment for airfoil shape optimization.

    The agent iteratively modifies CST Bernstein coefficients to
    optimize an aerodynamic objective while satisfying geometric
    constraints (minimum thickness, smoothness).

    Compatible with Gymnasium API (if installed) or standalone.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.cst_order = self.config["cst_order"]
        self.n_weights = self.cst_order + 1  # Per surface
        self.n_actions = 2 * self.n_weights   # Upper + lower weights
        self.alpha = self.config["alpha_deg"]
        self.Re = self.config["Re"]
        self.max_steps = self.config["max_steps"]
        self.action_scale = self.config["action_scale"]

        # Observation: [weights_upper, weights_lower, Cl, Cd, L/D, alpha_norm, step_frac]
        self.obs_dim = 2 * self.n_weights + 5

        # Define spaces (Gymnasium-compatible)
        if HAS_GYM:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )

        # State
        self.airfoil: Optional[CSTAirfoil] = None
        self.aero: Optional[AeroCoefficients] = None
        self.step_count = 0
        self.best_reward = -np.inf
        self.best_weights = None
        self.episode_rewards = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial airfoil.

        Returns:
            (observation, info) tuple
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize airfoil
        if self.config["initial_airfoil"] == "naca0012":
            w_upper, w_lower = naca0012_cst_weights(self.cst_order)
        else:
            # Random perturbation of NACA 0012
            w_upper, w_lower = naca0012_cst_weights(self.cst_order)
            w_upper += np.random.randn(self.n_weights) * 0.01
            w_lower += np.random.randn(self.n_weights) * 0.01

        self.airfoil = CSTAirfoil(
            weights_upper=w_upper.copy(),
            weights_lower=w_lower.copy(),
        )

        self.step_count = 0
        self.best_reward = -np.inf
        self.episode_rewards = []

        # Evaluate initial aerodynamics
        self._evaluate()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one optimization step.

        Args:
            action: Perturbation to CST weights, shape (2*n_weights,)
                    Values in [-1, 1], scaled by action_scale

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.clip(action, -1.0, 1.0)

        # Apply action: modify CST weights
        delta_upper = action[:self.n_weights] * self.action_scale
        delta_lower = action[self.n_weights:] * self.action_scale

        self.airfoil.weights_upper += delta_upper
        self.airfoil.weights_lower += delta_lower

        self.step_count += 1

        # Evaluate new aerodynamics
        self._evaluate()

        # Compute reward
        reward = self._compute_reward()
        self.episode_rewards.append(reward)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = (
                self.airfoil.weights_upper.copy(),
                self.airfoil.weights_lower.copy(),
            )

        # Termination conditions
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _evaluate(self) -> None:
        """Evaluate aerodynamic performance of current airfoil."""
        try:
            coords = generate_airfoil_coordinates(self.airfoil, self.config["n_surface_points"])
            self.aero = evaluate_airfoil(coords, self.alpha, self.Re)
        except Exception:
            self.aero = AeroCoefficients(
                Cl=0.0, Cd=1.0, Cd_pressure=1.0, Cd_friction=0.0,
                Cm=0.0, L_over_D=0.0, Cp_surface=np.zeros(1), converged=False,
            )

    def _compute_reward(self) -> float:
        """Compute reward signal.

        Reward = objective_value - constraint_penalties

        Constraints:
            1. Minimum thickness (structural requirement)
            2. Maximum thickness (drag consideration)
            3. Shape smoothness (manufacturability)
        """
        if not self.aero.converged:
            return -10.0

        # Objective
        obj = self.config["objective"]
        if obj == "L/D":
            objective_value = self.aero.L_over_D / 50.0  # Normalize ~50 to ~1
        elif obj == "Cl":
            objective_value = self.aero.Cl
        elif obj == "-Cd":
            objective_value = -self.aero.Cd * 100  # Negative drag, scaled
        else:
            objective_value = self.aero.L_over_D / 50.0

        # Thickness constraint penalty
        max_t, _ = compute_thickness(self.airfoil)
        penalty = 0.0

        if max_t < self.config["min_thickness"]:
            penalty += self.config["thickness_penalty"] * (
                self.config["min_thickness"] - max_t
            )
        if max_t > self.config["max_thickness"]:
            penalty += self.config["thickness_penalty"] * (
                max_t - self.config["max_thickness"]
            )

        # Smoothness penalty: penalize large second derivatives of CST weights
        d2_upper = np.diff(self.airfoil.weights_upper, n=2)
        d2_lower = np.diff(self.airfoil.weights_lower, n=2)
        smoothness = np.sum(d2_upper**2) + np.sum(d2_lower**2)
        penalty += self.config["smoothness_penalty"] * smoothness

        reward = objective_value - penalty

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = np.concatenate([
            self.airfoil.weights_upper,
            self.airfoil.weights_lower,
            np.array([
                self.aero.Cl,
                self.aero.Cd * 100,  # Scale for numerical stability
                self.aero.L_over_D / 50.0,
                self.alpha / 20.0,
                self.step_count / self.max_steps,
            ]),
        ]).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        """Return diagnostic information."""
        max_t, t_loc = compute_thickness(self.airfoil)
        return {
            "Cl": self.aero.Cl,
            "Cd": self.aero.Cd,
            "L/D": self.aero.L_over_D,
            "Cm": self.aero.Cm,
            "max_thickness": max_t,
            "thickness_location": t_loc,
            "step": self.step_count,
            "converged": self.aero.converged,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render current airfoil (for visualization)."""
        coords = generate_airfoil_coordinates(self.airfoil)
        return coords
