# AeroShapeOpt

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deep Reinforcement Learning for Aerodynamic Shape Optimization**

AeroShapeOpt trains a PPO agent to iteratively modify airfoil shapes to maximize aerodynamic efficiency (L/D ratio) while satisfying geometric constraints. The agent learns a generalizable optimization policy that operates in the CST (Class-Shape Transformation) parameter space, evaluated using a Hess-Smith panel method.

---

## The Problem

Airfoil shape optimization is a challenging inverse design problem: given aerodynamic objectives (maximize L/D, target Cl), find the geometry that achieves them. Traditional approaches require thousands of expensive CFD evaluations per optimization. A trained RL agent can propose near-optimal modifications in milliseconds.

## Approach

The optimization is formulated as a Markov Decision Process (MDP):

- **State**: CST weights + current aero coefficients (Cl, Cd, L/D)
- **Action**: Perturbation to CST Bernstein coefficients (continuous, bounded)
- **Reward**: Objective improvement - constraint penalties (thickness, smoothness)
- **Environment**: Panel method solver evaluates each proposed shape

```
    ┌──────────────────────────────────────────────┐
    │              PPO Agent (Actor-Critic)          │
    │  ┌────────────┐         ┌───────────────────┐ │
    │  │ Actor:     │         │ Critic:           │ │
    │  │ π(a|s)     │         │ V(s)              │ │
    │  │ → Δ(CST)   │         │ → expected return │ │
    │  └─────┬──────┘         └───────────────────┘ │
    └────────┼─────────────────────────────────────-─┘
             │ action: modify CST weights
             ▼
    ┌────────────────────────────────────┐
    │        Airfoil Environment          │
    │  CST Parameterization (Kulfan)     │
    │  → Airfoil coordinates             │
    │  → Panel Method (Hess-Smith)       │
    │  → Cl, Cd, L/D, Cp                │
    │  → Reward + constraints            │
    └────────────────────────────────────┘
```

## Key Features

- **CST Parameterization** (Kulfan, 2008): Class function × Bernstein polynomial shape function. Smooth, low-dimensional design space with guaranteed valid airfoils.
- **PPO-Clip** (Schulman et al., 2017): Standalone implementation with GAE, clipped surrogate objective, actor-critic architecture.
- **Hess-Smith Panel Method**: Source + vortex panels with Kutta condition. Includes empirical skin friction drag estimate (Schlichting, 1979).
- **Gymnasium-Compatible Environment**: Standard RL API (reset, step, render) for plug-and-play with Stable-Baselines3 or custom agents.
- **Constraint Handling**: Minimum/maximum thickness, smoothness penalties, with configurable weights.

## Physics

### CST Parameterization
```
z(ψ) = C(ψ) · S(ψ) + ψ · Δz_TE

C(ψ) = ψ^0.5 · (1-ψ)^1.0         [round nose, sharp TE]
S(ψ) = Σ A_i · B_{i,n}(ψ)         [Bernstein polynomial shape function]
B_{i,n} = C(n,i) · ψ^i · (1-ψ)^{n-i}  [Bernstein basis]
```

### Aerodynamic Evaluation
- Inviscid: Hess-Smith panel method → Cp → Cl, Cd_pressure
- Viscous estimate: Cf = 0.455 / (log₁₀ Re)^2.58 (Schlichting turbulent flat plate)
- Total: Cd = Cd_pressure + 2·Cf

## Installation

```bash
git clone https://github.com/Samarjithbiswas/AeroShapeOpt.git
cd AeroShapeOpt
pip install -e .
```

## Quick Start

```python
from aeroshapeopt.training.trainer import AeroShapeTrainer, RLConfig

config = RLConfig(total_timesteps=20000)
trainer = AeroShapeTrainer(config)
history = trainer.train()
trainer.save()
```

## References

- Kulfan, B.M. (2008). Universal Parametric Geometry Representation Method. *J. Aircraft*, 45(1), 142-158.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Hess, J.L. & Smith, A.M.O. (1967). Calculation of Potential Flow About Arbitrary Bodies. *Progress in Aerospace Sciences*, 8, 1-138.
- Viquerat, J. et al. (2021). Direct shape optimization through deep reinforcement learning. *J. Comp. Physics*.

## Author

**Samarjith Biswas, Ph.D.**
Research Scientist III, University of Arizona, New Frontiers of Sound (NewFoS) Center
[samarjithbiswas.com](https://samarjithbiswas.com)
