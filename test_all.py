"""Tests for AeroShapeOpt. Author: Samarjith Biswas"""
import pytest
import numpy as np
from aeroshapeopt.physics.cst import (
    CSTAirfoil, class_function, bernstein_basis, shape_function,
    cst_surface, generate_airfoil_coordinates, naca0012_cst_weights,
    compute_thickness, compute_leading_edge_radius,
)
from aeroshapeopt.physics.aero_solver import evaluate_airfoil
from aeroshapeopt.envs.airfoil_env import AirfoilOptEnv


class TestClassFunction:
    def test_boundary_values(self):
        """C(0) = 0, C(1) = 0 for N1=0.5, N2=1.0."""
        psi = np.array([1e-10, 0.5, 1.0 - 1e-10])
        C = class_function(psi, 0.5, 1.0)
        assert C[0] < 0.01  # Near zero at LE
        assert C[-1] < 0.01  # Near zero at TE
        assert C[1] > 0  # Positive in middle

    def test_round_nose_exponent(self):
        """N1=0.5 gives sqrt(x) behavior at LE (round nose)."""
        psi = np.array([0.01, 0.04])
        C = class_function(psi, 0.5, 1.0)
        # C(0.04)/C(0.01) should be ~sqrt(4) = 2 (approximately)
        ratio = C[1] / C[0]
        assert abs(ratio - 2.0) < 0.3  # sqrt scaling (approximate due to (1-x) term)


class TestBernsteinBasis:
    def test_partition_of_unity(self):
        """Sum of Bernstein basis functions equals 1 (partition of unity property)."""
        psi = np.linspace(0, 1, 50)
        for n in [3, 5, 8]:
            B = bernstein_basis(psi, n)
            sums = B.sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_shape(self):
        psi = np.linspace(0, 1, 20)
        B = bernstein_basis(psi, 5)
        assert B.shape == (20, 6)

    def test_non_negative(self):
        """Bernstein polynomials are non-negative on [0, 1]."""
        psi = np.linspace(0, 1, 100)
        B = bernstein_basis(psi, 7)
        assert np.all(B >= -1e-15)


class TestCSTAirfoil:
    @pytest.fixture
    def naca0012(self):
        w_u, w_l = naca0012_cst_weights(6)
        return CSTAirfoil(weights_upper=w_u, weights_lower=w_l)

    def test_naca0012_thickness(self, naca0012):
        """NACA 0012 should have max thickness ~12%."""
        max_t, _ = compute_thickness(naca0012)
        assert abs(max_t - 0.12) < 0.02

    def test_naca0012_symmetric(self, naca0012):
        """NACA 0012 is symmetric: upper = -lower."""
        np.testing.assert_allclose(
            naca0012.weights_upper, -naca0012.weights_lower, atol=1e-10
        )

    def test_coordinates_closed(self, naca0012):
        """First and last coordinates should be near TE."""
        coords = generate_airfoil_coordinates(naca0012, n_points=80)
        dist = np.linalg.norm(coords[0] - coords[-1])
        assert dist < 0.05

    def test_coordinates_shape(self, naca0012):
        coords = generate_airfoil_coordinates(naca0012, n_points=50)
        assert coords.shape[1] == 2
        assert len(coords) == 2 * 50 - 1

    def test_le_radius_positive(self, naca0012):
        r = compute_leading_edge_radius(naca0012)
        assert r > 0

    def test_surface_at_boundaries(self, naca0012):
        """z(0) = 0 and z(1) ≈ dz_te."""
        psi = np.array([1e-10, 1.0 - 1e-10])
        z = cst_surface(psi, naca0012.weights_upper)
        assert abs(z[0]) < 0.01
        assert abs(z[-1]) < 0.01  # dz_te = 0


class TestAeroSolver:
    @pytest.fixture
    def naca0012_coords(self):
        w_u, w_l = naca0012_cst_weights(6)
        airfoil = CSTAirfoil(weights_upper=w_u, weights_lower=w_l)
        return generate_airfoil_coordinates(airfoil, n_points=80)

    def test_zero_aoa_symmetric(self, naca0012_coords):
        """Symmetric airfoil at 0 AoA: Cl ~ 0."""
        aero = evaluate_airfoil(naca0012_coords, alpha_deg=0.0)
        assert abs(aero.Cl) < 0.3  # Should be near zero

    def test_coefficients_bounded(self, naca0012_coords):
        """Cl and Cd should be in reasonable ranges."""
        aero = evaluate_airfoil(naca0012_coords, alpha_deg=5.0)
        assert -2.0 < aero.Cl < 2.0
        assert 0.0 < aero.Cd < 1.0

    def test_drag_positive(self, naca0012_coords):
        """Drag should always be positive."""
        aero = evaluate_airfoil(naca0012_coords, alpha_deg=3.0)
        assert aero.Cd > 0

    def test_friction_drag_present(self, naca0012_coords):
        """Should include skin friction estimate."""
        aero = evaluate_airfoil(naca0012_coords, alpha_deg=0.0)
        assert aero.Cd_friction > 0


class TestRLEnvironment:
    @pytest.fixture
    def env(self):
        return AirfoilOptEnv({"max_steps": 10, "cst_order": 4})

    def test_reset_returns_observation(self, env):
        obs, info = env.reset(seed=42)
        assert len(obs) == env.obs_dim
        assert "Cl" in info

    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = np.zeros(env.n_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        assert len(obs) == env.obs_dim
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_zero_action_minimal_change(self, env):
        obs0, _ = env.reset(seed=42)
        action = np.zeros(env.n_actions)
        obs1, _, _, _, _ = env.step(action)
        # CST weights should be identical (zero action)
        n_w = env.n_weights
        np.testing.assert_allclose(obs0[:2*n_w], obs1[:2*n_w], atol=1e-10)

    def test_episode_terminates(self, env):
        env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 100:
            action = np.random.randn(env.n_actions) * 0.1
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done  # Should terminate within max_steps

    def test_render_returns_coords(self, env):
        env.reset(seed=42)
        coords = env.render()
        assert coords.shape[1] == 2
        assert len(coords) > 10
