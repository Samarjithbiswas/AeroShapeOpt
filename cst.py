"""
Class-Shape Transformation (CST) airfoil parameterization.

Implements the Kulfan CST method (Kulfan, 2008) for representing airfoil
geometries as a product of a class function and a shape function.

The CST representation:
    z(psi) = C(psi) * S(psi) + psi * dz_TE

where:
    psi = x/c  in [0, 1]  (normalized chord coordinate)
    C(psi) = psi^N1 * (1 - psi)^N2  (class function)
    S(psi) = sum_{i=0}^{n} A_i * S_i(psi)  (shape function via Bernstein polynomials)
    S_i(psi) = K_i * psi^i * (1 - psi)^{n-i}  (Bernstein basis)
    K_i = n! / (i! * (n-i)!)  (binomial coefficient)

For round-nosed airfoils: N1 = 0.5, N2 = 1.0
The coefficients A_i (shape weights) control the airfoil shape.

Reference: Kulfan, B.M. "Universal Parametric Geometry Representation
Method", J. Aircraft, 45(1), 142-158, 2008.

Author: Samarjith Biswas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import comb


@dataclass
class CSTAirfoil:
    """CST-parameterized airfoil.

    Attributes:
        weights_upper: Bernstein polynomial coefficients for upper surface
        weights_lower: Bernstein polynomial coefficients for lower surface
        dz_te: Trailing edge thickness (half, per surface)
        N1, N2: Class function exponents (0.5, 1.0 for round-nose airfoils)
    """

    weights_upper: np.ndarray
    weights_lower: np.ndarray
    dz_te: float = 0.0
    N1: float = 0.5
    N2: float = 1.0

    @property
    def n_upper(self) -> int:
        """Order of upper surface Bernstein polynomial."""
        return len(self.weights_upper) - 1

    @property
    def n_lower(self) -> int:
        """Order of lower surface Bernstein polynomial."""
        return len(self.weights_lower) - 1

    @property
    def n_params(self) -> int:
        """Total number of shape parameters."""
        return len(self.weights_upper) + len(self.weights_lower)


def class_function(psi: np.ndarray, N1: float = 0.5, N2: float = 1.0) -> np.ndarray:
    """CST class function: C(psi) = psi^N1 * (1 - psi)^N2.

    For round-nosed, sharp-trailing-edge airfoils: N1=0.5, N2=1.0
    This gives the characteristic sqrt(x) leading edge and linear
    trailing edge taper.

    Args:
        psi: Normalized chord coordinate in [0, 1]
        N1: Leading edge exponent (0.5 for round nose)
        N2: Trailing edge exponent (1.0 for sharp TE)

    Returns:
        Class function values
    """
    # Handle boundary values to avoid numerical issues
    psi = np.clip(psi, 1e-15, 1.0 - 1e-15)
    return psi**N1 * (1.0 - psi)**N2


def bernstein_basis(psi: np.ndarray, n: int) -> np.ndarray:
    """Compute all Bernstein basis polynomials of order n.

    B_{i,n}(psi) = C(n,i) * psi^i * (1-psi)^{n-i}

    where C(n,i) is the binomial coefficient.

    Args:
        psi: Evaluation points in [0, 1], shape (M,)
        n: Polynomial order

    Returns:
        Basis matrix of shape (M, n+1), where column i is B_{i,n}(psi)
    """
    M = len(psi)
    basis = np.zeros((M, n + 1))
    for i in range(n + 1):
        basis[:, i] = comb(n, i, exact=True) * psi**i * (1.0 - psi)**(n - i)
    return basis


def shape_function(psi: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """CST shape function: S(psi) = sum_i A_i * B_{i,n}(psi).

    Args:
        psi: Normalized chord coordinate, shape (M,)
        weights: Bernstein coefficients A_i, shape (n+1,)

    Returns:
        Shape function values, shape (M,)
    """
    n = len(weights) - 1
    basis = bernstein_basis(psi, n)
    return basis @ weights


def cst_surface(
    psi: np.ndarray,
    weights: np.ndarray,
    dz_te: float = 0.0,
    N1: float = 0.5,
    N2: float = 1.0,
) -> np.ndarray:
    """Compute CST surface coordinates.

    z(psi) = C(psi) * S(psi) + psi * dz_te

    Args:
        psi: Normalized x-coordinates in [0, 1]
        weights: Bernstein polynomial coefficients
        dz_te: Trailing edge offset (half-thickness at TE)
        N1, N2: Class function exponents

    Returns:
        z-coordinates (vertical), same shape as psi
    """
    C = class_function(psi, N1, N2)
    S = shape_function(psi, weights)
    return C * S + psi * dz_te


def generate_airfoil_coordinates(
    airfoil: CSTAirfoil,
    n_points: int = 100,
) -> np.ndarray:
    """Generate airfoil coordinates from CST parameters.

    Uses cosine spacing for better leading edge resolution.

    Args:
        airfoil: CSTAirfoil instance
        n_points: Points per surface

    Returns:
        Coordinates (N, 2), ordered TE-upper → LE → TE-lower
    """
    # Cosine spacing (Glauert transformation)
    beta = np.linspace(0, np.pi, n_points)
    psi = 0.5 * (1.0 - np.cos(beta))

    z_upper = cst_surface(psi, airfoil.weights_upper, airfoil.dz_te,
                           airfoil.N1, airfoil.N2)
    z_lower = cst_surface(psi, airfoil.weights_lower, -airfoil.dz_te,
                           airfoil.N1, airfoil.N2)

    # Upper: TE → LE (reversed), Lower: LE → TE
    x_upper = psi[::-1]
    y_upper = z_upper[::-1]
    x_lower = psi[1:]  # Skip LE (duplicate)
    y_lower = z_lower[1:]

    coords = np.vstack([
        np.column_stack([x_upper, y_upper]),
        np.column_stack([x_lower, y_lower]),
    ])

    return coords.astype(np.float64)


def naca0012_cst_weights(order: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Approximate NACA 0012 as CST weights (for initialization).

    Fits CST Bernstein coefficients to the NACA 0012 thickness distribution
    using least squares.

    Args:
        order: Bernstein polynomial order

    Returns:
        (weights_upper, weights_lower) arrays
    """
    # NACA 0012 thickness distribution at sample points
    psi = np.linspace(0.001, 0.999, 200)
    t = 0.12  # Max thickness
    # Standard NACA formula
    y_t = 5 * t * (0.2969 * np.sqrt(psi) - 0.1260 * psi
                    - 0.3516 * psi**2 + 0.2843 * psi**3 - 0.1036 * psi**4)

    # Fit CST: z = C(psi) * S(psi), so S(psi) = z / C(psi)
    C = class_function(psi, 0.5, 1.0)
    S_target = y_t / (C + 1e-15)

    # Least squares fit: S = B @ weights
    B = bernstein_basis(psi, order)
    weights, _, _, _ = np.linalg.lstsq(B, S_target, rcond=None)

    return weights, -weights  # Symmetric airfoil


def compute_thickness(airfoil: CSTAirfoil, n_points: int = 100) -> tuple[float, float]:
    """Compute maximum thickness and its location.

    Returns:
        (max_thickness, location_x) where location_x is in [0, 1]
    """
    psi = np.linspace(0.001, 0.999, n_points)
    z_upper = cst_surface(psi, airfoil.weights_upper, airfoil.dz_te,
                           airfoil.N1, airfoil.N2)
    z_lower = cst_surface(psi, airfoil.weights_lower, -airfoil.dz_te,
                           airfoil.N1, airfoil.N2)
    thickness = z_upper - z_lower
    idx = np.argmax(thickness)
    return float(thickness[idx]), float(psi[idx])


def compute_leading_edge_radius(airfoil: CSTAirfoil) -> float:
    """Approximate leading edge radius from CST weights.

    For CST with N1=0.5: R_LE ≈ 0.5 * A_0^2 (Kulfan, 2008).

    Returns:
        Leading edge radius (approximate)
    """
    A0_upper = airfoil.weights_upper[0]
    A0_lower = airfoil.weights_lower[0]
    # Average of upper and lower contributions
    return 0.5 * (A0_upper**2 + A0_lower**2) / 2
