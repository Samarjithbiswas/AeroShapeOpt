"""
Hess-Smith panel method for aerodynamic coefficient evaluation.

Lightweight implementation optimized for rapid RL environment evaluation.
Computes Cl (lift coefficient) and Cd_p (pressure drag coefficient) from
the surface pressure distribution.

For inviscid incompressible flow:
    Cp_i = 1 - (V_i / V_inf)^2
    Cl = -sum_i Cp_i * sin(theta_i) * L_i / c
    Cd_p = -sum_i Cp_i * cos(theta_i) * L_i / c

where theta_i is the panel angle, L_i is the panel length, and c is the chord.

Note: Inviscid flow has zero friction drag. We add an empirical skin friction
estimate Cf ~ 0.0027 * Re^(-1/7) to approximate total drag (Schlichting, 1979).

Author: Samarjith Biswas
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AeroCoefficients:
    """Aerodynamic force coefficients."""
    Cl: float          # Lift coefficient
    Cd: float          # Total drag coefficient (pressure + friction)
    Cd_pressure: float # Pressure drag only
    Cd_friction: float # Skin friction estimate
    Cm: float          # Pitching moment coefficient (about c/4)
    L_over_D: float    # Lift-to-drag ratio
    Cp_surface: np.ndarray  # Surface pressure distribution
    converged: bool    # Whether the solution is physically valid


def evaluate_airfoil(
    coords: np.ndarray,
    alpha_deg: float = 0.0,
    Re: float = 1e6,
) -> AeroCoefficients:
    """Evaluate aerodynamic coefficients for an airfoil.

    Uses Hess-Smith panel method for inviscid Cp, then adds empirical
    skin friction estimate.

    Args:
        coords: Airfoil coordinates (N, 2), ordered TE-upper → LE → TE-lower
        alpha_deg: Angle of attack [degrees]
        Re: Reynolds number (for friction drag estimate)

    Returns:
        AeroCoefficients with Cl, Cd, L/D, etc.
    """
    alpha = np.radians(alpha_deg)
    n_panels = len(coords) - 1

    if n_panels < 10:
        return AeroCoefficients(
            Cl=0.0, Cd=1.0, Cd_pressure=1.0, Cd_friction=0.0,
            Cm=0.0, L_over_D=0.0, Cp_surface=np.zeros(1), converged=False,
        )

    # Panel geometry
    xm = np.zeros(n_panels)
    ym = np.zeros(n_panels)
    dx = np.zeros(n_panels)
    dy = np.zeros(n_panels)
    length = np.zeros(n_panels)
    theta = np.zeros(n_panels)
    nx = np.zeros(n_panels)
    ny = np.zeros(n_panels)

    for i in range(n_panels):
        dx[i] = coords[i + 1, 0] - coords[i, 0]
        dy[i] = coords[i + 1, 1] - coords[i, 1]
        length[i] = np.sqrt(dx[i]**2 + dy[i]**2) + 1e-15
        theta[i] = np.arctan2(dy[i], dx[i])
        xm[i] = 0.5 * (coords[i, 0] + coords[i + 1, 0])
        ym[i] = 0.5 * (coords[i, 1] + coords[i + 1, 1])
        nx[i] = dy[i] / length[i]
        ny[i] = -dx[i] / length[i]

    # Freestream components
    u_inf = np.cos(alpha)
    v_inf = np.sin(alpha)

    # Build influence matrix (source + vortex)
    N = n_panels
    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)

    for i in range(N):
        for j in range(N):
            us_n, us_t, uv_n, uv_t = _panel_influence(
                xm[i], ym[i], coords[j, 0], coords[j, 1],
                theta[j], length[j], theta[i],
            )
            A[i, j] = us_n  # Source normal influence
            A[i, N] += uv_n  # Vortex normal influence
        rhs[i] = -(u_inf * nx[i] + v_inf * ny[i])

    # Kutta condition: V_t(first panel) + V_t(last panel) = 0
    for j in range(N):
        _, us_t0, _, uv_t0 = _panel_influence(
            xm[0], ym[0], coords[j, 0], coords[j, 1],
            theta[j], length[j], theta[0],
        )
        _, us_tN, _, uv_tN = _panel_influence(
            xm[N-1], ym[N-1], coords[j, 0], coords[j, 1],
            theta[j], length[j], theta[N-1],
        )
        A[N, j] = us_t0 + us_tN
        A[N, N] += uv_t0 + uv_tN

    t0x, t0y = np.cos(theta[0]), np.sin(theta[0])
    tNx, tNy = np.cos(theta[N-1]), np.sin(theta[N-1])
    rhs[N] = -(u_inf * (t0x + tNx) + v_inf * (t0y + tNy))

    # Solve
    try:
        solution = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return AeroCoefficients(
            Cl=0.0, Cd=1.0, Cd_pressure=1.0, Cd_friction=0.0,
            Cm=0.0, L_over_D=0.0, Cp_surface=np.zeros(N), converged=False,
        )

    sigma = solution[:-1]
    gamma = solution[-1]

    # Compute tangential velocity at each panel → Cp
    Cp = np.zeros(N)
    for i in range(N):
        vt = u_inf * np.cos(theta[i]) + v_inf * np.sin(theta[i])
        for j in range(N):
            _, us_t, _, uv_t = _panel_influence(
                xm[i], ym[i], coords[j, 0], coords[j, 1],
                theta[j], length[j], theta[i],
            )
            vt += sigma[j] * us_t + gamma * uv_t
        Cp[i] = 1.0 - vt**2

    # Force integration (panel method)
    # Cl = -sum(Cp * sin(theta) * L) / chord
    # Cd_p = -sum(Cp * cos(theta) * L) / chord (in freestream direction)
    chord = coords[:, 0].max() - coords[:, 0].min()
    if chord < 1e-6:
        chord = 1.0

    # Forces in body frame → rotate to wind frame
    Cn = -np.sum(Cp * nx * length) / chord  # Normal force coeff
    Ca = -np.sum(Cp * (-ny) * length) / chord  # Axial force coeff (tangent)

    # Rotate to wind frame
    Cl = Cn * np.cos(alpha) - Ca * np.sin(alpha)
    Cd_pressure = Cn * np.sin(alpha) + Ca * np.cos(alpha)

    # Empirical skin friction drag (flat plate turbulent, Schlichting 1979)
    # Cf = 0.455 / (log10(Re))^2.58 (for both surfaces)
    if Re > 100:
        Cf = 0.455 / (np.log10(Re + 1))**2.58
    else:
        Cf = 0.01
    Cd_friction = 2 * Cf  # Both surfaces

    Cd_total = abs(Cd_pressure) + Cd_friction

    # Pitching moment about c/4
    Cm = -np.sum(Cp * nx * length * (xm - 0.25)) / chord

    L_over_D = Cl / (Cd_total + 1e-10) if Cd_total > 1e-8 else 0.0

    # Sanity check
    converged = abs(Cl) < 10.0 and abs(Cd_total) < 2.0

    return AeroCoefficients(
        Cl=float(Cl), Cd=float(Cd_total),
        Cd_pressure=float(abs(Cd_pressure)), Cd_friction=float(Cd_friction),
        Cm=float(Cm), L_over_D=float(L_over_D),
        Cp_surface=Cp, converged=converged,
    )


def _panel_influence(
    xp: float, yp: float,
    x1: float, y1: float,
    theta_j: float, L_j: float,
    theta_i: float,
) -> tuple[float, float, float, float]:
    """Compute normal and tangential influence of panel j on point i.

    Returns: (source_normal, source_tangent, vortex_normal, vortex_tangent)
    """
    cos_j = np.cos(theta_j)
    sin_j = np.sin(theta_j)

    # Transform to panel-local coords
    xt = (xp - x1) * cos_j + (yp - y1) * sin_j
    yt = -(xp - x1) * sin_j + (yp - y1) * cos_j

    r1_sq = xt**2 + yt**2 + 1e-20
    r2_sq = (xt - L_j)**2 + yt**2 + 1e-20

    # Source panel velocity (local frame)
    u_s_local = 0.5 / (2 * np.pi) * np.log(r2_sq / r1_sq)
    v_s_local = 1.0 / (2 * np.pi) * (np.arctan2(yt, xt - L_j) - np.arctan2(yt, xt))

    # Vortex panel velocity (local frame)
    u_v_local = v_s_local  # Rotated 90°
    v_v_local = -u_s_local

    # Rotate to global frame
    u_s = u_s_local * cos_j - v_s_local * sin_j
    v_s = u_s_local * sin_j + v_s_local * cos_j
    u_v = u_v_local * cos_j - v_v_local * sin_j
    v_v = u_v_local * sin_j + v_v_local * cos_j

    # Project onto panel i normal and tangent
    ni_x = np.sin(theta_i)  # Approximation for normal
    ni_y = -np.cos(theta_i)
    # Actually use the correct normal/tangent from caller
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)

    source_normal = u_s * sin_i - v_s * cos_i  # Simplified
    source_tangent = u_s * cos_i + v_s * sin_i
    vortex_normal = u_v * sin_i - v_v * cos_i
    vortex_tangent = u_v * cos_i + v_v * sin_i

    return source_normal, source_tangent, vortex_normal, vortex_tangent
