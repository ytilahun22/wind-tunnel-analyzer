"""
analysis.py

Aerodynamic analysis utilities for the Wind Tunnel Analyzer project.

This module currently supports simple calculations based on a linear
lift-curve model, including a numerical solver (scipy.optimize.fsolve)
that finds the angle of attack required for a target lift coefficient.
"""

from scipy.optimize import fsolve


def compute_cl(alpha, cl0=0.10, lift_slope=0.11):
    """
    Compute lift coefficient using a linear lift model.

    Parameters
    ----------
    alpha : float
        Angle of attack in degrees.
    cl0 : float, optional
        Lift coefficient at zero angle of attack.
    lift_slope : float, optional
        Lift curve slope in units of 1/degree.

    Returns
    -------
    float
        Lift coefficient.
    """
    return cl0 + lift_slope * alpha


def print_lift_summary(alpha, cl0=0.10, lift_slope=0.11):
    """
    Print a simple summary of the lift coefficient for a given angle of attack.

    Parameters
    ----------
    alpha : float
        Angle of attack in degrees.
    cl0 : float, optional
        Lift coefficient at zero angle of attack.
    lift_slope : float, optional
        Lift curve slope in units of 1/degree.
    """
    cl = compute_cl(alpha, cl0=cl0, lift_slope=lift_slope)

    print("Wind Tunnel Analyzer")
    print("--------------------")
    print(f"Angle of attack : {alpha:.2f} deg")
    print(f"Lift coefficient: {cl:.3f}")


def alpha_residual(alpha, target_cl, cl0=0.10, lift_slope=0.11):
    """
    Residual of the linear lift model for a target lift coefficient.

        f(alpha) = cl0 + lift_slope * alpha - target_cl

    The angle of attack we want is the root of this function, f(alpha) = 0.

    Parameters
    ----------
    alpha : float or array_like
        Angle of attack in degrees (fsolve passes this in as an array).
    target_cl : float
        Desired lift coefficient.
    cl0 : float, optional
        Lift coefficient at zero angle of attack.
    lift_slope : float, optional
        Lift curve slope in units of 1/degree.

    Returns
    -------
    float or ndarray
        Difference between the model lift coefficient and the target.
    """
    return compute_cl(alpha, cl0=cl0, lift_slope=lift_slope) - target_cl


def solve_alpha(target_cl, cl0=0.10, lift_slope=0.11, alpha_guess=0.0):
    """
    Compute the angle of attack required to reach a target lift coefficient.

    Uses scipy.optimize.fsolve to find the root of alpha_residual().

    Parameters
    ----------
    target_cl : float
        Desired lift coefficient.
    cl0 : float, optional
        Lift coefficient at zero angle of attack.
    lift_slope : float, optional
        Lift curve slope in units of 1/degree.
    alpha_guess : float, optional
        Initial guess for the angle of attack in degrees.

    Returns
    -------
    float
        Angle of attack in degrees that satisfies f(alpha) = 0.

    Raises
    ------
    ValueError
        If lift_slope is zero or the solver does not converge.
    """
    if lift_slope == 0:
        raise ValueError("lift_slope must be nonzero to solve for alpha.")

    solution, info, ier, msg = fsolve(
        alpha_residual,
        x0=alpha_guess,
        args=(target_cl, cl0, lift_slope),
        full_output=True,
    )

    if ier != 1:
        raise ValueError(f"fsolve did not converge: {msg}")

    return float(solution[0])
