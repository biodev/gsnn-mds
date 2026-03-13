from math import sqrt
from statistics import NormalDist
from typing import Literal
import numpy as np

def power_two_sample_z(
    a_mu: float,
    a_std: float,
    b_mu: float,
    b_std: float,
    n_a: int,
    n_b: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided","larger","smaller"] = "two-sided",
) -> float:
    """
    Analytic power for two-sample z-test with known variances.

    H0: μa = μb
    H1 (two-sided): μa ≠ μb
       (larger):    μa > μb
       (smaller):   μa < μb

    Z = (X̄a - X̄b) / sqrt(σa^2/n_a + σb^2/n_b)
    Under true Δ = μa - μb, Z ~ Normal(κ, 1) with κ = Δ / SE.

    Returns:
        Power in [0,1].
    """
    nd = NormalDist()
    se = sqrt(a_std**2 / n_a + b_std**2 / n_b)
    kappa = (a_mu - b_mu) / se  # noncentrality (mean of Z under H1)

    if alternative == "two-sided":
        zcrit = nd.inv_cdf(1 - alpha / 2.0)
        # Power = P(Z > zcrit - κ) + P(Z < -zcrit - κ),  Z ~ N(0,1)
        return (1 - nd.cdf(zcrit - kappa)) + nd.cdf(-zcrit - kappa)
    elif alternative == "larger":
        zcrit = nd.inv_cdf(1 - alpha)
        return 1 - nd.cdf(zcrit - kappa)
    elif alternative == "smaller":
        zcrit = nd.inv_cdf(1 - alpha)
        return nd.cdf(-zcrit - kappa)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")