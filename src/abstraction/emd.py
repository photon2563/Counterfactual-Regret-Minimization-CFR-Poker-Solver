"""
Earth Mover's Distance (EMD) for comparing equity histograms.

EMD, also known as the Wasserstein distance, measures the minimum
"work" needed to transform one probability distribution into another.
It's the ideal metric for comparing equity histograms because it
respects the ordinal nature of hand strength.

For 1D histograms, EMD has a closed-form solution:
    EMD(p, q) = Σ_i |CDF_p(i) - CDF_q(i)|

This is O(n) where n = number of bins.

References:
    Johanson (2007). "Robust Strategies and Counter-Strategies:
    Building a Champion Level Computer Poker Player."
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional


def earth_movers_distance(
    hist1: np.ndarray,
    hist2: np.ndarray,
) -> float:
    """
    Compute 1D Earth Mover's Distance between two histograms.
    
    Uses the closed-form CDF difference formula:
        EMD(p, q) = Σ_i |CDF_p(i) - CDF_q(i)|
    
    Args:
        hist1: First probability distribution (must sum to 1)
        hist2: Second probability distribution (must sum to 1)
    
    Returns:
        EMD distance (non-negative float)
    """
    assert len(hist1) == len(hist2), "Histograms must have same length"
    
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    
    return float(np.sum(np.abs(cdf1 - cdf2)))


def earth_movers_distance_2d(
    hist1: np.ndarray,
    hist2: np.ndarray,
    ground_distance: Optional[np.ndarray] = None,
) -> float:
    """
    Compute EMD between 2D histograms using the linear programming formulation.
    
    For the potential-aware abstraction, we need to compare histograms over
    next-street buckets, where the ground distance between buckets is the
    distance between bucket centroids.
    
    For small numbers of buckets, this uses a simplified approach.
    
    Args:
        hist1: First distribution over buckets
        hist2: Second distribution over buckets
        ground_distance: Matrix of distances between buckets.
                        If None, uses absolute index difference.
    
    Returns:
        EMD distance
    """
    n = len(hist1)
    
    if ground_distance is None:
        # Default ground distance: absolute index difference
        # This reduces to the 1D EMD formula
        return earth_movers_distance(hist1, hist2)
    
    # For general ground distances, use the transportation simplex
    # For small n, we can use a greedy approach
    from scipy.optimize import linear_sum_assignment
    
    # Build cost matrix for transportation problem
    # We discretize the histograms into small units
    resolution = 1000
    supply = (hist1 * resolution).astype(int)
    demand = (hist2 * resolution).astype(int)
    
    # Adjust for rounding errors
    diff = supply.sum() - demand.sum()
    if diff > 0:
        supply[np.argmax(supply)] -= diff
    elif diff < 0:
        demand[np.argmax(demand)] += diff
    
    # Simple LP relaxation for EMD
    total_cost = 0.0
    remaining_supply = supply.astype(float)
    remaining_demand = demand.astype(float)
    
    # Greedy: always satisfy cheapest remaining transport
    while remaining_supply.sum() > 1e-6 and remaining_demand.sum() > 1e-6:
        # Find cheapest unused route
        best_cost = float('inf')
        best_i, best_j = 0, 0
        for i in range(n):
            if remaining_supply[i] < 1e-6:
                continue
            for j in range(n):
                if remaining_demand[j] < 1e-6:
                    continue
                if ground_distance[i, j] < best_cost:
                    best_cost = ground_distance[i, j]
                    best_i, best_j = i, j
        
        amount = min(remaining_supply[best_i], remaining_demand[best_j])
        total_cost += amount * best_cost
        remaining_supply[best_i] -= amount
        remaining_demand[best_j] -= amount
    
    return total_cost / resolution


def build_distance_matrix(
    histograms: List[np.ndarray],
) -> np.ndarray:
    """
    Build a pairwise EMD distance matrix for a set of histograms.
    
    Args:
        histograms: List of N histograms
    
    Returns:
        N × N symmetric distance matrix
    """
    n = len(histograms)
    distances = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = earth_movers_distance(histograms[i], histograms[j])
            distances[i, j] = d
            distances[j, i] = d
    
    return distances
