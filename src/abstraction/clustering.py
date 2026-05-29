"""
K-Medoids Clustering for Card Abstraction.

K-Medoids (PAM algorithm) clusters hands into strategically similar groups
using Earth Mover's Distance (EMD) as the distance metric.

Unlike K-Means, K-Medoids:
- Uses actual data points as cluster centers (medoids)
- Works with arbitrary distance metrics (not just Euclidean)
- Is more robust to outliers

This is critical for card abstraction because:
1. We need to work with EMD distances (not Euclidean)
2. The medoid of a cluster IS a real hand, making interpretation easier
3. The clustering must be deterministic for reproducible solver runs

References:
    Johanson (2007). "Robust Strategies and Counter-Strategies"
    Waugh et al. (2009). "A Practical Use of Imperfect Recall"
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import defaultdict


class KMedoids:
    """
    K-Medoids clustering using the PAM (Partition Around Medoids) algorithm.
    
    Usage:
        distances = build_distance_matrix(histograms)
        km = KMedoids(n_clusters=200, max_iter=100)
        labels = km.fit(distances)
    """
    
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Args:
            n_clusters: Number of clusters (buckets)
            max_iter: Maximum iterations for PAM
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)
        
        self.medoid_indices: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.inertia: float = float('inf')
    
    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Fit K-Medoids clustering to a precomputed distance matrix.
        
        Args:
            distance_matrix: N × N symmetric distance matrix
        
        Returns:
            Cluster labels for each data point (0 to n_clusters-1)
        """
        n = distance_matrix.shape[0]
        assert n >= self.n_clusters, \
            f"Need at least {self.n_clusters} points, got {n}"
        
        # Initialize: pick k random medoids
        medoids = self.rng.choice(n, size=self.n_clusters, replace=False)
        
        # Alternative: use BUILD initialization (greedy, like k-means++)
        medoids = self._build_init(distance_matrix)
        
        for iteration in range(self.max_iter):
            # ASSIGN: each point to nearest medoid
            labels = self._assign_labels(distance_matrix, medoids)
            
            # SWAP: try replacing each medoid with each non-medoid
            improved = False
            for m_idx in range(self.n_clusters):
                current_medoid = medoids[m_idx]
                cluster_members = np.where(labels == m_idx)[0]
                
                if len(cluster_members) == 0:
                    continue
                
                # Find best replacement in this cluster
                best_cost = np.sum(distance_matrix[current_medoid, cluster_members])
                best_candidate = current_medoid
                
                for candidate in cluster_members:
                    if candidate == current_medoid:
                        continue
                    cost = np.sum(distance_matrix[candidate, cluster_members])
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = candidate
                
                if best_candidate != current_medoid:
                    medoids[m_idx] = best_candidate
                    improved = True
            
            if not improved:
                break
        
        # Final assignment
        labels = self._assign_labels(distance_matrix, medoids)
        
        self.medoid_indices = medoids
        self.labels = labels
        self.inertia = self._compute_inertia(distance_matrix, medoids, labels)
        
        return labels
    
    def _build_init(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        BUILD initialization: greedy selection of initial medoids.
        
        Similar to k-means++ but uses precomputed distances.
        1. Pick the point that minimizes total distance to all others
        2. Iteratively pick the point that maximally reduces total cost
        """
        n = distance_matrix.shape[0]
        
        # First medoid: minimize total distance to all points
        total_distances = distance_matrix.sum(axis=1)
        medoids = [np.argmin(total_distances)]
        
        # Remaining medoids: greedy selection
        min_distances = distance_matrix[medoids[0]].copy()
        
        for _ in range(1, self.n_clusters):
            # For each candidate, compute gain from adding it
            gains = np.zeros(n)
            for candidate in range(n):
                if candidate in medoids:
                    gains[candidate] = -float('inf')
                    continue
                # Gain = reduction in total min-distance
                new_distances = np.minimum(min_distances, distance_matrix[candidate])
                gains[candidate] = np.sum(min_distances - new_distances)
            
            new_medoid = np.argmax(gains)
            medoids.append(new_medoid)
            min_distances = np.minimum(min_distances, distance_matrix[new_medoid])
        
        return np.array(medoids)
    
    def _assign_labels(
        self,
        distance_matrix: np.ndarray,
        medoids: np.ndarray,
    ) -> np.ndarray:
        """Assign each point to its nearest medoid."""
        distances_to_medoids = distance_matrix[:, medoids]
        return np.argmin(distances_to_medoids, axis=1)
    
    def _compute_inertia(
        self,
        distance_matrix: np.ndarray,
        medoids: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute total within-cluster distance."""
        total = 0.0
        for m_idx in range(len(medoids)):
            members = np.where(labels == m_idx)[0]
            total += np.sum(distance_matrix[medoids[m_idx], members])
        return total
    
    def silhouette_score(self, distance_matrix: np.ndarray) -> float:
        """
        Compute average silhouette score for the clustering.
        
        Silhouette ∈ [-1, 1]:
        - +1: points are well-clustered
        -  0: points are on cluster boundaries
        - -1: points are mis-clustered
        """
        assert self.labels is not None, "Must call fit() first"
        
        n = len(self.labels)
        scores = np.zeros(n)
        
        for i in range(n):
            # a(i) = average distance to other points in same cluster
            same_cluster = np.where(self.labels == self.labels[i])[0]
            same_cluster = same_cluster[same_cluster != i]
            
            if len(same_cluster) == 0:
                scores[i] = 0
                continue
            
            a_i = np.mean(distance_matrix[i, same_cluster])
            
            # b(i) = minimum average distance to points in any other cluster
            b_i = float('inf')
            for k in range(self.n_clusters):
                if k == self.labels[i]:
                    continue
                other_cluster = np.where(self.labels == k)[0]
                if len(other_cluster) == 0:
                    continue
                avg_dist = np.mean(distance_matrix[i, other_cluster])
                b_i = min(b_i, avg_dist)
            
            if b_i == float('inf'):
                scores[i] = 0
            else:
                scores[i] = (b_i - a_i) / max(a_i, b_i)
        
        return float(np.mean(scores))


class CardAbstraction:
    """
    Full card abstraction pipeline for Texas Hold'em.
    
    For each street (preflop, flop, turn, river):
    1. Compute equity histograms or EHS for all possible hands
    2. Build pairwise EMD distance matrix
    3. Cluster using K-Medoids into configurable bucket counts
    
    Potential-aware abstraction computes transition histograms
    showing how each hand maps to future-street buckets, then
    clusters based on similarity of these transition distributions.
    """
    
    def __init__(
        self,
        river_buckets: int = 50,
        turn_buckets: int = 50,
        flop_buckets: int = 50,
        preflop_buckets: int = 10,
    ):
        self.bucket_counts = {
            'river': river_buckets,
            'turn': turn_buckets,
            'flop': flop_buckets,
            'preflop': preflop_buckets,
        }
        self.clusterings: Dict[str, KMedoids] = {}
    
    def abstract_river(
        self,
        board: List[int],
        ehs_values: Dict[Tuple[int, int], float],
    ) -> Dict[Tuple[int, int], int]:
        """
        Cluster river hands by EHS value.
        
        On the river, EHS is a single scalar, so we can use simple
        histogram binning or 1D K-Medoids.
        
        Args:
            board: 5 community cards
            ehs_values: Maps (card1, card2) → EHS
        
        Returns:
            Maps (card1, card2) → bucket_id
        """
        hands = sorted(ehs_values.keys())
        values = np.array([ehs_values[h] for h in hands])
        
        # Simple approach: uniform binning by EHS
        n_buckets = self.bucket_counts['river']
        bucket_edges = np.linspace(0, 1, n_buckets + 1)
        
        assignments = {}
        for hand, ehs in zip(hands, values):
            bucket = min(int(ehs * n_buckets), n_buckets - 1)
            assignments[hand] = bucket
        
        return assignments
