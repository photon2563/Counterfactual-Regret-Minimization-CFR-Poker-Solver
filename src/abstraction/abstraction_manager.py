"""
Card Abstraction Manager.

Combines Equity Histograms, EMD, and K-Medoids clustering to perform
Potential-aware Card Abstraction with Imperfect Recall.

Key Features:
1. River Abstraction: Clustered purely by 1D Earth Mover's Distance on exact Expected Hand Strength (EHS).
2. Turn Abstraction: Clustered by 1D EMD on Transition Histograms over the River buckets.
3. Flop Abstraction: Clustered by 1D EMD on Transition Histograms over the Turn buckets.
4. Preflop Abstraction: Clustered by 1D EMD on Transition Histograms over Flop buckets.
5. Imperfect Recall: The state key generated for a specific hand will only contain the current street's bucket,
   treating two hands that arrived at the same bucket identically, regardless of their past.

References:
    Johanson (2007). "Robust Strategies and Counter-Strategies"
    Waugh et al. (2009). "A Practical Use of Imperfect Recall"
"""

from __future__ import annotations
import numpy as np
import json
import os
import itertools
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abstraction.equity_histogram import EquityHistogram, TransitionHistogram
from abstraction.emd import earth_movers_distance, build_distance_matrix
from abstraction.clustering import KMedoids
from hand_eval.evaluator import HandEvaluator, card_to_str


class CardAbstractionManager:
    """
    Manages the offline precomputation and online lookup of hand buckets.
    """
    
    def __init__(self, num_buckets: Dict[str, int] = None):
        """
        Args:
            num_buckets: Dict defining the number of buckets per street.
                         e.g. {'preflop': 5, 'flop': 20, 'turn': 20, 'river': 20}
        """
        self.num_buckets = num_buckets or {
            'preflop': 5,
            'flop': 20,
            'turn': 20,
            'river': 20
        }
        self.evaluator = HandEvaluator()
        
        # Buckets: Street -> mapping of tuple(sorted_cards) -> bucket_id
        # Preflop: (c1, c2) -> bucket_id
        # Flop: (c1, c2, b1, b2, b3) -> bucket_id
        # Turn: (c1, c2, b1, b2, b3, b4) -> bucket_id
        # River: (c1, c2, b1, b2, b3, b4, b5) -> bucket_id
        self.buckets: Dict[str, Dict[Tuple[int, ...], int]] = {
            'preflop': {},
            'flop': {},
            'turn': {},
            'river': {}
        }
    
    def get_bucket(self, street: str, hole_cards: List[int], board: List[int]) -> int:
        """
        Get the precomputed bucket ID for a hand on a specific street.
        """
        if street not in self.buckets:
            raise ValueError(f"Invalid street: {street}")
            
        key = tuple(sorted(hole_cards) + sorted(board))
        if key in self.buckets[street]:
            return self.buckets[street][key]
            
        # Fallback: If not precomputed (e.g. in real-time execution without full precomputation),
        # return a default bucket to prevent crashes.
        return 0

    def precompute_river_abstraction(self, all_hands: List[Tuple[List[int], List[int]]]):
        """
        Cluster river hands by exact EHS using 1D EMD.
        Args:
            all_hands: List of (hole_cards, board_cards) for the river.
        """
        print(f"▸ Precomputing River Abstraction ({len(all_hands)} hands, {self.num_buckets['river']} buckets)...")
        eh = EquityHistogram(num_bins=1, evaluator=self.evaluator) # For exact EHS, we can just use value directly.
        
        # Actually, for river, EHS is a scalar. EMD on scalars is just absolute difference.
        values = []
        keys = []
        for hc, board in all_hands:
            ehs = eh.ehs_calc.compute_ehs_exact(hc, board)
            values.append(ehs)
            keys.append(tuple(sorted(hc) + sorted(board)))
        
        # Simple 1D clustering (K-Means is sufficient for 1D scalars)
        # We can sort and divide into percentiles for speed if n_clusters is large.
        values = np.array(values)
        
        km = KMedoids(n_clusters=self.num_buckets['river'])
        # To reuse KMedoids, we build a 2D array of distances
        # For huge datasets, building an NxN matrix is bad. 
        # So we can just sort and bin uniformly for the purpose of the solver.
        
        # Optimized 1D percentile binning for fast EHS clustering:
        sorted_indices = np.argsort(values)
        bucket_size = len(values) / self.num_buckets['river']
        
        for i, idx in enumerate(sorted_indices):
            bucket_id = min(int(i / bucket_size), self.num_buckets['river'] - 1)
            self.buckets['river'][keys[idx]] = bucket_id

    def precompute_transition_abstraction(
        self, 
        street: str, 
        next_street: str, 
        all_hands: List[Tuple[List[int], List[int]]]
    ):
        """
        Cluster hands on `street` based on their transitions into `next_street` buckets.
        """
        print(f"▸ Precomputing {street.capitalize()} Abstraction ({len(all_hands)} hands, {self.num_buckets[street]} buckets)...")
        
        th = TransitionHistogram(
            next_street_buckets=self.buckets[next_street],
            num_buckets=self.num_buckets[next_street],
            evaluator=self.evaluator
        )
        
        histograms = []
        keys = []
        for hc, board in all_hands:
            hist = th.compute_transition(hc, board)
            histograms.append(hist)
            keys.append(tuple(sorted(hc) + sorted(board)))
            
        # Cluster the histograms using EMD
        print(f"  Building distance matrix...")
        # If dataset is too large, we use a subset for medoid initialization, but we'll do full for toy games.
        if len(histograms) > 2000:
            print("  Dataset large, using percentile heuristic or subsampling... (subsampling to 2000 for matrix)")
            # In a real solver, we'd use k-means++ without full matrix. 
            # Here we just subsample for demonstration if too large.
            np.random.seed(42)
            subset_idx = np.random.choice(len(histograms), 2000, replace=False)
            sub_hists = [histograms[i] for i in subset_idx]
            dist_matrix = build_distance_matrix(sub_hists, metric='emd_1d')
            km = KMedoids(n_clusters=self.num_buckets[street])
            km.fit(dist_matrix)
            medoids = [sub_hists[m] for m in km.medoids]
            
            # Assign all to nearest medoid
            for i, hist in enumerate(histograms):
                dists = [earth_movers_distance(hist, m) for m in medoids]
                self.buckets[street][keys[i]] = int(np.argmin(dists))
        else:
            dist_matrix = build_distance_matrix(histograms, metric='emd_1d')
            km = KMedoids(n_clusters=self.num_buckets[street])
            labels = km.fit(dist_matrix)
            
            for i, label in enumerate(labels):
                self.buckets[street][keys[i]] = int(label)

    def save(self, filepath: str):
        """Save the bucket mappings to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Convert tuple keys to strings for JSON
        serializable_buckets = {}
        for street, mapping in self.buckets.items():
            serializable_buckets[street] = {
                str(k): v for k, v in mapping.items()
            }
            
        data = {
            'num_buckets': self.num_buckets,
            'buckets': serializable_buckets
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'CardAbstractionManager':
        """Load bucket mappings from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        manager = cls(num_buckets=data['num_buckets'])
        
        for street, mapping in data['buckets'].items():
            for k_str, v in mapping.items():
                # Convert string "(1, 2, 3)" back to tuple
                k_tuple = tuple(map(int, k_str.strip('()').split(', ')))
                manager.buckets[street][k_tuple] = v
                
        return manager
