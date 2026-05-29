"""
Equity Histogram Generation for Card Abstraction.

For each hand at a given street (flop, turn), we compute a histogram
showing the distribution of equity outcomes across all possible
future board runouts. This captures not just current hand strength,
but the *potential* for the hand to improve or deteriorate.

Key concept: Two hands with the same current EHS but very different
equity distributions (e.g., a made hand vs. a draw) should be in
different buckets. Equity histograms capture this nuance.

Process:
1. For a given hand + partial board, enumerate all possible future boards
2. For each future board, compute hand equity (EHS)
3. Bin these equity values into a histogram
4. Use EMD distance between histograms for clustering

References:
    Johanson (2007). "Robust Strategies and Counter-Strategies"
    Waugh et al. (2009). "A Practical Use of Imperfect Recall"
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import combinations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hand_eval.evaluator import HandEvaluator, EHSCalculator, card_rank, card_suit


class EquityHistogram:
    """
    Generates equity distribution histograms for poker hands.
    
    A histogram represents the probability distribution of a hand's
    equity across all possible future board cards.
    """
    
    def __init__(
        self,
        num_bins: int = 50,
        evaluator: Optional[HandEvaluator] = None,
    ):
        """
        Args:
            num_bins: Number of bins in the equity histogram
            evaluator: Hand evaluator instance (created if None)
        """
        self.num_bins = num_bins
        self.evaluator = evaluator or HandEvaluator()
        self.ehs_calc = EHSCalculator(self.evaluator)
    
    def compute_river_histogram(
        self,
        hole_cards: List[int],
        board: List[int],
        num_opponents: int = 1,
    ) -> np.ndarray:
        """
        Compute equity histogram on the river (5 board cards).
        
        On the river, equity is deterministic against a random opponent,
        so the histogram has a single spike at the exact equity value.
        However, against multiple opponents, there's a distribution.
        
        For single opponent, we compute exact equity.
        """
        assert len(board) == 5, f"River needs 5 board cards, got {len(board)}"
        
        ehs = self.ehs_calc.compute_ehs_exact(hole_cards, board)
        
        hist = np.zeros(self.num_bins)
        bin_idx = min(int(ehs * self.num_bins), self.num_bins - 1)
        hist[bin_idx] = 1.0
        
        return hist
    
    def compute_turn_histogram(
        self,
        hole_cards: List[int],
        board: List[int],
        num_rollouts: int = 200,
    ) -> np.ndarray:
        """
        Compute equity histogram on the turn (4 board cards).
        
        For each possible river card, compute equity, then build histogram.
        """
        assert len(board) == 4, f"Turn needs 4 board cards, got {len(board)}"
        
        used = set(hole_cards + board)
        remaining = [c for c in range(52) if c not in used]
        
        equities = []
        for river_card in remaining:
            full_board = board + [river_card]
            ehs = self.ehs_calc.compute_ehs_exact(hole_cards, full_board)
            equities.append(ehs)
        
        return self._build_histogram(equities)
    
    def compute_flop_histogram(
        self,
        hole_cards: List[int],
        board: List[int],
        num_rollouts: int = 500,
    ) -> np.ndarray:
        """
        Compute equity histogram on the flop (3 board cards).
        
        For each possible turn+river runout, compute equity.
        Uses Monte Carlo sampling since full enumeration is expensive
        (C(45,2) = 990 possible runouts × C(43,2) opponent hands).
        """
        assert len(board) == 3, f"Flop needs 3 board cards, got {len(board)}"
        
        used = set(hole_cards + board)
        remaining = [c for c in range(52) if c not in used]
        
        rng = np.random.RandomState(42)
        equities = []
        
        for _ in range(num_rollouts):
            # Sample 2 cards for turn + river
            sampled = rng.choice(remaining, size=2, replace=False)
            full_board = board + list(sampled)
            
            # Compute equity on this complete board
            ehs = self.ehs_calc.compute_ehs(hole_cards, full_board, num_rollouts=50)
            equities.append(ehs)
        
        return self._build_histogram(equities)
    
    def compute_preflop_histogram(
        self,
        hole_cards: List[int],
        num_rollouts: int = 1000,
    ) -> np.ndarray:
        """
        Compute equity histogram preflop.
        
        Sample random 5-card boards and compute equity for each.
        """
        rng = np.random.RandomState(42)
        remaining = [c for c in range(52) if c not in hole_cards]
        
        equities = []
        for _ in range(num_rollouts):
            board = list(rng.choice(remaining, size=5, replace=False))
            ehs = self.ehs_calc.compute_ehs(hole_cards, board, num_rollouts=50)
            equities.append(ehs)
        
        return self._build_histogram(equities)
    
    def _build_histogram(self, equities: List[float]) -> np.ndarray:
        """Convert a list of equity samples into a normalized histogram."""
        hist = np.zeros(self.num_bins)
        
        for eq in equities:
            bin_idx = min(int(eq * self.num_bins), self.num_bins - 1)
            hist[bin_idx] += 1
        
        # Normalize to probability distribution
        total = hist.sum()
        if total > 0:
            hist /= total
        else:
            hist[:] = 1.0 / self.num_bins  # Uniform if no data
        
        return hist


class TransitionHistogram:
    """
    Computes transition histograms for potential-aware abstraction.
    
    For each hand at street S, computes the probability distribution
    over buckets at street S+1. This captures how a hand's strategic
    value evolves as more cards are dealt.
    
    Process:
    1. Cluster hands at street S+1 (e.g., river) into K buckets
    2. For each hand at street S (e.g., turn), simulate all S+1 cards
    3. Record which S+1 bucket each outcome lands in
    4. Build histogram over the K buckets
    5. Cluster street S hands by similarity of transition histograms
    """
    
    def __init__(
        self,
        next_street_buckets: Dict[Tuple[int, int], int],
        num_buckets: int,
        evaluator: Optional[HandEvaluator] = None,
    ):
        """
        Args:
            next_street_buckets: Mapping from (card1, card2) → bucket_id
                                for the next street
            num_buckets: Number of buckets at the next street
            evaluator: Hand evaluator instance
        """
        self.next_buckets = next_street_buckets
        self.num_buckets = num_buckets
        self.evaluator = evaluator or HandEvaluator()
    
    def compute_transition(
        self,
        hole_cards: List[int],
        board: List[int],
    ) -> np.ndarray:
        """
        Compute transition histogram for a hand at the current street.
        
        For each possible next card, look up which bucket the hand
        falls into at the next street, and build a histogram.
        
        Args:
            hole_cards: Player's 2 hole cards
            board: Current board cards
        
        Returns:
            Histogram over next-street buckets (normalized)
        """
        used = set(hole_cards + board)
        remaining = [c for c in range(52) if c not in used]
        
        hist = np.zeros(self.num_buckets)
        
        for card in remaining:
            new_board = board + [card]
            # Look up the bucket for this hand on the next street
            # The bucket depends on the hole cards and full board
            hand_key = tuple(sorted(hole_cards))
            
            if hand_key in self.next_buckets:
                bucket = self.next_buckets[hand_key]
                hist[bucket] += 1
        
        # Normalize
        total = hist.sum()
        if total > 0:
            hist /= total
        else:
            hist[:] = 1.0 / self.num_buckets
        
        return hist
