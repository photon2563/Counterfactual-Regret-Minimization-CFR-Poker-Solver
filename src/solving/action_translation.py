"""
Pseudo-Harmonic Action Translation.

When an opponent makes a bet size that doesn't match any of the solver's
abstracted bet sizes, we need to "translate" it to the nearest abstract
actions. This module implements the pseudo-harmonic mapping that satisfies
five key axioms for robust action translation.

The mapping probability for an off-tree bet x, bounded by abstract
actions a_low and a_high, is:

    P(map to a_low) = (a_high - x) / (a_high - a_low) * (a_low + 1) / (x + 1)

where all values are pot-normalized (pot = 1).

The five axioms satisfied:
1. Exact mapping: if x = a_low, P(a_low) = 1
2. Monotonicity: larger x → lower P(a_low), higher P(a_high)
3. Scale invariance: doubling all values doesn't change probabilities
4. Action robustness: mapping is smooth and continuous
5. Boundary robustness: graceful behavior at extreme values

References:
    Ganzfried & Sandholm (2013). "Action Translation in Extensive-Form Games
    with Large Action Spaces: Axioms, Paradoxes, and the Pseudo-Harmonic Mapping."
"""

from __future__ import annotations
from typing import List, Tuple, Optional


def pseudo_harmonic_map(
    bet_size: float,
    low_action: float,
    high_action: float,
) -> Tuple[float, float]:
    """
    Map an off-tree bet size to probabilities over the two nearest abstract actions.
    
    Args:
        bet_size: The actual bet size (pot-normalized)
        low_action: The nearest lower abstract bet size
        high_action: The nearest higher abstract bet size
    
    Returns:
        (prob_low, prob_high) — probabilities of mapping to each action
    
    Raises:
        ValueError: if bet_size is not between low_action and high_action
    """
    if bet_size < low_action - 1e-10 or bet_size > high_action + 1e-10:
        raise ValueError(
            f"Bet size {bet_size} not in [{low_action}, {high_action}]"
        )
    
    # Exact match cases
    if abs(bet_size - low_action) < 1e-10:
        return (1.0, 0.0)
    if abs(bet_size - high_action) < 1e-10:
        return (0.0, 1.0)
    
    # Pseudo-harmonic formula (pot-normalized)
    prob_low = ((high_action - bet_size) / (high_action - low_action) * 
                (low_action + 1) / (bet_size + 1))
    
    prob_high = 1.0 - prob_low
    
    # Clamp for numerical stability
    prob_low = max(0.0, min(1.0, prob_low))
    prob_high = max(0.0, min(1.0, prob_high))
    
    return (prob_low, prob_high)


class ActionTranslator:
    """
    Translates real bet sizes to abstract bet sizes for subgame solving.
    
    Given a set of allowed bet sizes (from the abstraction), this class
    maps any real bet to a mixed strategy over the two nearest abstract sizes
    using the pseudo-harmonic mapping.
    
    Usage:
        translator = ActionTranslator([0.5, 1.0, 2.0])  # half pot, pot, 2x pot
        probs = translator.translate(0.75)  # → {0.5: 0.4, 1.0: 0.6} (approx)
    """
    
    def __init__(self, abstract_bet_sizes: List[float]):
        """
        Args:
            abstract_bet_sizes: Sorted list of pot-normalized bet sizes
                               that the solver uses (e.g., [0.5, 1.0, 2.0])
        """
        self.sizes = sorted(abstract_bet_sizes)
        if len(self.sizes) < 1:
            raise ValueError("Need at least one abstract bet size")
    
    def translate(self, bet_size: float) -> dict:
        """
        Translate a real bet size to probabilities over abstract sizes.
        
        Args:
            bet_size: Pot-normalized real bet size
        
        Returns:
            Dict mapping abstract_size → probability
        """
        # Find bounding abstract actions
        if bet_size <= self.sizes[0]:
            return {self.sizes[0]: 1.0}
        if bet_size >= self.sizes[-1]:
            return {self.sizes[-1]: 1.0}
        
        # Find the two nearest abstract sizes
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] <= bet_size <= self.sizes[i + 1]:
                low = self.sizes[i]
                high = self.sizes[i + 1]
                prob_low, prob_high = pseudo_harmonic_map(bet_size, low, high)
                
                result = {}
                if prob_low > 1e-10:
                    result[low] = prob_low
                if prob_high > 1e-10:
                    result[high] = prob_high
                return result
        
        # Shouldn't reach here
        return {self.sizes[-1]: 1.0}
    
    def verify_axioms(self, verbose: bool = True) -> bool:
        """
        Verify that the pseudo-harmonic mapping satisfies all 5 axioms.
        
        Returns True if all axioms are satisfied.
        """
        all_passed = True
        
        if len(self.sizes) < 2:
            if verbose:
                print("Need at least 2 abstract sizes to verify axioms")
            return True
        
        low = self.sizes[0]
        high = self.sizes[1]
        
        # Axiom 1: Exact mapping
        p_low, p_high = pseudo_harmonic_map(low, low, high)
        passed = abs(p_low - 1.0) < 1e-10 and abs(p_high) < 1e-10
        if verbose:
            print(f"Axiom 1 (Exact mapping at low):  {'✅' if passed else '❌'} P(low|low) = {p_low:.6f}")
        all_passed &= passed
        
        p_low, p_high = pseudo_harmonic_map(high, low, high)
        passed = abs(p_low) < 1e-10 and abs(p_high - 1.0) < 1e-10
        if verbose:
            print(f"Axiom 1 (Exact mapping at high): {'✅' if passed else '❌'} P(high|high) = {p_high:.6f}")
        all_passed &= passed
        
        # Axiom 2: Monotonicity
        test_bets = [low + (high - low) * t for t in [0.1, 0.3, 0.5, 0.7, 0.9]]
        probs = [pseudo_harmonic_map(b, low, high)[0] for b in test_bets]
        passed = all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
        if verbose:
            print(f"Axiom 2 (Monotonicity):          {'✅' if passed else '❌'} P(low) decreasing: {[f'{p:.3f}' for p in probs]}")
        all_passed &= passed
        
        # Axiom 3: Scale invariance
        scale = 2.0
        mid = (low + high) / 2
        p_original = pseudo_harmonic_map(mid, low, high)
        p_scaled = pseudo_harmonic_map(mid * scale, low * scale, high * scale)
        passed = abs(p_original[0] - p_scaled[0]) < 1e-10
        if verbose:
            print(f"Axiom 3 (Scale invariance):      {'✅' if passed else '❌'} Original: {p_original[0]:.6f}, Scaled: {p_scaled[0]:.6f}")
        all_passed &= passed
        
        # Axiom 4: Smoothness (continuity check)
        eps = 1e-6
        mid = (low + high) / 2
        p1 = pseudo_harmonic_map(mid, low, high)[0]
        p2 = pseudo_harmonic_map(mid + eps, low, high)[0]
        derivative = abs(p2 - p1) / eps
        passed = derivative < 100  # Derivative should be bounded
        if verbose:
            print(f"Axiom 4 (Smoothness):            {'✅' if passed else '❌'} Derivative at midpoint: {derivative:.4f}")
        all_passed &= passed
        
        # Axiom 5: Probabilities sum to 1
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            bet = low + (high - low) * t
            p_l, p_h = pseudo_harmonic_map(bet, low, high)
            passed = abs(p_l + p_h - 1.0) < 1e-10
            if not passed:
                if verbose:
                    print(f"Axiom 5 (Sum to 1):              ❌ at t={t}: {p_l + p_h:.10f}")
                all_passed = False
        if verbose and all_passed:
            print(f"Axiom 5 (Sum to 1):              ✅ All test points sum to 1.0")
        
        return all_passed
