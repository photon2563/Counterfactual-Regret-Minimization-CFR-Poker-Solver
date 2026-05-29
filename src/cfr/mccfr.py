"""
Monte Carlo Counterfactual Regret Minimization (MCCFR).

Implements two key sampling variants:
1. External Sampling MCCFR: Samples opponent and chance actions, traverses
   all of the current player's actions. Lower variance than outcome sampling.
2. Outcome Sampling MCCFR: Samples a single trajectory per iteration.
   Highest variance but cheapest per iteration.

External Sampling is the recommended default — it provides a good balance
of per-iteration cost and convergence speed.

References:
    Lanctot et al. (2009). "Monte Carlo Sampling for Regret Minimization
    in Extensive Games."
"""

from __future__ import annotations
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_engine.game_state import GameState, Player, Action
from cfr.vanilla_cfr import InfoSetData, VanillaCFR


class ExternalSamplingMCCFR(VanillaCFR):
    """
    External Sampling MCCFR.
    
    Samples chance and opponent actions from their current strategy,
    but traverses ALL actions of the traversing player.
    
    This provides unbiased regret estimates with lower variance than
    outcome sampling, at the cost of slightly more traversal per iteration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def train(
        self,
        initial_state: GameState,
        num_iterations: int = 1,
        alternating: bool = True,
    ) -> float:
        """
        Run External Sampling MCCFR training.
        
        External sampling always uses alternating updates (one player
        traverses per iteration).
        """
        total_value = 0.0
        
        for _ in range(num_iterations):
            self.iteration += 1
            
            for update_player in [Player.PLAYER_0, Player.PLAYER_1]:
                self._es_cfr(
                    state=initial_state,
                    update_player=update_player,
                )
            
            if self.dcfr_params is not None:
                self._apply_dcfr_discounting()
        
        return total_value
    
    def _es_cfr(
        self,
        state: GameState,
        update_player: Player,
    ) -> float:
        """
        External Sampling CFR recursive traversal.
        
        Args:
            state: Current game state
            update_player: The player whose regrets we're updating
        
        Returns:
            Counterfactual value for the update_player.
        """
        if state.is_terminal():
            return state.terminal_utility(update_player)
        
        if state.is_chance_node():
            # Sample a single chance outcome
            outcomes = state.chance_outcomes()
            probs = [p for _, p in outcomes]
            idx = self.rng.choices(range(len(outcomes)), weights=probs, k=1)[0]
            next_state, _ = outcomes[idx]
            return self._es_cfr(next_state, update_player)
        
        player = state.current_player()
        actions = state.legal_actions()
        info_key = state.information_set_key()
        info_set = self._get_info_set(info_key, actions)
        strategy = self.get_current_strategy(info_set)
        
        if player == update_player:
            # Traverse ALL actions for the update player
            action_values = np.zeros(len(actions), dtype=np.float64)
            
            for i, action in enumerate(actions):
                next_state = state.apply_action(action)
                action_values[i] = self._es_cfr(next_state, update_player)
            
            # Node value = weighted average under current strategy
            node_value = np.dot(strategy, action_values)
            
            # Update regrets
            for i in range(len(actions)):
                info_set.cumulative_regret[i] += action_values[i] - node_value
            
            if self.cfr_plus:
                info_set.cumulative_regret = np.maximum(info_set.cumulative_regret, 0)
            
            # Update strategy sum
            weight = self.iteration if self.linear_averaging else 1.0
            info_set.strategy_sum += strategy * weight
            
            return node_value
        else:
            # Sample a single action for the opponent
            action_idx = self.rng.choices(range(len(actions)), weights=strategy.tolist(), k=1)[0]
            action = actions[action_idx]
            next_state = state.apply_action(action)
            return self._es_cfr(next_state, update_player)


class OutcomeSamplingMCCFR(VanillaCFR):
    """
    Outcome Sampling MCCFR.
    
    Samples a single trajectory from root to terminal node per iteration.
    This is the cheapest per-iteration cost but has the highest variance.
    
    Uses importance sampling to correct for the sampling bias.
    Epsilon-on-policy exploration ensures all actions are sampled.
    """
    
    def __init__(self, epsilon: float = 0.6, **kwargs):
        """
        Args:
            epsilon: Exploration parameter. With probability epsilon,
                    sample uniformly; otherwise sample from strategy.
                    Higher epsilon = more exploration, lower variance.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.rng = random.Random()
    
    def set_seed(self, seed: int):
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def train(
        self,
        initial_state: GameState,
        num_iterations: int = 1,
        alternating: bool = True,
    ) -> float:
        total_value = 0.0
        
        for _ in range(num_iterations):
            self.iteration += 1
            
            for update_player in [Player.PLAYER_0, Player.PLAYER_1]:
                self._os_cfr(
                    state=initial_state,
                    update_player=update_player,
                    pi_i=1.0,       # Reach prob of update player
                    pi_neg_i=1.0,   # Reach prob of opponent + chance
                    sample_prob=1.0, # Probability of sampling this trajectory
                )
            
            if self.dcfr_params is not None:
                self._apply_dcfr_discounting()
        
        return total_value
    
    def _os_cfr(
        self,
        state: GameState,
        update_player: Player,
        pi_i: float,
        pi_neg_i: float,
        sample_prob: float,
    ) -> float:
        """
        Outcome Sampling CFR.
        
        Args:
            state: Current game state
            update_player: Player whose regrets we're updating
            pi_i: Reach probability of update_player
            pi_neg_i: Reach probability of all other players (opponent + chance)
            sample_prob: Probability of this trajectory being sampled
        
        Returns:
            Importance-weighted counterfactual value estimate.
        """
        if state.is_terminal():
            return state.terminal_utility(update_player) / sample_prob
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            probs = [p for _, p in outcomes]
            idx = self.rng.choices(range(len(outcomes)), weights=probs, k=1)[0]
            next_state, prob = outcomes[idx]
            return self._os_cfr(
                next_state, update_player,
                pi_i, pi_neg_i * prob, sample_prob * prob
            )
        
        player = state.current_player()
        actions = state.legal_actions()
        info_key = state.information_set_key()
        info_set = self._get_info_set(info_key, actions)
        strategy = self.get_current_strategy(info_set)
        
        # Epsilon-on-policy sampling distribution
        sample_strategy = self.epsilon / len(actions) + (1 - self.epsilon) * strategy
        
        # Sample one action
        action_idx = self.rng.choices(
            range(len(actions)), weights=sample_strategy.tolist(), k=1
        )[0]
        action = actions[action_idx]
        
        if player == update_player:
            next_state = state.apply_action(action)
            child_value = self._os_cfr(
                next_state, update_player,
                pi_i * strategy[action_idx],
                pi_neg_i,
                sample_prob * sample_strategy[action_idx],
            )
            
            # Compute counterfactual values
            # W = pi_{-i} / sample_prob (importance weight for suffix)
            W = pi_neg_i / sample_prob
            
            # For the sampled action
            node_value = strategy[action_idx] * child_value
            
            # Update regrets (estimated)
            for i in range(len(actions)):
                if i == action_idx:
                    regret = (child_value - node_value) * W
                else:
                    regret = -node_value * W
                info_set.cumulative_regret[i] += regret
            
            if self.cfr_plus:
                info_set.cumulative_regret = np.maximum(info_set.cumulative_regret, 0)
            
            weight = self.iteration if self.linear_averaging else 1.0
            info_set.strategy_sum += pi_i * strategy * weight / sample_prob
            
            return node_value
        else:
            next_state = state.apply_action(action)
            return self._os_cfr(
                next_state, update_player,
                pi_i,
                pi_neg_i * strategy[action_idx],
                sample_prob * sample_strategy[action_idx],
            )
