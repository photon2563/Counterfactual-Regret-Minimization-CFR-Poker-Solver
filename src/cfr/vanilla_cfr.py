"""
Vanilla Counterfactual Regret Minimization (CFR).

This is the foundational CFR algorithm that guarantees convergence to a
Nash equilibrium in two-player zero-sum extensive-form games.

The algorithm works by:
1. Traversing the full game tree on each iteration
2. Computing counterfactual values at each information set
3. Accumulating regret for each action (how much better it would have been)
4. Using Regret Matching to derive the next iteration's strategy
5. Tracking the average strategy, which converges to Nash equilibrium

Also supports CFR+ mode with three key modifications:
- Non-negative regret clamping (floor at zero)
- Alternating updates (update one player per iteration)
- Linear averaging (weight iteration t strategy by t)

Mathematical Foundations:
    Let σ be a strategy profile, I an information set, a an action.
    
    Counterfactual value:
        v_i(σ, I) = Σ_{h∈I} π_{-i}^σ(h) · Σ_{z∈Z_h} π^σ(h,z) · u_i(z)
    
    Instantaneous counterfactual regret:
        r^t(I, a) = v_i(σ^t|_{I→a}, I) - v_i(σ^t, I)
    
    Cumulative regret:
        R^T(I, a) = Σ_{t=1}^{T} r^t(I, a)
    
    Regret matching strategy:
        σ^{T+1}(I, a) = R^T_+(I, a) / Σ_b R^T_+(I, b)
    where R_+ = max(R, 0)

References:
    Zinkevich et al. (2007). "Regret Minimization in Games with
    Incomplete Information."
    Tammelin (2014). "Solving Large Imperfect Information Games Using CFR+"
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_engine.game_state import GameState, Player, Action


@dataclass
class InfoSetData:
    """
    Data stored for each information set during CFR training.
    
    Attributes:
        num_actions: Number of legal actions at this information set
        cumulative_regret: Accumulated counterfactual regret for each action
        strategy_sum: Sum of all strategies weighted by reach probability
                     (for computing the average strategy)
        action_list: The legal actions at this information set
    """
    num_actions: int
    cumulative_regret: np.ndarray
    strategy_sum: np.ndarray
    action_list: List[Action] = field(default_factory=list)
    
    @staticmethod
    def create(actions: List[Action]) -> 'InfoSetData':
        n = len(actions)
        return InfoSetData(
            num_actions=n,
            cumulative_regret=np.zeros(n, dtype=np.float64),
            strategy_sum=np.zeros(n, dtype=np.float64),
            action_list=list(actions),
        )


class VanillaCFR:
    """
    Vanilla CFR solver with optional CFR+ and DCFR modes.
    
    Usage:
        game = KuhnPoker()
        solver = VanillaCFR()
        
        for i in range(10000):
            solver.train(game.initial_state())
        
        strategy = solver.get_average_strategy()
        exploitability = solver.compute_exploitability(game)
    """
    
    def __init__(
        self,
        cfr_plus: bool = False,
        dcfr_params: Optional[Dict[str, float]] = None,
        linear_averaging: bool = False,
    ):
        """
        Initialize the CFR solver.
        
        Args:
            cfr_plus: Enable CFR+ (non-negative regrets + linear averaging)
            dcfr_params: Dict with 'alpha', 'beta', 'gamma' for Discounted CFR.
                        If None, standard CFR is used.
            linear_averaging: Weight iteration t's strategy contribution by t.
        """
        self.info_sets: Dict[str, InfoSetData] = {}
        self.cfr_plus = cfr_plus
        self.dcfr_params = dcfr_params
        self.linear_averaging = linear_averaging or cfr_plus
        self.iteration = 0
        
        # Tracking metrics
        self.exploitability_history: List[float] = []
        self.ev_history: List[float] = []
    
    def _get_info_set(self, key: str, actions: List[Action]) -> InfoSetData:
        """Get or create the InfoSetData for an information set."""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetData.create(actions)
        return self.info_sets[key]
    
    def get_current_strategy(self, info_set: InfoSetData) -> np.ndarray:
        """
        Compute the current strategy using Regret Matching.
        
        Strategy is proportional to positive cumulative regrets.
        If all regrets are non-positive, use uniform random.
        
        Returns:
            Probability distribution over actions.
        """
        regrets = info_set.cumulative_regret.copy()
        positive_regret = np.maximum(regrets, 0)
        total = positive_regret.sum()
        
        if total > 0:
            return positive_regret / total
        else:
            # Uniform distribution when no positive regrets
            return np.ones(info_set.num_actions) / info_set.num_actions
    
    def get_average_strategy(self) -> Dict[str, Dict[Action, float]]:
        """
        Compute the average strategy across all iterations.
        
        The average strategy is what converges to Nash equilibrium.
        
        Returns:
            Dict mapping info_set_key → {action: probability}
        """
        result = {}
        for key, data in self.info_sets.items():
            total = data.strategy_sum.sum()
            if total > 0:
                avg_strategy = data.strategy_sum / total
            else:
                avg_strategy = np.ones(data.num_actions) / data.num_actions
            
            result[key] = {
                data.action_list[i]: float(avg_strategy[i])
                for i in range(data.num_actions)
            }
        return result
    
    def train(
        self,
        initial_state: GameState,
        num_iterations: int = 1,
        alternating: bool = False,
    ) -> float:
        """
        Run CFR training for the specified number of iterations.
        
        Args:
            initial_state: The root game state (chance node).
            num_iterations: Number of iterations to run.
            alternating: Use alternating updates (update one player per iteration).
                        Automatically enabled in CFR+ mode.
        
        Returns:
            Average game value for Player 0 over the training iterations.
        """
        use_alternating = alternating or self.cfr_plus
        total_value = 0.0
        
        for _ in range(num_iterations):
            self.iteration += 1
            
            if use_alternating:
                # Alternating updates: update Player 0, then Player 1
                for update_player in [Player.PLAYER_0, Player.PLAYER_1]:
                    self._cfr_recursive(
                        state=initial_state,
                        reach_probs=np.ones(3),  # [P0_reach, P1_reach, chance_reach]
                        update_player=update_player,
                    )
            else:
                # Simultaneous updates
                value = self._cfr_recursive(
                    state=initial_state,
                    reach_probs=np.ones(3),
                    update_player=None,
                )
                total_value += value
            
            # Apply DCFR discounting at end of iteration
            if self.dcfr_params is not None:
                self._apply_dcfr_discounting()
        
        return total_value / num_iterations if not use_alternating else 0.0
    
    def _cfr_recursive(
        self,
        state: GameState,
        reach_probs: np.ndarray,  # [P0_reach, P1_reach, chance_reach]
        update_player: Optional[Player],
    ) -> float:
        """
        Recursive CFR traversal.
        
        Args:
            state: Current game state
            reach_probs: Reach probabilities [P0, P1, chance]
            update_player: If set, only update this player's regrets (alternating).
                          If None, update all players simultaneously.
        
        Returns:
            Expected counterfactual value for Player 0.
        """
        # Terminal node: return utility
        if state.is_terminal():
            return state.terminal_utility(Player.PLAYER_0)
        
        # Chance node: weighted average over all outcomes
        if state.is_chance_node():
            value = 0.0
            for next_state, prob in state.chance_outcomes():
                new_reach = reach_probs.copy()
                new_reach[2] *= prob  # Update chance reach probability
                value += prob * self._cfr_recursive(next_state, new_reach, update_player)
            return value
        
        # Decision node
        player = state.current_player()
        player_idx = int(player)
        actions = state.legal_actions()
        info_key = state.information_set_key()
        info_set = self._get_info_set(info_key, actions)
        
        # Get current strategy via regret matching
        strategy = self.get_current_strategy(info_set)
        
        # Compute counterfactual values for each action
        action_values = np.zeros(len(actions), dtype=np.float64)
        node_value = 0.0
        
        for i, action in enumerate(actions):
            next_state = state.apply_action(action)
            new_reach = reach_probs.copy()
            new_reach[player_idx] *= strategy[i]
            
            action_values[i] = self._cfr_recursive(next_state, new_reach, update_player)
            node_value += strategy[i] * action_values[i]
        
        # Update regrets and strategy sum (only for the acting player)
        if update_player is None or player == update_player:
            # Counterfactual reach probability (all players EXCEPT current)
            opponent_idx = 1 - player_idx
            cf_reach = reach_probs[opponent_idx] * reach_probs[2]
            
            # The recursive function returns values from Player 0's perspective.
            # In a zero-sum game, Player 1's values are negated.
            # We need regrets from the CURRENT player's perspective.
            sign = 1.0 if player_idx == 0 else -1.0
            
            # Counterfactual regret = (action_value - node_value) * counterfactual_reach
            for i in range(len(actions)):
                regret = sign * (action_values[i] - node_value) * cf_reach
                info_set.cumulative_regret[i] += regret
            
            # CFR+: clamp negative regrets to zero
            if self.cfr_plus:
                info_set.cumulative_regret = np.maximum(info_set.cumulative_regret, 0)
            
            # Update strategy sum (for average strategy computation)
            weight = self.iteration if self.linear_averaging else 1.0
            info_set.strategy_sum += reach_probs[player_idx] * strategy * weight
        
        return node_value
    
    def _apply_dcfr_discounting(self):
        """
        Apply Discounted CFR discounting to regrets and strategy sums.
        
        DCFR uses three hyperparameters:
        - alpha: discount positive regrets by t^α / (t^α + 1)
        - beta:  discount negative regrets by t^β / (t^β + 1)  
        - gamma: discount strategy sum contributions by (t/(t+1))^γ
        """
        t = self.iteration
        alpha = self.dcfr_params.get('alpha', 1.5)
        beta = self.dcfr_params.get('beta', 0.0)
        gamma = self.dcfr_params.get('gamma', 2.0)
        
        pos_discount = (t ** alpha) / (t ** alpha + 1)
        neg_discount = (t ** beta) / (t ** beta + 1)
        strat_discount = (t / (t + 1)) ** gamma
        
        for info_set in self.info_sets.values():
            # Discount positive and negative regrets separately
            positive_mask = info_set.cumulative_regret > 0
            negative_mask = info_set.cumulative_regret < 0
            
            info_set.cumulative_regret[positive_mask] *= pos_discount
            info_set.cumulative_regret[negative_mask] *= neg_discount
            
            # Discount strategy sum
            info_set.strategy_sum *= strat_discount
    
    def compute_exploitability(self, game) -> float:
        """
        Compute the exploitability of the current average strategy.
        
        Exploitability = BR_value(P0) + BR_value(P1)
        where BR_value(Pi) is the value a best response achieves against Pi.
        
        In a Nash equilibrium, exploitability = 0.
        
        Returns:
            Exploitability in game-units (e.g., chips per hand).
        """
        from evaluation.best_response import BestResponse
        br = BestResponse()
        avg_strategy = self.get_average_strategy()
        return br.compute_exploitability(game, avg_strategy)
    
    def train_and_track(
        self,
        game,
        num_iterations: int,
        eval_every: int = 100,
        alternating: bool = False,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train with periodic exploitability evaluation.
        
        Args:
            game: The PokerGame instance
            num_iterations: Total iterations to run
            eval_every: Evaluate exploitability every N iterations
            alternating: Use alternating updates
            verbose: Print progress
        
        Returns:
            Dict with 'iterations' and 'exploitability' lists for plotting.
        """
        iterations_list = []
        exploit_list = []
        
        for i in range(1, num_iterations + 1):
            self.train(game.initial_state(), num_iterations=1, alternating=alternating)
            
            if i % eval_every == 0 or i == 1 or i == num_iterations:
                exploit = self.compute_exploitability(game)
                iterations_list.append(i)
                exploit_list.append(exploit)
                
                if verbose:
                    print(f"Iteration {i:>6d} | Exploitability: {exploit:.6f} | "
                          f"Info sets: {len(self.info_sets)}")
        
        return {
            'iterations': iterations_list,
            'exploitability': exploit_list,
        }
    
    def print_strategy(self, game_name: str = ""):
        """Pretty-print the converged average strategy."""
        avg_strategy = self.get_average_strategy()
        
        print(f"\n{'='*60}")
        print(f"  Converged Strategy — {game_name}")
        print(f"  ({self.iteration} iterations, {len(self.info_sets)} information sets)")
        print(f"{'='*60}")
        
        for key in sorted(avg_strategy.keys()):
            actions = avg_strategy[key]
            action_strs = [f"{a.name}={p:.4f}" for a, p in sorted(actions.items())]
            print(f"  {key:>12s} → {', '.join(action_strs)}")
        
        print(f"{'='*60}\n")
