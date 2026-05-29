"""
Best Response Algorithm and Exploitability Computation.

Computes exploitability — the gold-standard quality metric for GTO solvers.

    exploitability = BR_EV(P0 vs σ_P1) + BR_EV(P1 vs σ_P0)

where BR_EV(Pi vs σ_Pj) is the maximum expected value player i can achieve
by choosing any valid (info-set-respecting) strategy against j's strategy.

A Nash equilibrium has exploitability = 0.

This implementation uses an efficient recursive approach where the BR
player's optimal action at each info set is determined by comparing
counterfactual values that are properly aggregated across all states
in the info set.

Algorithm (from Johanson et al. "Accelerating Best Response Calculation"):
1. Walk the tree, computing the counterfactual value of each action at
   each BR player info set, weighted by the probability of reaching
   that state given opponent + chance actions.
2. At each BR player info set, select the action maximizing this
   weighted counterfactual value.
3. Re-traverse the tree with the constructed BR strategy to compute EV.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_engine.game_state import GameState, PokerGame, Player, Action


class BestResponse:
    """
    Computes exact exploitability via information-set-level best response.
    
    For small games (< 20 info sets per player), uses exhaustive enumeration
    of all pure strategies to find the exact best response.
    For larger games, uses the recursive counterfactual value aggregation.
    """
    
    MAX_ENUM_INFOSETS = 20  # Max info sets for exhaustive enumeration
    
    def compute_exploitability(
        self,
        game: PokerGame,
        strategy: Dict[str, Dict[Action, float]],
    ) -> float:
        """Total exploitability = BR(P0) + BR(P1)."""
        br_p0 = self._compute_br_ev(game, Player.PLAYER_0, strategy)
        br_p1 = self._compute_br_ev(game, Player.PLAYER_1, strategy)
        return br_p0 + br_p1
    
    def compute_exploitability_mbb(
        self,
        game: PokerGame,
        strategy: Dict[str, Dict[Action, float]],
        big_blind: float = 2.0,
    ) -> float:
        """Exploitability in milli-big-blinds per hand."""
        return (self.compute_exploitability(game, strategy) / big_blind) * 1000.0
    
    def _compute_br_ev(
        self,
        game: PokerGame,
        br_player: Player,
        opponent_strategy: Dict[str, Dict[Action, float]],
    ) -> float:
        """
        Compute the EV of the best response for br_player.
        
        Collects BR player info sets, then enumerates or recursively
        computes the optimal strategy.
        """
        # First, collect all BR player info sets and their actions
        br_info_sets = {}  # key -> list of actions
        self._collect_info_sets(
            game.initial_state(), br_player, br_info_sets
        )
        
        if len(br_info_sets) <= self.MAX_ENUM_INFOSETS:
            return self._enumerate_br(
                game, br_player, opponent_strategy, br_info_sets
            )
        else:
            # For large games, use the recursive approach
            return self._recursive_br(
                game, br_player, opponent_strategy
            )
    
    def _collect_info_sets(
        self,
        state: GameState,
        br_player: Player,
        info_sets: Dict[str, List[Action]],
    ) -> None:
        """Collect all info sets and their actions for the BR player."""
        if state.is_terminal():
            return
        
        if state.is_chance_node():
            for next_state, _ in state.chance_outcomes():
                self._collect_info_sets(next_state, br_player, info_sets)
            return
        
        player = state.current_player()
        actions = state.legal_actions()
        info_key = state.information_set_key()
        
        if player == br_player and info_key not in info_sets:
            info_sets[info_key] = actions
        
        for action in actions:
            next_state = state.apply_action(action)
            self._collect_info_sets(next_state, br_player, info_sets)
    
    def _enumerate_br(
        self,
        game: PokerGame,
        br_player: Player,
        opponent_strategy: Dict[str, Dict[Action, float]],
        br_info_sets: Dict[str, List[Action]],
    ) -> float:
        """
        Find the best response by enumerating all pure strategies.
        
        For each possible pure strategy (one action per info set),
        compute the resulting EV. Return the maximum.
        """
        keys = sorted(br_info_sets.keys())
        action_lists = [br_info_sets[k] for k in keys]
        
        best_ev = float('-inf')
        
        for actions_combo in product(*action_lists):
            # Build a pure strategy from this combination
            br_strategy = {}
            for key, action in zip(keys, actions_combo):
                br_strategy[key] = {
                    a: (1.0 if a == action else 0.0)
                    for a in br_info_sets[key]
                }
            
            # Compute EV with this strategy
            ev = self._compute_ev(
                game.initial_state(),
                br_player,
                br_player,
                br_strategy,
                opponent_strategy,
            )
            best_ev = max(best_ev, ev)
        
        return best_ev
    
    def _recursive_br(
        self,
        game: PokerGame,
        br_player: Player,
        opponent_strategy: Dict[str, Dict[Action, float]],
    ) -> float:
        """
        Compute BR using recursive counterfactual value aggregation.
        
        This works for larger games where enumeration is infeasible.
        It computes the correct info-set-level BR by aggregating
        action values across states within each info set.
        """
        # Walk tree collecting action values per info set
        info_set_values = defaultdict(lambda: defaultdict(float))
        
        self._recursive_br_walk(
            state=game.initial_state(),
            br_player=br_player,
            opp_strat=opponent_strategy,
            pi_opp=1.0,  # opponent × chance reach probability
            info_set_values=info_set_values,
        )
        
        # Build BR strategy from best actions
        br_strategy = {}
        for key, action_vals in info_set_values.items():
            best_action = max(action_vals, key=action_vals.get)
            br_strategy[key] = {a: (1.0 if a == best_action else 0.0)
                               for a in action_vals}
        
        # Compute EV
        return self._compute_ev(
            game.initial_state(),
            br_player,
            br_player,
            br_strategy,
            opponent_strategy,
        )
    
    def _recursive_br_walk(
        self,
        state: GameState,
        br_player: Player,
        opp_strat: Dict[str, Dict[Action, float]],
        pi_opp: float,
        info_set_values: Dict[str, Dict[Action, float]],
    ) -> float:
        """
        Walk the tree computing BR action values recursively.
        
        Returns the counterfactual value for br_player at this state.
        """
        if state.is_terminal():
            return state.terminal_utility(br_player)
        
        if state.is_chance_node():
            value = 0.0
            for next_state, prob in state.chance_outcomes():
                value += prob * self._recursive_br_walk(
                    next_state, br_player, opp_strat,
                    pi_opp * prob, info_set_values,
                )
            return value
        
        player = state.current_player()
        actions = state.legal_actions()
        info_key = state.information_set_key()
        
        if player == br_player:
            # Compute value of each action
            action_vals = {}
            for action in actions:
                next_state = state.apply_action(action)
                action_vals[action] = self._recursive_br_walk(
                    next_state, br_player, opp_strat,
                    pi_opp, info_set_values,
                )
            
            # Accumulate into info set data
            for action, val in action_vals.items():
                info_set_values[info_key][action] += pi_opp * val
            
            # Return max value (optimistic estimate)
            return max(action_vals.values())
        else:
            # Opponent follows strategy
            strat = opp_strat.get(info_key, None)
            value = 0.0
            if strat is None:
                prob = 1.0 / len(actions)
                for action in actions:
                    next_state = state.apply_action(action)
                    value += prob * self._recursive_br_walk(
                        next_state, br_player, opp_strat,
                        pi_opp * prob, info_set_values,
                    )
            else:
                for action in actions:
                    prob = strat.get(action, 0.0)
                    if prob > 1e-10:
                        next_state = state.apply_action(action)
                        value += prob * self._recursive_br_walk(
                            next_state, br_player, opp_strat,
                            pi_opp * prob, info_set_values,
                        )
            return value
    
    def _compute_ev(
        self,
        state: GameState,
        player: Player,
        br_player: Player,
        br_strategy: Dict[str, Dict[Action, float]],
        opp_strat: Dict[str, Dict[Action, float]],
    ) -> float:
        """Compute EV when BR player follows br_strategy, opponent follows opp_strat."""
        if state.is_terminal():
            return state.terminal_utility(player)
        
        if state.is_chance_node():
            value = 0.0
            for next_state, prob in state.chance_outcomes():
                value += prob * self._compute_ev(
                    next_state, player, br_player, br_strategy, opp_strat
                )
            return value
        
        current = state.current_player()
        actions = state.legal_actions()
        info_key = state.information_set_key()
        
        strat = br_strategy.get(info_key) if current == br_player else opp_strat.get(info_key)
        
        if strat is None:
            prob = 1.0 / len(actions)
            return sum(
                prob * self._compute_ev(
                    state.apply_action(a), player, br_player, br_strategy, opp_strat
                )
                for a in actions
            )
        
        return sum(
            strat.get(a, 0.0) * self._compute_ev(
                state.apply_action(a), player, br_player, br_strategy, opp_strat
            )
            for a in actions if strat.get(a, 0.0) > 1e-10
        )
    
    def compute_game_value(
        self,
        game: PokerGame,
        strategy: Dict[str, Dict[Action, float]],
        player: Player = Player.PLAYER_0,
    ) -> float:
        """Compute EV when both players follow the same strategy profile."""
        return self._ev_walk(game.initial_state(), player, strategy)
    
    def _ev_walk(
        self,
        state: GameState,
        player: Player,
        strategy: Dict[str, Dict[Action, float]],
    ) -> float:
        if state.is_terminal():
            return state.terminal_utility(player)
        if state.is_chance_node():
            return sum(
                p * self._ev_walk(ns, player, strategy)
                for ns, p in state.chance_outcomes()
            )
        
        actions = state.legal_actions()
        info_key = state.information_set_key()
        strat = strategy.get(info_key)
        
        if strat is None:
            prob = 1.0 / len(actions)
            return sum(prob * self._ev_walk(state.apply_action(a), player, strategy)
                      for a in actions)
        
        return sum(
            strat.get(a, 0.0) * self._ev_walk(state.apply_action(a), player, strategy)
            for a in actions if strat.get(a, 0.0) > 1e-10
        )
