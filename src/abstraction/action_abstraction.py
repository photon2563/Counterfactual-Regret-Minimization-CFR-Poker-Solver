"""
Action Abstraction (Bet Sizing).

Limits the infinite action space of continuous bet sizes in No-Limit games
to a discrete set of strategic options (e.g., 0.5x pot, 1x pot, All-in).

Wraps an existing GameState and intercepts legal_actions() to return
only the discrete abstract actions.

References:
    Gilpin et al. (2007). "Better Automated Abstraction Techniques
    for Imperfect Information Games"
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Callable

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_engine.game_state import GameState, Action, Player


class BetAction(int):
    """
    Extension of Action to include bet amounts.
    Used internally to represent specific sizing.
    """
    def __new__(cls, action_type: int, amount: float = 0.0):
        obj = super().__new__(cls, action_type)
        obj.amount = amount
        return obj


class ActionAbstractedState(GameState):
    """
    Wraps a GameState to restrict its action space.
    
    If the underlying game state has continuous action spaces (e.g.,
    returns a generic BET action but allows any amount), this wrapper
    replaces that with specific BetActions (e.g., BET 0.5x, BET 1x).
    """
    
    def __init__(self, state: GameState, allowed_fractions: List[float]):
        """
        Args:
            state: The underlying true game state
            allowed_fractions: List of pot fractions allowed for bets/raises
                               e.g. [0.5, 1.0, 2.0]
        """
        self.state = state
        self.allowed_fractions = allowed_fractions
    
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def is_chance_node(self) -> bool:
        return self.state.is_chance_node()
    
    def current_player(self) -> Player:
        return self.state.current_player()
    
    def legal_actions(self) -> List[Action]:
        base_actions = self.state.legal_actions()
        
        # If the game doesn't support parameterized bets yet (like Kuhn/Leduc),
        # we just return the base actions.
        # If it's a generic No-Limit state, we would expand BET/RAISE.
        
        expanded_actions = []
        for a in base_actions:
            if a == Action.BET or a == Action.RAISE:
                # E.g. Pot is $100. We allow 50%, 100% bets.
                # In a real NLHE state, we'd query state.pot_size()
                # For this wrapper demonstration, we just yield the standard action
                # alongside the theoretical bet actions.
                expanded_actions.append(a)
                # Future:
                # pot = self.state.current_pot()
                # for frac in self.allowed_fractions:
                #     expanded_actions.append(BetAction(a, pot * frac))
            else:
                expanded_actions.append(a)
                
        return expanded_actions
    
    def apply_action(self, action: Action) -> 'GameState':
        next_state = self.state.apply_action(action)
        return ActionAbstractedState(next_state, self.allowed_fractions)
    
    def terminal_utility(self, player: Player) -> float:
        return self.state.terminal_utility(player)
    
    def information_set_key(self) -> str:
        return self.state.information_set_key()
    
    def chance_outcomes(self) -> List[Tuple['GameState', float]]:
        outcomes = self.state.chance_outcomes()
        return [
            (ActionAbstractedState(s, self.allowed_fractions), p)
            for s, p in outcomes
        ]
        
    @property
    def num_players(self) -> int:
        return self.state.num_players
