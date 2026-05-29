"""
Abstract base for extensive-form game states.

Provides the interface that all poker game environments must implement
for compatibility with CFR algorithms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class Player(IntEnum):
    """Player identifiers."""
    CHANCE = -1  # Nature / chance player (deals cards)
    PLAYER_0 = 0
    PLAYER_1 = 1


class Action(IntEnum):
    """
    Standard poker actions.
    
    These cover all actions needed for Kuhn, Leduc, and simplified NLHE.
    For NLHE with continuous bet sizing, actions are extended with bet amounts.
    """
    FOLD = 0
    CHECK = 1  # Also used for "pass" in Kuhn
    CALL = 2
    BET = 3    # Fixed bet (1 unit in Kuhn, varies in Leduc)
    RAISE = 4  # Raise on top of opponent's bet


class GameState(ABC):
    """
    Abstract base class representing a state in an extensive-form game.
    
    Encapsulates the full game state including private information,
    public information, and the history of actions taken.
    
    CFR algorithms interact with games exclusively through this interface.
    """
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if this state is a terminal (leaf) node."""
        pass
    
    @abstractmethod
    def is_chance_node(self) -> bool:
        """Return True if this state is a chance node (nature acts)."""
        pass
    
    @abstractmethod
    def current_player(self) -> Player:
        """Return the player whose turn it is to act."""
        pass
    
    @abstractmethod
    def legal_actions(self) -> List[Action]:
        """Return the list of legal actions at this state."""
        pass
    
    @abstractmethod
    def apply_action(self, action: Action) -> 'GameState':
        """
        Apply an action and return the resulting game state.
        
        This should return a NEW GameState (immutable pattern) to support
        recursive CFR traversal without state corruption.
        """
        pass
    
    @abstractmethod
    def terminal_utility(self, player: Player) -> float:
        """
        Return the utility (payoff) for the given player at a terminal state.
        
        In zero-sum games: utility(P0) = -utility(P1).
        """
        pass
    
    @abstractmethod
    def information_set_key(self) -> str:
        """
        Return a unique string key identifying this player's information set.
        
        The information set contains all states that are indistinguishable
        to the current player — i.e., they see the same private cards and
        the same public action history, but the opponent's cards differ.
        
        This key is used as the dictionary key for storing regrets and
        strategies in the CFR algorithm.
        """
        pass
    
    @abstractmethod
    def chance_outcomes(self) -> List[Tuple['GameState', float]]:
        """
        Return all possible chance outcomes with their probabilities.
        
        Only valid when is_chance_node() returns True.
        Returns: List of (next_state, probability) tuples.
        """
        pass
    
    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of strategic players (excluding chance)."""
        pass


class PokerGame(ABC):
    """
    Abstract base class for a poker game variant.
    
    Provides factory methods for creating initial game states
    and metadata about the game.
    """
    
    @abstractmethod
    def initial_state(self) -> GameState:
        """Create and return the initial (root) game state."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this game variant."""
        pass
    
    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of strategic players."""
        pass
