"""
Blueprint Strategy Pre-computation and Serialization.

The blueprint is the pre-computed GTO strategy for the abstracted game.
It's generated offline via millions of CFR iterations and stored as a
lookup table for real-time use.

The blueprint serves two purposes:
1. Direct play: use as the strategy when exact subgame solving is too slow
2. Opponent range estimation: initialize ranges for subgame solving
   based on the blueprint's reach probabilities

Serialization format (JSON):
{
    "game": "leduc_holdem",
    "iterations": 10000,
    "algorithm": "vanilla_cfr",
    "exploitability": 0.123,
    "strategy": {
        "info_set_key": {"ACTION_NAME": probability, ...},
        ...
    },
    "metadata": {
        "created_at": "...",
        "game_value_p0": -0.0556,
    }
}
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Optional, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_engine.game_state import PokerGame, Player, Action
from cfr.vanilla_cfr import VanillaCFR
from evaluation.best_response import BestResponse


class Blueprint:
    """
    Manages blueprint strategy computation and persistence.
    
    Usage:
        blueprint = Blueprint(game)
        blueprint.compute(num_iterations=100000, algorithm='cfr_plus')
        blueprint.save('blueprint_leduc.json')
        
        # Later:
        blueprint = Blueprint.load('blueprint_leduc.json')
        strategy = blueprint.strategy
    """
    
    def __init__(self, game: Optional[PokerGame] = None):
        self.game = game
        self.strategy: Dict[str, Dict[str, float]] = {}
        self.metadata: Dict[str, Any] = {}
        self.exploitability: float = float('inf')
    
    def compute(
        self,
        num_iterations: int = 10000,
        algorithm: str = 'vanilla',
        eval_every: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Compute the blueprint strategy via CFR.
        
        Args:
            num_iterations: Training iterations
            algorithm: 'vanilla', 'cfr_plus', or 'dcfr'
            eval_every: Evaluate exploitability every N iterations
            verbose: Print progress
        
        Returns:
            Training metrics
        """
        assert self.game is not None, "Game must be set"
        
        # Select algorithm
        if algorithm == 'cfr_plus':
            solver = VanillaCFR(cfr_plus=True)
        elif algorithm == 'dcfr':
            solver = VanillaCFR(dcfr_params={'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0})
        else:
            solver = VanillaCFR()
        
        if verbose:
            print(f"▸ Computing blueprint ({algorithm}, {num_iterations:,} iterations)...")
        
        t0 = time.time()
        metrics = solver.train_and_track(
            self.game, num_iterations=num_iterations,
            eval_every=eval_every, verbose=verbose,
        )
        elapsed = time.time() - t0
        
        # Extract strategy
        avg_strategy = solver.get_average_strategy()
        
        # Convert Action enum keys to strings for JSON serialization
        self.strategy = {}
        for key, action_probs in avg_strategy.items():
            self.strategy[key] = {
                action.name: prob for action, prob in action_probs.items()
            }
        
        # Compute final exploitability
        br = BestResponse()
        self.exploitability = br.compute_exploitability(self.game, avg_strategy)
        game_value = br.compute_game_value(self.game, avg_strategy, Player.PLAYER_0)
        
        self.metadata = {
            'game': self.game.name(),
            'algorithm': algorithm,
            'iterations': num_iterations,
            'exploitability': self.exploitability,
            'game_value_p0': game_value,
            'training_time': elapsed,
            'info_sets': len(self.strategy),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        
        if verbose:
            print(f"  Blueprint computed in {elapsed:.2f}s")
            print(f"  Exploitability: {self.exploitability:.6f}")
            print(f"  Game value (P0): {game_value:.6f}")
            print(f"  Info sets: {len(self.strategy)}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """Save blueprint to JSON file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        data = {
            'metadata': self.metadata,
            'strategy': self.strategy,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Blueprint saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Blueprint':
        """Load blueprint from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        bp = cls()
        bp.metadata = data['metadata']
        bp.strategy = data['strategy']
        bp.exploitability = bp.metadata.get('exploitability', float('inf'))
        
        return bp
    
    def get_action_strategy(self, info_set_key: str) -> Optional[Dict[str, float]]:
        """
        Look up the strategy for an information set.
        
        Returns dict mapping action_name → probability, or None if unknown.
        """
        return self.strategy.get(info_set_key)
    
    def get_reach_probability(
        self,
        info_set_key: str,
        action_name: str,
    ) -> float:
        """Get the probability of taking a specific action at an info set."""
        strat = self.strategy.get(info_set_key)
        if strat is None:
            return 0.0
        return strat.get(action_name, 0.0)
