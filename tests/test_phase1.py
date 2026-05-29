"""
Phase 1 Test Suite — Verification against known analytical results.

Kuhn Poker Nash Equilibrium (α = 1/3):
- Game value: -1/18 ≈ -0.0556 for Player 0
- Exploitability: 0
- Specific strategy probabilities verified against Wikipedia

Run: python -m pytest tests/test_phase1.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game_engine.kuhn_poker import KuhnPoker, KuhnState
from game_engine.game_state import Player, Action
from cfr.vanilla_cfr import VanillaCFR
from evaluation.best_response import BestResponse


def test_kuhn_state_transitions():
    """Verify Kuhn Poker game tree structure."""
    state = KuhnState(cards=(0, 1), history='')
    
    assert not state.is_terminal()
    assert not state.is_chance_node()
    assert state.current_player() == Player.PLAYER_0
    assert state.legal_actions() == [Action.CHECK, Action.BET]
    
    # Check → P1 to act
    s_check = state.apply_action(Action.CHECK)
    assert s_check.history == 'p'
    assert s_check.current_player() == Player.PLAYER_1
    
    # Check → Check → terminal (showdown)
    s_cc = s_check.apply_action(Action.CHECK)
    assert s_cc.history == 'pp'
    assert s_cc.is_terminal()
    # J(0) vs Q(1): P1 wins → P0 gets -1
    assert s_cc.terminal_utility(Player.PLAYER_0) == -1.0
    assert s_cc.terminal_utility(Player.PLAYER_1) == 1.0
    
    print("✅ test_kuhn_state_transitions passed")


def test_kuhn_terminal_utilities():
    """Verify all terminal utilities."""
    # Bet-Fold: P0 bet, P1 fold → P0 wins P1's ante
    s = KuhnState(cards=(0, 2), history='bf')
    assert s.is_terminal()
    assert s.terminal_utility(Player.PLAYER_0) == 1.0
    assert s.terminal_utility(Player.PLAYER_1) == -1.0
    
    # Bet-Call: P0 bet, P1 call → showdown with 2 each
    s = KuhnState(cards=(2, 0), history='bc', pot=[2, 2])
    assert s.is_terminal()
    assert s.terminal_utility(Player.PLAYER_0) == 2.0   # K > J
    assert s.terminal_utility(Player.PLAYER_1) == -2.0
    
    # Check-Bet-Fold: P0 checks, P1 bets, P0 folds
    s = KuhnState(cards=(0, 1), history='pbf')
    assert s.is_terminal()
    assert s.terminal_utility(Player.PLAYER_0) == -1.0
    assert s.terminal_utility(Player.PLAYER_1) == 1.0
    
    print("✅ test_kuhn_terminal_utilities passed")


def test_kuhn_info_sets():
    """Verify information set key computation."""
    # P0 with Jack at root
    s = KuhnState(cards=(0, 1), history='')
    assert s.information_set_key() == '0:'
    
    # P1 with Queen after P0 checks
    s = KuhnState(cards=(0, 1), history='p')
    assert s.information_set_key() == '1:p'
    
    # P0 with King facing bet after check
    s = KuhnState(cards=(2, 0), history='pb')
    assert s.information_set_key() == '2:pb'
    
    print("✅ test_kuhn_info_sets passed")


def test_nash_exploitability_zero():
    """The known Nash equilibrium should have zero exploitability."""
    game = KuhnPoker()
    br = BestResponse()
    nash = KuhnPoker.known_nash_equilibrium()
    
    exploit = br.compute_exploitability(game, nash)
    assert abs(exploit) < 1e-8, f"Nash exploitability should be ~0, got {exploit}"
    
    print("✅ test_nash_exploitability_zero passed")


def test_nash_game_value():
    """The known Nash equilibrium should give game value of -1/18."""
    game = KuhnPoker()
    br = BestResponse()
    nash = KuhnPoker.known_nash_equilibrium()
    
    gv = br.compute_game_value(game, nash, Player.PLAYER_0)
    expected = -1.0 / 18.0
    assert abs(gv - expected) < 1e-8, f"Game value should be {expected}, got {gv}"
    
    gv1 = br.compute_game_value(game, nash, Player.PLAYER_1)
    assert abs(gv1 + expected) < 1e-8, f"P1 game value should be {-expected}, got {gv1}"
    
    print("✅ test_nash_game_value passed")


def test_cfr_convergence():
    """CFR should converge to low exploitability on Kuhn Poker."""
    game = KuhnPoker()
    solver = VanillaCFR()
    
    metrics = solver.train_and_track(
        game, num_iterations=5000, eval_every=5000, verbose=False
    )
    
    final_exploit = metrics['exploitability'][-1]
    assert final_exploit < 0.05, f"Exploitability should be < 0.05, got {final_exploit}"
    
    # Game value should be close to -1/18
    avg_strat = solver.get_average_strategy()
    br = BestResponse()
    gv = br.compute_game_value(game, avg_strat, Player.PLAYER_0)
    assert abs(gv - (-1.0/18.0)) < 0.01, f"Game value should be ~-0.0556, got {gv}"
    
    print(f"✅ test_cfr_convergence passed (exploit={final_exploit:.6f}, gv={gv:.6f})")


def test_cfr_strategy_approaches_nash():
    """CFR average strategy should be close to known Nash values."""
    game = KuhnPoker()
    solver = VanillaCFR()
    solver.train(game.initial_state(), num_iterations=10000)
    
    avg = solver.get_average_strategy()
    nash = KuhnPoker.known_nash_equilibrium()
    
    max_error = 0
    for key in nash:
        if key in avg:
            for action in nash[key]:
                cfr_prob = avg[key].get(action, 0.0)
                nash_prob = nash[key].get(action, 0.0)
                error = abs(cfr_prob - nash_prob)
                max_error = max(max_error, error)
    
    # Note: Kuhn has a FAMILY of Nash equilibria parameterized by α ∈ [0, 1/3]
    # CFR may converge to a different member than α = 1/3, so we use a loose threshold
    assert max_error < 0.35, f"Max strategy error should be < 0.35, got {max_error}"
    
    print(f"✅ test_cfr_strategy_approaches_nash passed (max_error={max_error:.4f})")


def test_cfr_plus():
    """CFR+ should also converge."""
    game = KuhnPoker()
    solver = VanillaCFR(cfr_plus=True)
    
    metrics = solver.train_and_track(
        game, num_iterations=5000, eval_every=5000, verbose=False
    )
    
    final_exploit = metrics['exploitability'][-1]
    assert final_exploit < 0.05, f"CFR+ exploitability should be < 0.05, got {final_exploit}"
    
    print(f"✅ test_cfr_plus passed (exploit={final_exploit:.6f})")


def test_dcfr():
    """DCFR should also converge."""
    game = KuhnPoker()
    solver = VanillaCFR(dcfr_params={'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0})
    
    metrics = solver.train_and_track(
        game, num_iterations=5000, eval_every=5000, verbose=False
    )
    
    final_exploit = metrics['exploitability'][-1]
    assert final_exploit < 0.05, f"DCFR exploitability should be < 0.05, got {final_exploit}"
    
    print(f"✅ test_dcfr passed (exploit={final_exploit:.6f})")


if __name__ == '__main__':
    test_kuhn_state_transitions()
    test_kuhn_terminal_utilities()
    test_kuhn_info_sets()
    test_nash_exploitability_zero()
    test_nash_game_value()
    test_cfr_convergence()
    test_cfr_strategy_approaches_nash()
    test_cfr_plus()
    test_dcfr()
    
    print("\n" + "=" * 60)
    print("  ALL PHASE 1 TESTS PASSED ✅")
    print("=" * 60)
